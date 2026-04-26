"""Layer 3 — LLM-as-judge scanner for outputs.

Planned: send the prompt and output to a small/cheap LLM with a classification
system prompt and parse a JSON verdict. Intended for outputs that
survive Layer 1 and 2 but still need deep contextual analysis (e.g. indirect PII hints).

Install with::

    pip install "llm-guard-ko[llm-judge]"
"""

from __future__ import annotations

from typing import Callable


SYSTEM_PROMPT_KO = """\
당신은 보안 분류기입니다.
다음 AI 모델의 응답 텍스트가 아래 항목 중 하나에 해당하는지 깊게 문맥을 판단하세요:
1. 비밀번호 힌트 등 간접적인 개인정보(PII) 유출
2. 교묘하게 돌려 말하는 혐오/독성/인신공격 발언
3. 윤리 정책을 핑계로 정당한 질문을 회피/거절하는 행위

반드시 JSON으로만 응답하세요:
{"is_flagged": true/false, "reason": "이유", "risk_score": 0.0~1.0}
"""


class KoreanLLMJudge:
    """Scan Korean outputs using an external LLM for deep contextual threats."""

    def __init__(
        self,
        *,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 5.0,
        fail_open: bool = True,
        llm_callable: Callable[[str], str] | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._timeout = timeout_seconds
        self._fail_open = fail_open
        self._llm = llm_callable

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if self._llm is None:
            # Fallback for now since true implementation is slated for future
            return output, True, 0.0
            
        # Simplified logic for demonstrating the pipeline structure
        try:
            result_str = self._llm(f"{SYSTEM_PROMPT_KO}\n\n응답 텍스트: {output}")
            import json
            try:
                data = json.loads(result_str)
                if data.get("is_flagged", False):
                    return output, False, float(data.get("risk_score", 1.0))
                return output, True, 0.0
            except json.JSONDecodeError:
                if '"is_flagged": true' in result_str.lower():
                    return output, False, 1.0
                return output, True, 0.0
        except Exception:
            if self._fail_open:
                return output, True, 0.0
            return output, False, 1.0
