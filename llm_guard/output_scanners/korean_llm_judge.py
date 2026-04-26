"""Layer 3 — Local LLM-as-judge scanner for outputs.

Uses a lightweight, locally hosted LLM (e.g. 1B~3B parameter instruction-tuned models)
to perform deep contextual safety analysis. This guarantees that sensitive outputs
never leave the local environment while still benefiting from LLM reasoning.

Requires the ``llm-judge`` extra::

    pip install "llm-guard-ko[llm-judge]"
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from transformers import Pipeline

DEFAULT_LOCAL_LLM = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

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
    """Scan Korean outputs using a Local Lightweight LLM for deep contextual threats."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_LOCAL_LLM,
        fail_open: bool = True,
        llm_callable: Callable[[str], str] | None = None,
        device: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name
        self._fail_open = fail_open
        self._llm_callable = llm_callable
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._pipeline: Pipeline | None = None

    def _ensure_model(self) -> None:
        # If user provided a custom callable, we don't load the local model
        if self._llm_callable is not None or self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "KoreanLLMJudge requires the 'llm-judge' extra: "
                "pip install llm-guard-ko[llm-judge]"
            ) from exc

        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        device_id = -1 if self._device == "cpu" else int(self._device.split(":")[-1]) if ":" in self._device else 0

        # Load local instruction-tuned LLM (optimized for EXAONE 3.5 or similar)
        # Note: torch_dtype=auto handles the best available precision automatically
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_name,
            device=device_id,
            torch_dtype="auto",
            trust_remote_code=self._trust_remote_code,
        )

    def _generate_local(self, prompt: str) -> str:
        if self._pipeline is None:
            return '{"is_flagged": false}'
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_KO},
            {"role": "user", "content": f"응답 텍스트: {prompt}"}
        ]
        
        out = self._pipeline(
            messages,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
        )
        # Extract generated text from pipeline output
        try:
            # transformers text-generation pipeline with chat template returns list of dicts
            generated_text = out[0]["generated_text"][-1]["content"] # type: ignore
            return str(generated_text)
        except Exception:
            # Fallback parsing
            return str(out[0].get("generated_text", "")) # type: ignore

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        self._ensure_model()
        
        try:
            if self._llm_callable is not None:
                result_str = self._llm_callable(f"{SYSTEM_PROMPT_KO}\n\n응답 텍스트: {output}")
            else:
                result_str = self._generate_local(output)
                
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
