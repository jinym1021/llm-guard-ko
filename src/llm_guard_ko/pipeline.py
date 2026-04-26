"""Three-layer Korean guardrail pipeline (NOT YET IMPLEMENTED).

Combines KoreanPII (regex) → KoreanSemantic (embeddings) →
KoreanLLMJudge (LLM) with escalation: each layer only runs when the
prior layer has flagged the text as suspicious. The pipeline itself
implements the Scanner protocol so it can be dropped into any
llm-guard ``scan_prompt(scanners, ...)`` call as a single scanner.

Ships in 0.2.0 alongside KoreanSemantic.
"""

from __future__ import annotations


class KoreanPipeline:
    """Placeholder — ships in 0.2.0."""

    def __init__(self, *, use_llm_judge: bool = False) -> None:
        self._use_llm_judge = use_llm_judge

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        raise NotImplementedError("KoreanPipeline is scheduled for 0.2.0.")
