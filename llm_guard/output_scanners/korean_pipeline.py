"""3-layer escalating pipeline for Korean output (LLM responses).

Orchestrates Layer 1 → Layer 2 → Layer 3, where each layer only runs
if the previous layer flagged the text:

    Layer 1 (KoreanPII + KoreanToxicity + KoreanNoRefusal, regex/heuristic, ~ms)
         ↓ only if flagged
    Layer 2 (KoreanSensitive + KoreanSemantic, light AI, ~10–50 ms)
         ↓ only if still flagged
    Layer 3 (KoreanFactualConsistency + KoreanLLMJudge, deep context/LLM, ~hundreds of ms)

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
The original prompt is passed to each layer for protocol conformance.
PII is always redacted from the response regardless of final verdict.
"""

from __future__ import annotations

from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_sensitive import KoreanSensitive
from llm_guard.output_scanners.korean_semantic import KoreanSemantic
from llm_guard.output_scanners.korean_factual_consistency import KoreanFactualConsistency
from llm_guard.output_scanners.korean_llm_judge import KoreanLLMJudge


class KoreanPipeline:
    """3-layer escalating scanner for Korean LLM responses.

    Args:
        pii: Layer 1 PII scanner.
        toxicity: Layer 1 Toxicity scanner.
        no_refusal: Layer 1 No-Refusal scanner.
        sensitive: Layer 2 Sensitive information scanner.
        semantic: Layer 2 Semantic similarity scanner.
        factual_consistency: Layer 3 Factual consistency scanner.
        llm_judge: Layer 3 LLM-based judge.
    """

    def __init__(
        self,
        *,
        # Layer 1
        pii: KoreanPII | None = None,
        toxicity: KoreanToxicity | None = None,
        no_refusal: KoreanNoRefusal | None = None,
        # Layer 2
        sensitive: KoreanSensitive | None = None,
        semantic: KoreanSemantic | None = None,
        # Layer 3
        factual_consistency: KoreanFactualConsistency | None = None,
        llm_judge: KoreanLLMJudge | None = None,
    ) -> None:
        # Layer 1
        self._pii = pii if pii is not None else KoreanPII()
        self._toxicity = toxicity if toxicity is not None else KoreanToxicity()
        self._no_refusal = no_refusal if no_refusal is not None else KoreanNoRefusal()

        # Layer 2
        self._sensitive = sensitive if sensitive is not None else KoreanSensitive()
        self._semantic = semantic if semantic is not None else KoreanSemantic()

        # Layer 3
        self._factual_consistency = (
            factual_consistency if factual_consistency is not None else KoreanFactualConsistency()
        )
        self._llm_judge = llm_judge if llm_judge is not None else KoreanLLMJudge()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan *output* (LLM response) through the escalating Korean pipeline.

        Returns:
            ``(sanitized_output, is_valid, risk_score)``.
            *sanitized_output* always has PII redacted.
        """
        current_output = output

        # --- Layer 1: Regex/Substring (Fast) ---
        current_output, pii_valid, pii_risk = self._pii.scan(prompt, current_output)
        _, tox_valid, tox_risk = self._toxicity.scan(prompt, current_output)
        _, ref_valid, ref_risk = self._no_refusal.scan(prompt, current_output)

        # PII is special because it redacts; others just flag.
        l1_valid = pii_valid and tox_valid and ref_valid
        l1_risk = max(pii_risk, tox_risk, ref_risk)

        if l1_valid:
            return current_output, True, 0.0

        # --- Layer 2: Light AI (NER/Embedding) ---
        _, sens_valid, sens_risk = self._sensitive.scan(prompt, current_output)
        _, sem_valid, sem_risk = self._semantic.scan(prompt, current_output)

        l2_valid = sens_valid and sem_valid
        l2_risk = max(sens_risk, sem_risk)

        if l2_valid:
            return current_output, True, 0.0

        # --- Layer 3: LLM Judge & Deep Context (Slow) ---
        _, fac_valid, fac_risk = self._factual_consistency.scan(prompt, current_output)
        _, judge_valid, judge_risk = self._llm_judge.scan(prompt, current_output)

        l3_valid = fac_valid and judge_valid
        l3_risk = max(fac_risk, judge_risk)

        if l3_valid:
            return current_output, True, 0.0

        return current_output, False, l3_risk
