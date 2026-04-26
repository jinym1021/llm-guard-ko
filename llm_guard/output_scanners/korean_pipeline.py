"""3-layer escalating pipeline for Korean output (LLM responses).

Orchestrates Layer 1 → Layer 2 → Layer 3, where each layer only runs
if the previous layer flagged the text:

    Layer 1 (KoreanPII + KoreanToxicity + KoreanNoRefusal, regex/heuristic, ~ms)
         ↓ only if flagged
    Layer 2 (KoreanSensitive + KoreanSemantic, light AI, ~10–50 ms)
         ↓ only if still flagged
    Layer 3 (KoreanFactualConsistency? + KoreanLLMJudge, deep context/LLM, ~hundreds of ms)

If any later layer clears a flag from an earlier layer, the response is
treated as safe — higher layers are assumed more accurate. PII is always
redacted from the response regardless of final verdict.

KoreanFactualConsistency is OPTIONAL because it requires the caller to
configure a Korean NLI model explicitly. When ``factual_consistency`` is
None (the default), Layer 3 runs only the LLM judge.

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
"""

from __future__ import annotations

from llm_guard.output_scanners.korean_factual_consistency import KoreanFactualConsistency
from llm_guard.output_scanners.korean_llm_judge import KoreanLLMJudge
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_semantic import KoreanSemantic
from llm_guard.output_scanners.korean_sensitive import KoreanSensitive
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity


class KoreanPipeline:
    """3-layer escalating scanner for Korean LLM responses.

    Args:
        pii: Layer 1 PII scanner. Defaults to ``KoreanPII()``.
        toxicity: Layer 1 Toxicity scanner. Defaults to ``KoreanToxicity()``.
        no_refusal: Layer 1 No-Refusal scanner. Defaults to ``KoreanNoRefusal()``.
        sensitive: Layer 2 Sensitive (NER) scanner. Defaults to ``KoreanSensitive()``.
        semantic: Layer 2 Semantic similarity scanner. Defaults to ``KoreanSemantic()``.
        factual_consistency: Optional Layer 3 NLI scanner. Skipped when ``None``.
        llm_judge: Layer 3 local-LLM judge. Defaults to ``KoreanLLMJudge()``.
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
        self._pii = pii if pii is not None else KoreanPII()
        self._toxicity = toxicity if toxicity is not None else KoreanToxicity()
        self._no_refusal = no_refusal if no_refusal is not None else KoreanNoRefusal()

        self._sensitive = sensitive if sensitive is not None else KoreanSensitive()
        self._semantic = semantic if semantic is not None else KoreanSemantic()

        # Optional: requires explicit NLI model_name, so don't auto-construct.
        self._factual_consistency = factual_consistency
        self._llm_judge = llm_judge if llm_judge is not None else KoreanLLMJudge()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan *output* through the escalating Korean output pipeline.

        Returns:
            ``(sanitized_output, is_valid, risk_score)``.
            *sanitized_output* always has Layer-1 PII redacted.
            ``risk_score=0.0`` when safe; otherwise the max risk reported
            by the layer that produced the final unsafe verdict.
        """
        # --- Layer 1: Regex/Substring (Fast) — PII redacts in place ---
        sanitized, pii_valid, pii_risk = self._pii.scan(prompt, output)
        _, tox_valid, tox_risk = self._toxicity.scan(prompt, sanitized)
        _, ref_valid, ref_risk = self._no_refusal.scan(prompt, sanitized)

        if pii_valid and tox_valid and ref_valid:
            return sanitized, True, 0.0

        # --- Layer 2: Light AI (NER/Embedding) ---
        _, sens_valid, sens_risk = self._sensitive.scan(prompt, sanitized)
        _, sem_valid, sem_risk = self._semantic.scan(prompt, sanitized)

        if sens_valid and sem_valid:
            return sanitized, True, 0.0

        # --- Layer 3: Deep context (NLI + LLM judge) ---
        if self._factual_consistency is not None:
            _, fac_valid, fac_risk = self._factual_consistency.scan(prompt, sanitized)
        else:
            fac_valid, fac_risk = True, 0.0

        _, judge_valid, judge_risk = self._llm_judge.scan(prompt, sanitized)

        if fac_valid and judge_valid:
            return sanitized, True, 0.0

        return sanitized, False, max(fac_risk, judge_risk)
