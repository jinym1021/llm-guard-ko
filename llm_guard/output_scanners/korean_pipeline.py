"""3-layer escalating pipeline for Korean output (LLM responses).

Orchestrates Layer 1 → Layer 2 → Layer 3, where each layer only runs
if the previous layer flagged the text:

    Layer 1 (KoreanPII + KoreanInjection, regex/heuristic, ~ms)
         ↓ only if flagged
    Layer 2 (KoreanSemantic, sentence-embedding, ~10–50 ms)
         ↓ only if still flagged
    Layer 3 (KoreanContentFilter, SGuard classifier, ~hundreds of ms)

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
The original prompt is passed to each layer for protocol conformance;
PII and injection scanners use it for context. KoreanSemantic and
KoreanContentFilter output scanners ignore it and classify the response.
PII is always redacted from the response regardless of final verdict.
"""

from __future__ import annotations

from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_sensitive import KoreanSensitive
from llm_guard.output_scanners.korean_llm_judge import KoreanLLMJudge
from llm_guard.output_scanners.korean_factual_consistency import KoreanFactualConsistency
from llm_guard.input_scanners.korean_content_filter import DEFAULT_THRESHOLD as DEFAULT_CF_THRESHOLD
from llm_guard.input_scanners.korean_semantic import DEFAULT_THRESHOLD as DEFAULT_SEM_THRESHOLD
from llm_guard.output_scanners.korean_content_filter import KoreanContentFilter
from llm_guard.output_scanners.korean_injection import KoreanInjection
from llm_guard.output_scanners.korean_semantic import KoreanSemantic


class KoreanPipeline:
    """3-layer escalating scanner for Korean LLM responses.

    Args:
        pii: Pre-built Layer 1 PII scanner. Defaults to ``KoreanPII()``.
        injection: Pre-built Layer 1 injection scanner. Defaults to
            ``KoreanInjection()``.
        semantic: Pre-built Layer 2 semantic scanner. Defaults to
            ``KoreanSemantic(threshold=semantic_threshold)``.
        content_filter: Pre-built Layer 3 content-filter scanner.
            Defaults to ``KoreanContentFilter(threshold=cf_threshold)``.
        semantic_threshold: Threshold forwarded to the default
            ``KoreanSemantic`` instance (ignored if *semantic* is given).
        cf_threshold: Threshold forwarded to the default
            ``KoreanContentFilter`` instance (ignored if *content_filter*
            is given).
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
        # Layer 3 (LLM)
        llm_judge: KoreanLLMJudge | None = None,
        factual_consistency: KoreanFactualConsistency | None = None,
    ) -> None:
        # Layer 1
        self._pii = pii if pii is not None else KoreanPII()
        self._toxicity = toxicity if toxicity is not None else KoreanToxicity()
        self._no_refusal = no_refusal if no_refusal is not None else KoreanNoRefusal()
        
        # Layer 2
        self._sensitive = sensitive if sensitive is not None else KoreanSensitive()
        self._semantic = semantic if semantic is not None else KoreanSemantic()
        
        # Layer 3
        self._llm_judge = llm_judge if llm_judge is not None else KoreanLLMJudge()
        self._factual_consistency = factual_consistency if factual_consistency is not None else KoreanFactualConsistency()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan *output* (LLM response) through the escalating Korean pipeline."""
        
        current_output = output

        # --- Layer 1: Regex/Substring ---
        current_output, pii_valid, pii_risk = self._pii.scan(prompt, current_output)
        if not pii_valid: return current_output, False, pii_risk
        
        current_output, tox_valid, tox_risk = self._toxicity.scan(prompt, current_output)
        if not tox_valid: return current_output, False, tox_risk
        
        current_output, ref_valid, ref_risk = self._no_refusal.scan(prompt, current_output)
        if not ref_valid: return current_output, False, ref_risk


        # --- Layer 2: Light AI (NER/Embedding) ---
        current_output, sens_valid, sens_risk = self._sensitive.scan(prompt, current_output)
        if not sens_valid: return current_output, False, sens_risk
        
        current_output, sem_valid, sem_risk = self._semantic.scan(prompt, current_output)
        if not sem_valid: return current_output, False, sem_risk


        # --- Layer 3: LLM Judge & Deep Context ---
        current_output, fac_valid, fac_risk = self._factual_consistency.scan(prompt, current_output)
        if not fac_valid: return current_output, False, fac_risk

        current_output, judge_valid, judge_risk = self._llm_judge.scan(prompt, current_output)
        if not judge_valid: return current_output, False, judge_risk

        return current_output, True, 0.0
        pii: KoreanPII | None = None,
        injection: KoreanInjection | None = None,
        semantic: KoreanSemantic | None = None,
        content_filter: KoreanContentFilter | None = None,
        semantic_threshold: float = DEFAULT_SEM_THRESHOLD,
        cf_threshold: float = DEFAULT_CF_THRESHOLD,
    ) -> None:
        self._pii = pii if pii is not None else KoreanPII()
        self._injection = injection if injection is not None else KoreanInjection()
        self._semantic = (
            semantic if semantic is not None else KoreanSemantic(threshold=semantic_threshold)
        )
        self._content_filter = (
            content_filter
            if content_filter is not None
            else KoreanContentFilter(threshold=cf_threshold)
        )

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan *output* (LLM response) through the 3-layer Korean pipeline.

        Args:
            prompt: Original user prompt, forwarded to each sub-scanner for
                protocol conformance.
            output: LLM response to classify.

        Returns:
            ``(sanitized, is_valid, risk_score)``.
            *sanitized* always has PII redacted from the response.
            ``is_valid=True`` means safe; ``risk_score=0.0`` when safe.
        """
        # Layer 1 — regex/heuristic (always runs)
        sanitized, pii_valid, _ = self._pii.scan(prompt, output)
        _, inj_valid, _ = self._injection.scan(prompt, sanitized)

        if pii_valid and inj_valid:
            return sanitized, True, 0.0

        # Layer 2 — semantic similarity (only reached if Layer 1 flagged)
        _, sem_valid, _ = self._semantic.scan(prompt, sanitized)
        if sem_valid:
            return sanitized, True, 0.0

        # Layer 3 — SGuard classifier (only reached if Layer 2 also flagged)
        _, cf_valid, cf_risk = self._content_filter.scan(prompt, sanitized)
        if cf_valid:
            return sanitized, True, 0.0

        return sanitized, False, cf_risk
