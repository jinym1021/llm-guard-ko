"""3-layer escalating pipeline for Korean output (LLM responses).

Orchestrates Layer 1 → Layer 2 → Layer 3, where each layer only runs
if the previous layer cleared the text, acting as a fallback to catch what was missed:

    Layer 1 (KoreanPII, KoreanToxicity, KoreanNoRefusal - Regex/Heuristics)
         ↓ if cleared
    Layer 2 (KoreanSensitive, KoreanSemantic - NER/Embeddings)
         ↓ if cleared
    Layer 3 (KoreanFactualConsistency, KoreanLLMJudge - NLI/LLM API)

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
The original prompt is passed to each layer for protocol conformance.
"""

from __future__ import annotations

from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_sensitive import KoreanSensitive
from llm_guard.output_scanners.korean_semantic import KoreanSemantic
from llm_guard.output_scanners.korean_llm_judge import KoreanLLMJudge
from llm_guard.output_scanners.korean_factual_consistency import KoreanFactualConsistency

class KoreanPipeline:
    """3-layer escalating scanner for Korean LLM responses."""

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
