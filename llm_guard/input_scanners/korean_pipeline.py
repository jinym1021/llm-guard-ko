"""3-layer escalating pipeline for Korean input (prompts).

Orchestrates Layer 1 → Layer 2 → Layer 3, where each layer only runs
if the previous layer flagged the text:

    Layer 1 (KoreanPII + KoreanInjection, regex/heuristic, ~ms)
         ↓ only if flagged
    Layer 2 (KoreanSemantic, sentence-embedding, ~10–50 ms)
         ↓ only if still flagged
    Layer 3 (KoreanContentFilter, SGuard classifier, ~hundreds of ms)

If any later layer clears a flag from an earlier layer, the text is
treated as safe — higher layers are assumed more accurate. PII is
always redacted before further processing regardless of final verdict.

Escalation contract (matches llm-guard's Scanner protocol):
    scan(prompt) -> (sanitized_text, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe
    risk_score == 0.0 when safe; == content-filter max prob when unsafe.
"""

from __future__ import annotations

from llm_guard.input_scanners.korean_content_filter import (
    DEFAULT_THRESHOLD as DEFAULT_CF_THRESHOLD,
    KoreanContentFilter,
)
from llm_guard.input_scanners.korean_injection import KoreanInjection
from llm_guard.input_scanners.korean_pii import KoreanPII
from llm_guard.input_scanners.korean_semantic import (
    DEFAULT_THRESHOLD as DEFAULT_SEM_THRESHOLD,
    KoreanSemantic,
)


class KoreanPipeline:
    """3-layer escalating scanner for Korean prompts.

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

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Scan *prompt* through the 3-layer Korean pipeline.

        Returns:
            ``(sanitized, is_valid, risk_score)``.
            *sanitized* always has PII redacted.
            ``is_valid=True`` means safe; ``risk_score=0.0`` when safe.
        """
        # Layer 1 — regex/heuristic (always runs)
        sanitized, pii_valid, _ = self._pii.scan(prompt)
        _, inj_valid, _ = self._injection.scan(sanitized)

        if pii_valid and inj_valid:
            return sanitized, True, 0.0

        # Layer 2 — semantic similarity (only reached if Layer 1 flagged)
        _, sem_valid, _ = self._semantic.scan(sanitized)
        if sem_valid:
            return sanitized, True, 0.0

        # Layer 3 — SGuard classifier (only reached if Layer 2 also flagged)
        _, cf_valid, cf_risk = self._content_filter.scan(sanitized)
        if cf_valid:
            return sanitized, True, 0.0

        return sanitized, False, cf_risk


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run Phase 1 (Layer 1) of the Korean Guardrail Pipeline.")
    parser.add_argument("prompt", help="The text to scan.")
    parser.add_argument("--no-redact", action="store_true", help="Don't redact PII, only flag it.")
    args = parser.parse_args()

    # Phase 1 consists of PII and Injection scanners.
    pii_scanner = KoreanPII(redact=not args.no_redact)
    inj_scanner = KoreanInjection()

    print(f"\n--- Phase 1 (Layer 1) Scan ---")
    print(f"Input: {args.prompt}")

    sanitized, pii_valid, _ = pii_scanner.scan(args.prompt)
    _, inj_valid, _ = inj_scanner.scan(sanitized)

    is_valid = pii_valid and inj_valid

    print(f"Valid: {is_valid}")
    if not is_valid:
        reasons = []
        if not pii_valid:
            reasons.append("PII detected")
        if not inj_valid:
            reasons.append("Injection detected")
        print(f"Reasons: {', '.join(reasons)}")

    print(f"Sanitized: {sanitized}")
    print(f"------------------------------\n")

    if not is_valid:
        sys.exit(1)
