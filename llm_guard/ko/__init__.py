"""llm-guard-ko — Korean-language guardrails for llm-guard.

Public API (Phase 1):
    KoreanPII        — Layer 1 PII regex scanner (with redaction)
    KoreanInjection  — Layer 1 prompt-injection / jailbreak regex scanner

Ships as a subpackage of ``llm_guard`` so it integrates with upstream
``scan_prompt`` / ``scan_output`` without modification::

    from llm_guard import scan_prompt
    from llm_guard.ko import KoreanPII, KoreanInjection

    sanitized, valid, risk = scan_prompt(
        [KoreanPII(), KoreanInjection()],
        prompt,
    )

Later phases will add KoreanSemantic (Layer 2), KoreanContentFilter
(Layer 3, SGuard-based), and KoreanPipeline (escalation orchestrator).
"""

from __future__ import annotations

from llm_guard.ko.injection import KoreanInjection
from llm_guard.ko.pii import KoreanPII

__version__ = "0.1.0"

__all__ = [
    "KoreanPII",
    "KoreanInjection",
    "__version__",
]
