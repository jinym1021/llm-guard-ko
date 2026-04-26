"""Layer 3 — Korean content filter (output).

Classifies LLM responses against 5 MLCommons safety categories using
the SGuard-ContentFilter-2B-v1 local classifier. Intended to run only
after Layer 1 and Layer 2 have flagged the response (escalation contract).

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
The ``prompt`` is accepted for API consistency but not used — risk is
assessed on the model's *response*.

Requires the ``content-filter`` extra::

    pip install llm-guard-ko[content-filter]
"""

from __future__ import annotations

from llm_guard.input_scanners.korean_content_filter import (
    CATEGORIES,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    KoreanContentFilter as _InputFilter,
)


class KoreanContentFilter:
    """Scan Korean LLM responses using the SGuard local content-filter model.

    Each of the 5 MLCommons categories is evaluated independently.
    A response is flagged if any category's unsafe probability exceeds
    ``threshold``.

    Args:
        model_name: HuggingFace model ID. Defaults to
            ``SamsungSDS-Research/SGuard-ContentFilter-2B-v1``.
        threshold: Per-category unsafe-probability threshold [0, 1].
            Defaults to 0.5.
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).
            ``None`` auto-selects CUDA if available, else CPU.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self._inner = _InputFilter(
            model_name=model_name,
            threshold=threshold,
            device=device,
        )

    def scan_detailed(self, output: str) -> dict[str, dict]:
        """Return per-category classification detail for *output*.

        Returns:
            Mapping of category name → ``{"unsafe": bool, "prob": float}``.
        """
        return self._inner.scan_detailed(output)

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan the LLM *output* for unsafe content.

        Args:
            prompt: Original user prompt (unused; accepted for protocol conformance).
            output: LLM response to classify.

        Returns:
            ``(output, is_valid, risk_score)`` where ``is_valid=True``
            means safe and ``risk_score`` is the max unsafe probability
            across all categories (0.0 when safe).
        """
        _, is_valid, risk_score = self._inner.scan(output)
        return output, is_valid, risk_score
