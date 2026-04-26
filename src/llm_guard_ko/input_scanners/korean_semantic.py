"""Layer 2 — Korean semantic-similarity scanner (NOT YET IMPLEMENTED).

Planned implementation uses ``sentence-transformers`` with
``jhgan/ko-sroberta-multitask``. Seed sentences representing dangerous
intents are embedded once; incoming prompts are scored against them
and flagged when cosine similarity exceeds a threshold.

Install with::

    pip install "llm-guard-ko[semantic]"

Usage (future)::

    scanner = KoreanSemantic(threshold=0.75)
    sanitized, is_valid, risk = scanner.scan(prompt)
"""

from __future__ import annotations


class KoreanSemantic:
    """Placeholder — ships in 0.2.0."""

    def __init__(
        self,
        *,
        model_name: str = "jhgan/ko-sroberta-multitask",
        threshold: float = 0.75,
        seeds: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._seeds = seeds

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        raise NotImplementedError(
            "KoreanSemantic is scheduled for 0.2.0. Install the "
            "[semantic] extra and watch the changelog."
        )
