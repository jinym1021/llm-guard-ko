"""Layer 2 — Korean semantic similarity scanner (output).

Detects LLM responses that are semantically close to known dangerous
content using sentence embeddings (jhgan/ko-sroberta-multitask).

Output-scanner contract: ``scan(prompt, output) -> (output, is_valid, risk_score)``.
The ``prompt`` is accepted for API consistency but not used — the risk
is assessed on the model's *response*, not the original question.

Requires the ``semantic`` extra::

    pip install llm-guard-ko[semantic]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_guard.input_scanners.korean_semantic import (
    DEFAULT_MODEL,
    DEFAULT_SEED_SENTENCES,
    DEFAULT_THRESHOLD,
    _cosine_similarity,
)

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer


class KoreanSemantic:
    """Scan Korean LLM responses for dangerous semantic content.

    Args:
        model_name: HuggingFace model ID for sentence embeddings.
            Defaults to ``jhgan/ko-sroberta-multitask``.
        threshold: Cosine similarity threshold above which a response is
            flagged. Range [0, 1]; higher = stricter. Defaults to 0.75.
        seed_sentences: Seed "dangerous content" sentences to compare
            against. Defaults to the same seeds as the input scanner.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        seed_sentences: list[str] | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._seeds = seed_sentences if seed_sentences is not None else DEFAULT_SEED_SENTENCES

        self._model: "SentenceTransformer | None" = None
        self._seed_embeddings: "np.ndarray | None" = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "KoreanSemantic requires the 'semantic' extra: "
                "pip install llm-guard-ko[semantic]"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        self._seed_embeddings = self._model.encode(
            self._seeds, convert_to_numpy=True, normalize_embeddings=False
        )

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan *output* (the LLM response) for dangerous semantic content.

        Args:
            prompt: Original user prompt (unused; accepted for protocol conformance).
            output: LLM response to check.

        Returns:
            ``(output, is_valid, risk_score)`` where ``is_valid=True``
            means safe and ``risk_score`` is in [0, 1].
        """
        import numpy as np

        self._ensure_model()
        assert self._model is not None and self._seed_embeddings is not None

        embedding: np.ndarray = self._model.encode(
            output, convert_to_numpy=True, normalize_embeddings=False
        )
        sims = _cosine_similarity(embedding, self._seed_embeddings)
        max_sim: float = float(np.max(sims))

        if max_sim >= self._threshold:
            return output, False, float(max_sim)
        return output, True, 0.0
