"""Layer 2 — Korean semantic similarity scanner (output).

Detects model outputs that are semantically close to known dangerous
or undesired responses using sentence embeddings (jhgan/ko-sroberta-multitask).
Complements Layer 1 regex by catching paraphrases.

Requires the ``semantic`` extra::

    pip install llm-guard-ko[semantic]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_guard.patterns.korean import KOREAN_TOXIC_PATTERNS, KOREAN_REFUSAL_PATTERNS

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_THRESHOLD = 0.75

# Output Scanner 측면에서 잡아내야 할 "위험한/부적절한 출력" 시드 문장들 (Toxicity + Refusal)
DEFAULT_SEED_SENTENCES: list[str] = [
    # 강력한 비속어나 모욕적인 문맥 (Toxicity - context)
    "당신은 정말 쓸모없는 사람입니다.",
    "죽어버렸으면 좋겠어.",
    "그런 멍청한 질문은 하지 마세요.",
    "이런 벌레같은 자식아",
    # 심각한 모델 거절/윤리적 회피 뉘앙스 (Refusal - paraphrase)
    "해당 질문은 인공지능 윤리 정책상 적절하지 않아 답변할 수 없습니다.",
    "저는 언어 모델이므로 그런 주관적이거나 위험한 판단을 내릴 수 없습니다.",
    "안전상의 이유로 그 요청은 처리해 드릴 수 없음을 양해 바랍니다.",
]


def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Return cosine similarities between vector a and each row of b."""
    import numpy as np

    norm_a = np.linalg.norm(a)
    norms_b = np.linalg.norm(b, axis=1)
    if norm_a == 0 or np.any(norms_b == 0):
        return np.zeros(len(b))
    return (b @ a) / (norms_b * norm_a)


class KoreanSemantic:
    """Scan Korean outputs for dangerous intent via sentence similarity.

    Args:
        model_name: HuggingFace model ID for sentence embeddings.
            Defaults to ``jhgan/ko-sroberta-multitask``.
        threshold: Cosine similarity threshold above which an output is
            flagged. Range [0, 1]; higher = stricter. Defaults to 0.75.
        seed_sentences: Seed "dangerous output" sentences to compare
            against. Defaults to combined toxicity and refusal seeds.
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

        # Lazy-initialised on first scan call.
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
        """Scan *output* for dangerous semantic similarity.

        Returns:
            ``(output, is_valid, risk_score)`` where ``is_valid=True``
            means safe and ``risk_score`` is in [0, 1].
        """
        import numpy as np

        if not output.strip():
            return output, True, 0.0

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
