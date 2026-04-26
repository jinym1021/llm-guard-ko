"""Layer 1, Layer 2, and Layer 3 — Korean NoRefusal scanner for outputs.

Detects phrases commonly used by Korean models when refusing to answer
a prompt due to policy or safety guidelines.

- Layer 1: Substring matching for fast detection of common refusal phrases.
- Layer 2: Semantic embedding similarity to catch paraphrased refusals.
- Layer 3: LLM-based verification for deep contextual refusals.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Callable

from llm_guard.patterns.korean import KOREAN_REFUSAL_PATTERNS
from llm_guard.util import get_logger

LOGGER = get_logger()

def _cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Return cosine similarities between vector a and each row of b."""
    import numpy as np
    norm_a = np.linalg.norm(a)
    norms_b = np.linalg.norm(b, axis=1)
    if norm_a == 0 or np.any(norms_b == 0):
        return np.zeros(len(b))
    return (b @ a) / (norms_b * norm_a)

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_EMBEDDING_THRESHOLD = 0.75
DEFAULT_LLM_THRESHOLD = 0.5

DEFAULT_REFUSAL_SEEDS = [
    "해당 질문은 인공지능 윤리 정책상 적절하지 않아 답변할 수 없습니다.",
    "저는 언어 모델이므로 그런 주관적이거나 위험한 판단을 내릴 수 없습니다.",
    "안전상의 이유로 그 요청은 처리해 드릴 수 없음을 양해 바랍니다.",
]


class KoreanNoRefusal:
    """Scan Korean output text for refusal patterns.
    
    Escalates from Substring Matching (Layer 1) -> Semantic Embedding (Layer 2) -> LLM Judge (Layer 3).
    """

    def __init__(
        self,
        *,
        patterns: list[str] | None = None,
        use_layer2: bool = True,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
        seed_sentences: list[str] | None = None,
        use_layer3: bool = False,
        llm_api_func: Callable[[str], float] | None = None,
        llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    ) -> None:
        # Layer 1
        self._patterns = patterns if patterns is not None else KOREAN_REFUSAL_PATTERNS
        self._patterns_lower = [p.lower() for p in self._patterns]
        
        # Layer 2
        self._use_layer2 = use_layer2
        self._embedding_model_name = embedding_model_name
        self._embedding_threshold = embedding_threshold
        self._seeds = seed_sentences if seed_sentences is not None else DEFAULT_REFUSAL_SEEDS
        
        self._model: "SentenceTransformer | None" = None
        self._seed_embeddings: "np.ndarray | None" = None
        
        # Layer 3
        self._use_layer3 = use_layer3
        self._llm_api_func = llm_api_func
        self._llm_threshold = llm_threshold

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "KoreanNoRefusal Layer 2 requires the 'semantic' extra: "
                "pip install llm-guard-ko[semantic]"
            ) from exc
        self._model = SentenceTransformer(self._embedding_model_name)
        self._seed_embeddings = self._model.encode(
            self._seeds, convert_to_numpy=True, normalize_embeddings=False
        )

    def _scan_layer3(self, output: str) -> float:
        if self._llm_api_func is None:
            LOGGER.warning("Layer 3 is enabled but no llm_api_func was provided. Skipping.")
            return 0.0
            
        prompt = f"다음 한국어 문장이 AI/언어 모델로서 사용자의 요청을 거절하거나 답변을 회피하는 의도를 담고 있는지 0.0에서 1.0 사이의 실수로만 대답하시오.\n\n문장: '{output}'\n\n점수:"
        
        try:
            score = self._llm_api_func(prompt)
            return float(score)
        except Exception as e:
            LOGGER.error(f"LLM API call failed during Layer 3 scan: {e}")
            return 0.0

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        output_lower = output.lower()

        # --- Layer 1: Substring Match ---
        for pattern in self._patterns_lower:
            if pattern in output_lower:
                return output, False, 1.0

        # --- Layer 2: Semantic Similarity ---
        if self._use_layer2:
            import numpy as np
            
            self._ensure_model()
            assert self._model is not None and self._seed_embeddings is not None

            embedding: np.ndarray = self._model.encode(
                output, convert_to_numpy=True, normalize_embeddings=False
            )
            sims = _cosine_similarity(embedding, self._seed_embeddings)
            max_sim: float = float(np.max(sims))

            if max_sim >= self._embedding_threshold:
                return output, False, float(max_sim)

        # --- Layer 3: LLM Scan ---
        if self._use_layer3:
            llm_risk = self._scan_layer3(output)
            if llm_risk >= self._llm_threshold:
                return output, False, llm_risk

        return output, True, 0.0
