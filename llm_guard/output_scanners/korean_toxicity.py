"""Layer 1, Layer 2, and Layer 3 — Korean Toxicity scanner for outputs.

- Layer 1: Regex patterns to catch evasive slang/profanity.
- Layer 2: Sentence embedding similarity against a list of toxic seed sentences.
- Layer 3: LLM-based verification for deep contextual toxicity.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Pattern, Callable

from llm_guard.patterns.korean import KOREAN_TOXIC_PATTERNS
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

DEFAULT_TOXIC_SEEDS = [
    "당신은 정말 쓸모없는 사람입니다.",
    "죽어버렸으면 좋겠어.",
    "그런 멍청한 질문은 하지 마세요.",
    "이런 벌레같은 자식아",
]


class KoreanToxicity:
    """Scan Korean output text for toxic words/profanity.
    
    Escalates from Regex (Layer 1) -> Semantic Embedding (Layer 2) -> LLM Judge (Layer 3).
    """

    def __init__(
        self,
        *,
        redact: bool = False,
        patterns: list[str] | None = None,
        use_layer2: bool = True,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
        seed_sentences: list[str] | None = None,
        use_layer3: bool = False,
        llm_api_func: Callable[[str], float] | None = None,
        llm_threshold: float = DEFAULT_LLM_THRESHOLD,
    ) -> None:
        self._redact = redact
        
        # Layer 1
        source = patterns if patterns is not None else KOREAN_TOXIC_PATTERNS
        self._compiled: list[Pattern[str]] = [re.compile(p) for p in source]
        
        # Layer 2
        self._use_layer2 = use_layer2
        self._embedding_model_name = embedding_model_name
        self._embedding_threshold = embedding_threshold
        self._seeds = seed_sentences if seed_sentences is not None else DEFAULT_TOXIC_SEEDS
        
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
                "KoreanToxicity Layer 2 requires the 'semantic' extra: "
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
            
        prompt = f"다음 한국어 문장에 혐오 표현, 비속어, 인신공격, 또는 유해한 의도가 포함되어 있는지 0.0에서 1.0 사이의 실수로만 대답하시오.\n\n문장: '{output}'\n\n점수:"
        
        try:
            score = self._llm_api_func(prompt)
            return float(score)
        except Exception as e:
            LOGGER.error(f"LLM API call failed during Layer 3 scan: {e}")
            return 0.0

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        spans: list[tuple[int, int]] = []
        highest_risk = 0.0

        # --- Layer 1: Regex Scan ---
        for pattern in self._compiled:
            for m in pattern.finditer(output):
                spans.append((m.start(), m.end()))
                highest_risk = 1.0

        # --- Layer 2: Semantic Similarity Scan ---
        if self._use_layer2 and not spans:
            import numpy as np
            
            self._ensure_model()
            assert self._model is not None and self._seed_embeddings is not None

            embedding: np.ndarray = self._model.encode(
                output, convert_to_numpy=True, normalize_embeddings=False
            )
            sims = _cosine_similarity(embedding, self._seed_embeddings)
            max_sim: float = float(np.max(sims))

            if max_sim >= self._embedding_threshold:
                highest_risk = float(max_sim)
                return output, False, highest_risk

        # --- Layer 3: LLM Scan ---
        if self._use_layer3 and not spans:
            llm_risk = self._scan_layer3(output)
            if llm_risk >= self._llm_threshold:
                highest_risk = max(highest_risk, llm_risk)
                return output, False, highest_risk

        if not spans:
            return output, True, 0.0

        if not self._redact:
            return output, False, highest_risk

        # Merge overlapping/duplicate spans
        spans.sort(key=lambda x: x[0])
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        # Replace right-to-left with asterisks matching the length
        sanitized = output
        for start, end in reversed(merged):
            length = end - start
            redacted_str = "*" * length
            sanitized = sanitized[:start] + redacted_str + sanitized[end:]

        return sanitized, False, highest_risk
