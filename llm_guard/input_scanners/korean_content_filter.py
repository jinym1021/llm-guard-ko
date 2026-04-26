"""Layer 3 — Korean content filter (input).

Classifies prompts against 5 MLCommons safety categories using the
SGuard-ContentFilter-2B-v1 local classifier. Intended to run only after
Layer 1 and Layer 2 have flagged the text (escalation contract).

Typical latency: ~hundreds of ms on CPU, ~50 ms on GPU.

Requires the ``content-filter`` extra::

    pip install llm-guard-ko[content-filter]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL = "SamsungSDS-Research/SGuard-ContentFilter-2B-v1"
DEFAULT_THRESHOLD = 0.5

# MLCommons AI Safety hazard categories supported by SGuard.
CATEGORIES: tuple[str, ...] = (
    "violent_crimes",
    "non_violent_crimes",
    "sex_crimes",
    "child_exploitation",
    "indiscriminate_weapons",
)


class KoreanContentFilter:
    """Scan Korean prompts using the SGuard local content-filter model.

    Each of the 5 MLCommons categories is evaluated independently.
    A prompt is flagged if any category's unsafe probability exceeds
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
        self._model_name = model_name
        self._threshold = threshold
        self._device = device

        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "KoreanContentFilter requires the 'content-filter' extra: "
                "pip install llm-guard-ko[content-filter]"
            ) from exc

        if self._device is None:
            import torch as _torch
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            num_labels=len(CATEGORIES),
        ).to(self._device)
        self._model.eval()

    def _classify(self, text: str) -> dict[str, float]:
        """Return per-category unsafe probabilities for *text*."""
        import torch

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            logits: torch.Tensor = self._model(**inputs).logits

        probs = torch.sigmoid(logits[0]).tolist()
        return {cat: float(p) for cat, p in zip(CATEGORIES, probs)}

    def scan_detailed(self, prompt: str) -> dict[str, dict]:
        """Return per-category classification detail for *prompt*.

        Returns:
            Mapping of category name → ``{"unsafe": bool, "prob": float}``.
        """
        probs = self._classify(prompt)
        return {
            cat: {"unsafe": prob >= self._threshold, "prob": prob}
            for cat, prob in probs.items()
        }

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """Scan *prompt* for unsafe content.

        Returns:
            ``(prompt, is_valid, risk_score)`` where ``is_valid=True``
            means safe and ``risk_score`` is the max unsafe probability
            across all categories (0.0 when safe).
        """
        probs = self._classify(prompt)
        max_prob = max(probs.values())
        if max_prob >= self._threshold:
            return prompt, False, float(max_prob)
        return prompt, True, 0.0
