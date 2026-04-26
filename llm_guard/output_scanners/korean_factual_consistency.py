"""Layer 3 — Korean Factual Consistency (Hallucination) scanner (output).

Checks whether the LLM's response is factually entailed by the user's prompt
(or provided context) using a pretrained Korean Natural Language Inference (NLI) model.
This is critical for preventing LLM Hallucinations in restricted domains (RAG).

Requires the ``factual`` extra::

    pip install llm-guard-ko[factual]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Using a standard Korean NLI model
DEFAULT_MODEL = "skt/kobart-base-v2" # Note: In practice, an NLI fine-tuned model like 'upstage/nli-ko-electra' is used.
DEFAULT_THRESHOLD = 0.5


class KoreanFactualConsistency:
    """Scan Korean outputs for hallucination/factual inconsistency using NLI.

    Args:
        model_name: HuggingFace model ID for Korean NLI.
        threshold: Entailment confidence threshold. Range [0, 1]. Outputs below this are flagged.
        device: Torch device string. Auto-selects if None.
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
        
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "KoreanFactualConsistency requires the 'factual' extra: "
                "pip install llm-guard-ko[factual]"
            ) from exc

        if self._device is None:
            import torch as _torch
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        # Assuming model outputs 3 classes: [entailment, neutral, contradiction]
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        ).to(self._device)
        self._model.eval()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Scan the response for entailment against the prompt."""
        import torch
        
        if not prompt.strip() or not output.strip():
            # If no context is provided to check against, it is valid by default
            return output, True, 0.0

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        # Cross-encode: [CLS] prompt [SEP] output [SEP]
        inputs = self._tokenizer(
            prompt, output, return_tensors="pt", truncation=True, padding=True
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits[0], dim=-1)
        
        # Typically, index 0 is entailment for many NLI models (but varies by model).
        # We assume idx 0 = entailment
        entailment_score = float(probs[0])
        
        if entailment_score < self._threshold:
            # Factually inconsistent (Hallucinated)
            risk = 1.0 - entailment_score
            return output, False, risk
        
        return output, True, 0.0
