"""Layer 3 — Korean Factual Consistency (Hallucination) scanner (output).

Checks whether the LLM's response is factually entailed by the user's prompt
(or provided context) using a pretrained Korean Natural Language Inference (NLI) model.
This is critical for preventing LLM Hallucinations in restricted domains (RAG).

Requires the ``factual`` extra::

    pip install llm-guard-ko[factual]

The ``model_name`` argument is **required**. There is no good public Korean
NLI default model that is small enough to ship transparently, so the user
must explicitly pass an NLI-finetuned model ID (e.g. one fine-tuned on
KLUE-NLI). Passing a generic backbone (e.g. a seq2seq model) will produce
meaningless entailment scores, so we refuse to silently default.
"""

from __future__ import annotations

DEFAULT_THRESHOLD = 0.5
ENTAILMENT_INDEX = 0  # Convention used here: index 0 of softmax = entailment.


class KoreanFactualConsistency:
    """Scan Korean outputs for hallucination/factual inconsistency using NLI.

    Args:
        model_name: HuggingFace model ID for a Korean NLI classifier
            (e.g. one fine-tuned on KLUE-NLI). **Required** — no default,
            because a wrong choice silently produces meaningless scores.
            The model must output 3 logits ordered ``[entailment, neutral, contradiction]``;
            override :data:`ENTAILMENT_INDEX` if your model differs.
        threshold: Entailment confidence threshold in [0, 1]. Outputs whose
            entailment probability falls below this are flagged.
        device: Torch device string. Auto-selects if None.
    """

    def __init__(
        self,
        *,
        model_name: str,
        threshold: float = DEFAULT_THRESHOLD,
        device: str | None = None,
    ) -> None:
        if not model_name:
            raise ValueError(
                "KoreanFactualConsistency requires an explicit `model_name` of a "
                "Korean NLI-finetuned model. There is no safe default."
            )
        self._model_name = model_name
        self._threshold = threshold
        self._device = device

        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        try:
            import torch as _torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "KoreanFactualConsistency requires the 'factual' extra: "
                "pip install llm-guard-ko[factual]"
            ) from exc

        if self._device is None:
            self._device = "cuda" if _torch.cuda.is_available() else "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name
        ).to(self._device)
        self._model.eval()

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """Check that *output* is entailed by *prompt* under the NLI model.

        Returns:
            ``(output, is_valid, risk_score)``. ``is_valid=False`` and
            ``risk_score = 1 - entailment_prob`` when the entailment
            probability falls below the threshold. Empty prompt or output
            short-circuits to ``is_valid=True`` (nothing to verify against).
        """
        import torch

        if not prompt.strip() or not output.strip():
            return output, True, 0.0

        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(
            prompt, output, return_tensors="pt", truncation=True, padding=True
        ).to(self._device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits[0], dim=-1)

        entailment_score = float(probs[ENTAILMENT_INDEX])

        if entailment_score < self._threshold:
            return output, False, 1.0 - entailment_score

        return output, True, 0.0
