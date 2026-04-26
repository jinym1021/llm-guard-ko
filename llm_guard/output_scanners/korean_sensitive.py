"""Layer 2 — Korean Sensitive (PII) NER scanner (output).

Detects and optionally redacts sensitive entities (Person, Location, Organization, etc.)
in the LLM output using a lightweight Korean NER (Named Entity Recognition) model.
This complements Layer 1 regex by catching context-dependent PII that lacks a fixed pattern.

Requires the ``sensitive`` extra::

    pip install llm-guard-ko[sensitive]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import Pipeline

# Using a robust and lightweight Korean NER model.
DEFAULT_MODEL = "beomi/KcELECTRA-base-v2022"
DEFAULT_THRESHOLD = 0.8

# Map NER tags to REDACTION_MARKERS
# B-PS/I-PS: Person, B-LC/I-LC: Location, B-OG/I-OG: Organization
TAG_MAP = {
    "PS": "[PERSON]",
    "LC": "[LOCATION]",
    "OG": "[ORGANIZATION]",
}


class KoreanSensitive:
    """Scan Korean outputs for sensitive entities using NER.

    Args:
        model_name: HuggingFace model ID for Korean NER. Defaults to ``beomi/KcELECTRA-base-v2022``.
        threshold: Confidence threshold above which an entity is redacted/flagged. Range [0, 1].
        redact: If True, replace detected entities with markers like [PERSON].
        device: Torch device string. Auto-selects if None.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        redact: bool = True,
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold
        self._redact = redact
        self._device = device
        
        # Lazy initialization
        self._pipeline: Pipeline | None = None

    def _ensure_model(self) -> None:
        if self._pipeline is not None:
            return
        
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "KoreanSensitive requires the 'sensitive' extra: "
                "pip install llm-guard-ko[sensitive]"
            ) from exc

        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Determine device index for transformers pipeline (-1 for CPU, 0+ for GPU)
        device_id = -1 if self._device == "cpu" else int(self._device.split(":")[-1]) if ":" in self._device else 0

        # Initialize the NER pipeline
        self._pipeline = pipeline(
            "ner",
            model=self._model_name,
            tokenizer=self._model_name,
            device=device_id,
            aggregation_strategy="simple" # Groups B- and I- tags into single words
        )

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        self._ensure_model()
        assert self._pipeline is not None

        # Predict entities
        entities: list[dict[str, Any]] = self._pipeline(output) # type: ignore
        
        # Filter by threshold and known mapped tags
        valid_entities = []
        highest_risk = 0.0
        
        for ent in entities:
            # entity_group e.g., 'PS', 'LC', 'OG' due to aggregation_strategy="simple"
            group = ent.get("entity_group", "")
            score = float(ent.get("score", 0.0))
            
            if score >= self._threshold and group in TAG_MAP:
                valid_entities.append(ent)
                highest_risk = max(highest_risk, score)

        if not valid_entities:
            return output, True, 0.0

        if not self._redact:
            # Found PII but user doesn't want redaction -> return original output as invalid
            return output, False, highest_risk

        # Redact right-to-left to avoid index shifting
        valid_entities.sort(key=lambda x: x["start"], reverse=True)
        sanitized = output
        
        for ent in valid_entities:
            start, end = ent["start"], ent["end"]
            marker = TAG_MAP.get(ent["entity_group"], "[REDACTED]")
            sanitized = sanitized[:start] + marker + sanitized[end:]

        return sanitized, False, highest_risk
