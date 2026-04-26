"""Layer 1 and Layer 2 — Korean PII scanner for outputs.

Matches Korean-specific PII patterns (Layer 1: regex) and context-dependent
PII using a lightweight Korean NER model (Layer 2: beomi/KcELECTRA-base-ner).

Requires the ``sensitive`` extra for Layer 2::
    pip install llm-guard-ko[sensitive]
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Pattern

from llm_guard.patterns.korean import KOREAN_PII_PATTERNS

if TYPE_CHECKING:
    from transformers import Pipeline

REDACTION_MARKER = "[REDACTED]"
DEFAULT_NER_MODEL = "beomi/KcELECTRA-base-ner"
DEFAULT_NER_THRESHOLD = 0.8

# Map NER tags to specific redaction markers
TAG_MAP = {
    "PS": "[PERSON]",
    "LC": "[LOCATION]",
    "OG": "[ORGANIZATION]",
}


class KoreanPII:
    """Scan Korean text for PII in the model output.

    Escalates from Regex (Layer 1) -> NER (Layer 2).
    """

    def __init__(
        self,
        *,
        redact: bool = True,
        patterns: dict[str, str] | None = None,
        use_layer2: bool = True,
        ner_model_name: str = DEFAULT_NER_MODEL,
        ner_threshold: float = DEFAULT_NER_THRESHOLD,
        device: str | None = None,
    ) -> None:
        self._redact = redact

        # Layer 1 Initialization
        source = patterns if patterns is not None else KOREAN_PII_PATTERNS
        self._compiled: list[tuple[str, Pattern[str]]] = [
            (label, re.compile(pattern)) for label, pattern in source.items()
        ]

        # Layer 2 Initialization
        self._use_layer2 = use_layer2
        self._ner_model_name = ner_model_name
        self._ner_threshold = ner_threshold
        self._device = device
        self._pipeline: Pipeline | None = None

    def _ensure_model(self) -> None:
        if self._pipeline is not None:
            return

        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "KoreanPII Layer 2 requires the 'sensitive' extra: "
                "pip install llm-guard-ko[sensitive]"
            ) from exc

        if self._device is None:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        device_id = -1 if self._device == "cpu" else int(self._device.split(":")[-1]) if ":" in self._device else 0

        self._pipeline = pipeline(
            "ner",
            model=self._ner_model_name,
            tokenizer=self._ner_model_name,
            device=device_id,
            aggregation_strategy="simple"
        )

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        spans: list[tuple[int, int, str]] = []
        highest_risk = 0.0

        # --- Layer 1: Regex Scan ---
        for label, pattern in self._compiled:
            for m in pattern.finditer(output):
                spans.append((m.start(), m.end(), REDACTION_MARKER))
                highest_risk = 1.0

        # --- Layer 2: NER Scan ---
        if self._use_layer2:
            self._ensure_model()
            assert self._pipeline is not None

            entities: list[dict[str, Any]] = self._pipeline(output) # type: ignore
            for ent in entities:
                group = ent.get("entity_group", "")
                score = float(ent.get("score", 0.0))

                if score >= self._ner_threshold and group in TAG_MAP:
                    spans.append((ent["start"], ent["end"], TAG_MAP[group]))
                    highest_risk = max(highest_risk, score)

        if not spans:
            return output, True, 0.0

        if not self._redact:
            return output, False, highest_risk

        # Merge overlapping/duplicate spans
        spans.sort(key=lambda x: x[0])
        merged: list[tuple[int, int, str]] = [spans[0]]
        for start, end, marker in spans[1:]:
            last_start, last_end, last_marker = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end), last_marker)
            else:
                merged.append((start, end, marker))

        # Replace right-to-left
        sanitized = output
        for start, end, marker in reversed(merged):
            sanitized = sanitized[:start] + marker + sanitized[end:]

        return sanitized, False, highest_risk
