"""Layer 1 — Korean PII regex scanner for outputs.

Matches Korean-specific PII patterns (RRN, phone, business/bank/credit
numbers) and returns llm-guard's Scanner protocol 3-tuple:
    (sanitized_text, is_valid, risk_score)
"""

from __future__ import annotations

import re
from typing import Pattern

from llm_guard_ko.patterns.korean import KOREAN_PII_PATTERNS

REDACTION_MARKER = "[REDACTED]"


class KoreanPII:
    """Scan Korean text for PII in the model output and optionally redact matches."""

    def __init__(
        self,
        *,
        redact: bool = True,
        patterns: dict[str, str] | None = None,
    ) -> None:
        self._redact = redact
        source = patterns if patterns is not None else KOREAN_PII_PATTERNS
        self._compiled: list[tuple[str, Pattern[str]]] = [
            (label, re.compile(pattern)) for label, pattern in source.items()
        ]

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        # Collect all match spans across all patterns.
        spans: list[tuple[int, int]] = []
        for _label, pattern in self._compiled:
            for m in pattern.finditer(output):
                spans.append((m.start(), m.end()))

        if not spans:
            return output, True, 0.0

        if not self._redact:
            return output, False, 1.0

        # Merge overlapping/duplicate spans
        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        # Replace right-to-left
        sanitized = output
        for start, end in reversed(merged):
            sanitized = sanitized[:start] + REDACTION_MARKER + sanitized[end:]

        return sanitized, False, 1.0
