"""Layer 1 — Korean Toxicity scanner for outputs.

Uses regex patterns to quickly detect and optionally redact Korean slang,
profanity, and evasion techniques.
"""

from __future__ import annotations

import re
from typing import Pattern

from llm_guard.patterns.korean import KOREAN_TOXIC_PATTERNS


class KoreanToxicity:
    """Scan Korean output text for toxic words/profanity."""

    def __init__(
        self,
        *,
        redact: bool = False,
        patterns: list[str] | None = None,
    ) -> None:
        self._redact = redact
        source = patterns if patterns is not None else KOREAN_TOXIC_PATTERNS
        self._compiled: list[Pattern[str]] = [re.compile(p) for p in source]

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        spans: list[tuple[int, int]] = []
        for pattern in self._compiled:
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

        # Replace right-to-left with asterisks matching the length
        sanitized = output
        for start, end in reversed(merged):
            length = end - start
            redacted_str = "*" * length
            sanitized = sanitized[:start] + redacted_str + sanitized[end:]

        return sanitized, False, 1.0
