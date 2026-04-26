"""Layer 1 — Korean PII regex scanner (output side).

Checks the LLM's **response** for Korean PII and optionally redacts it.
Conforms to llm-guard's output Scanner protocol::

    scan(prompt, output) -> (sanitized_output, is_valid, risk_score)

``prompt`` is accepted but ignored — PII in the model's response is flagged
regardless of what the user asked.
"""

from __future__ import annotations

import re
from typing import Pattern

from llm_guard.input_scanners.korean_patterns import KOREAN_PII_PATTERNS

REDACTION_MARKER = "[REDACTED]"


class KoreanPII:
    """Scan the LLM output for Korean PII and optionally redact matches.

    Args:
        redact: If True (default), replace matches with ``[REDACTED]`` in
            the returned text. If False, return the original output but
            still flag it as invalid.
        patterns: Optional override for the pattern dict. Defaults to
            :data:`llm_guard.input_scanners.korean_patterns.KOREAN_PII_PATTERNS`.
    """

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
        spans: list[tuple[int, int]] = []
        for _label, pattern in self._compiled:
            for m in pattern.finditer(output):
                spans.append((m.start(), m.end()))

        if not spans:
            return output, True, 0.0

        if not self._redact:
            return output, False, 1.0

        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        sanitized = output
        for start, end in reversed(merged):
            sanitized = sanitized[:start] + REDACTION_MARKER + sanitized[end:]

        return sanitized, False, 1.0
