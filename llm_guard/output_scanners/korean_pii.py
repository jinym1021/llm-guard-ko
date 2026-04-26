"""Layer 1 — Korean PII regex scanner for outputs.

Matches Korean-specific PII patterns (RRN, phone, business/bank/credit
numbers) and returns llm-guard's Scanner protocol 3-tuple:
    (sanitized_text, is_valid, risk_score)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Pattern

from llm_guard.input_scanners.korean_patterns import load_pii_rules

REDACTION_MARKER = "[REDACTED]"


class KoreanPII:
    """Scan the LLM output for Korean PII and optionally redact matches.

    Args:
        redact: If True (default), replace matches with ``[REDACTED]`` in
            the returned text. If False, return the original output but
            still flag it as invalid.
        patterns: Optional override for the pattern dict. When provided,
            takes precedence over *rule_file* and the default rule resolution.
        rule_file: Path to a ``pii_rule.json`` file. When provided, rules are
            loaded from the file instead of the default resolution order
            (``$LLM_GUARD_PII_RULES`` env var → bundled ``pii_rule.json``).
            See :func:`~llm_guard.input_scanners.korean_patterns.load_pii_rules`.
    """

    def __init__(
        self,
        *,
        redact: bool = True,
        patterns: dict[str, str] | None = None,
        rule_file: str | Path | None = None,
    ) -> None:
        self._redact = redact
        if rule_file is not None:
            source = load_pii_rules(rule_file)
        elif patterns is not None:
            source = patterns
        else:
            source = load_pii_rules()
        self._compiled: list[tuple[str, Pattern[str]]] = [
            (label, re.compile(pattern)) for label, pattern in source.items()
        ]

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        # Collect all match spans across all patterns.
        spans: list[tuple[int, int]] = []
        for _label, pattern in self._compiled:
            spans.extend((m.start(), m.end()) for m in pattern.finditer(output))

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
