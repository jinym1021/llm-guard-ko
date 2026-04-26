"""Layer 1 — Korean PII regex scanner.

Matches Korean-specific PII patterns (RRN, phone, business/bank/credit
numbers) and returns llm-guard's Scanner protocol 3-tuple::

    (sanitized_text, is_valid, risk_score)

Conventions match upstream llm-guard:
    is_valid == True  -> safe
    risk_score        -> 0.0 safe, 1.0 risky
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Pattern

from llm_guard.input_scanners.korean_patterns import KOREAN_PII_PATTERNS, load_pii_rules

REDACTION_MARKER = "[REDACTED]"


class KoreanPII:
    """Scan Korean text for PII and optionally redact matches.

    Args:
        redact: If True (default), replace matches with ``[REDACTED]`` in
            the returned text. If False, return the original text but
            still flag it as invalid.
        patterns: Optional override for the pattern dict. Defaults to
            :data:`llm_guard.input_scanners.korean_patterns.KOREAN_PII_PATTERNS`.
        rule_file: Path to a ``pii_rule.json`` file. When provided, rules are
            loaded from the file instead of *patterns* or the built-in defaults.
            Each entry maps a Korean rule name (e.g. ``"주민등록번호"``) to either
            a plain example string (``"971021-2333333"``) or a regex pattern
            directly. See :func:`~llm_guard.input_scanners.korean_patterns.load_pii_rules`.
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
            source = KOREAN_PII_PATTERNS
        self._compiled: list[tuple[str, Pattern[str]]] = [
            (label, re.compile(pattern)) for label, pattern in source.items()
        ]

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        # Collect all match spans across all patterns.
        spans: list[tuple[int, int]] = []
        for _label, pattern in self._compiled:
            for m in pattern.finditer(prompt):
                spans.append((m.start(), m.end()))

        if not spans:
            return prompt, True, 0.0

        if not self._redact:
            return prompt, False, 1.0

        # Merge overlapping/duplicate spans so each region is redacted
        # exactly once. Sort by start, then greedily extend.
        spans.sort()
        merged: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:  # overlap or touch
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        # Replace right-to-left so earlier indices remain valid.
        sanitized = prompt
        for start, end in reversed(merged):
            sanitized = sanitized[:start] + REDACTION_MARKER + sanitized[end:]

        return sanitized, False, 1.0
