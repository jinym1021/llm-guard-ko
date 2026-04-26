"""Layer 1 — Korean prompt-injection / jailbreak regex scanner.

Matches characteristic Korean jailbreak phrases. Returns llm-guard's
Scanner protocol 3-tuple ``(sanitized_text, is_valid, risk_score)``.

Does not redact by default — for injection attempts, the caller almost
always wants to reject the whole prompt rather than patch it up.
"""

from __future__ import annotations

import re
from typing import Pattern

from llm_guard.ko.patterns import KOREAN_INJECTION_PATTERNS


class KoreanInjection:
    """Scan Korean text for prompt-injection phrases."""

    def __init__(
        self,
        *,
        patterns: dict[str, str] | None = None,
    ) -> None:
        source = patterns if patterns is not None else KOREAN_INJECTION_PATTERNS
        self._compiled: list[tuple[str, Pattern[str]]] = [
            (label, re.compile(pattern, re.IGNORECASE))
            for label, pattern in source.items()
        ]

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        for _label, pattern in self._compiled:
            if pattern.search(prompt):
                return prompt, False, 1.0
        return prompt, True, 0.0
