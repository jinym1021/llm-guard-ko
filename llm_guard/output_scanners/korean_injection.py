"""Layer 1 — Korean prompt-injection / jailbreak regex scanner (output side).

Checks if the LLM's own **response** contains injection or jailbreak phrases.
This is uncommon but possible when an adversary tricks the model into echoing
or continuing an attack payload.

Conforms to llm-guard's output Scanner protocol::

    scan(prompt, output) -> (sanitized_output, is_valid, risk_score)
"""

from __future__ import annotations

import re
from typing import Pattern

from llm_guard.input_scanners.korean_patterns import KOREAN_INJECTION_PATTERNS


class KoreanInjection:
    """Scan the LLM output for Korean injection/jailbreak phrases."""

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

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        for _label, pattern in self._compiled:
            if pattern.search(output):
                return output, False, 1.0
        return output, True, 0.0
