"""Layer 1 — Korean NoRefusal scanner for outputs.

Detects phrases commonly used by Korean models when refusing to answer
a prompt due to policy or safety guidelines.
"""

from __future__ import annotations

from llm_guard.patterns.korean import KOREAN_REFUSAL_PATTERNS


class KoreanNoRefusal:
    """Scan Korean output text for refusal patterns."""

    def __init__(
        self,
        *,
        patterns: list[str] | None = None,
    ) -> None:
        self._patterns = patterns if patterns is not None else KOREAN_REFUSAL_PATTERNS
        # Lowercase the patterns for case-insensitive matching
        self._patterns_lower = [p.lower() for p in self._patterns]

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not output.strip():
            return output, True, 0.0

        output_lower = output.lower()

        for pattern in self._patterns_lower:
            if pattern in output_lower:
                return output, False, 1.0

        return output, True, 0.0
