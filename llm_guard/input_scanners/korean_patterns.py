"""Utility for loading Korean-specific regex patterns.

Handles loading PII rules from JSON files and inferring regexes from examples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from regex_builder import infer_from_example
from llm_guard.patterns.korean import KOREAN_PII_PATTERNS as DEFAULT_PII_PATTERNS

DEFAULT_RULES_PATH = Path(__file__).parent / "pii_rule.json"
_REGEX_INDICATOR_CHARS = frozenset(r"\[()*+?.{}^$|")


def _is_regex(value: str) -> bool:
    """Return True if *value* looks like a regex rather than a plain example."""
    return any(c in _REGEX_INDICATOR_CHARS for c in value)


def load_pii_rules(path: str | Path | None = None) -> dict[str, str]:
    """Load PII rules from a JSON file and return a ``{label: regex}`` dict.

    Resolution order (first match wins):
    1. *path* argument — explicit per-call override.
    2. ``$LLM_GUARD_PII_RULES`` environment variable.
    3. Local ``pii_rules.json`` or ``pii_rule.json`` in the CWD.
    4. :data:`DEFAULT_RULES_PATH` — bundled default.

    Returns:
        Mapping from rule label to compiled-ready regex string.
    """
    if path is None:
        env = os.environ.get("LLM_GUARD_PII_RULES")
        if env:
            path = Path(env)
        elif Path("pii_rules.json").exists():
            path = Path("pii_rules.json")
        elif Path("pii_rule.json").exists():
            path = Path("pii_rule.json")
        else:
            path = DEFAULT_RULES_PATH

    if not Path(path).exists():
        # Fallback to hardcoded defaults if file not found
        return DEFAULT_PII_PATTERNS

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PII rules file must be a JSON object, got {type(data).__name__}")

    rules: dict[str, str] = {}
    for label, value in data.items():
        if isinstance(value, list):
            patterns = [v if _is_regex(v) else infer_from_example(v) for v in value]
            rules[label] = f"(?:{'|'.join(patterns)})"
        elif isinstance(value, str):
            rules[label] = value if _is_regex(value) else infer_from_example(value)
        else:
            raise ValueError(f"Value for {label!r} must be str or list, got {type(value).__name__}")
    return rules
