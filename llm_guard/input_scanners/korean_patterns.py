"""Korean-specific regex patterns for PII and prompt-injection detection.

Each entry maps a label (used in logging/inspection) to a regex string.
Patterns are intentionally conservative: precision over recall for
Layer 1, because false positives here get escalated to Layer 2/3 by
``KoreanPipeline``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# PII patterns — match on Korean-specific identifiers.
# ---------------------------------------------------------------------------
# Note on RRN (주민등록번호): the 7th digit encodes sex + century
# (1-4 valid for modern records; 5-8 for foreign residents). We accept
# 1-8 to cover both citizens and foreign residents.
KOREAN_PII_PATTERNS: dict[str, str] = {
    "resident_registration_number": r"\d{6}-[1-8]\d{6}",
    "phone_number_mobile": r"01[016789]-?\d{3,4}-?\d{4}",
    "phone_number_landline": r"0(?:2|[3-6][1-5])-?\d{3,4}-?\d{4}",
    "business_registration_number": r"\d{3}-\d{2}-\d{5}",
    "bank_account_number": r"\d{3,6}-\d{2,6}-\d{4,7}",
    "credit_card_number": r"\d{4}-\d{4}-\d{4}-\d{4}",
}

# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak phrases — match case-insensitively.
# ---------------------------------------------------------------------------
# Each pattern is a short Korean phrase characteristic of injection
# attempts. Keep these as substrings, not full-line anchors, because
# attackers embed them mid-sentence.
KOREAN_INJECTION_PATTERNS: dict[str, str] = {
    "ignore_previous": r"이전\s*(?:지시|명령|프롬프트|규칙)(?:을|를)?\s*(?:모두\s*)?(?:무시|잊|지워)",
    "you_are_now": r"지금부터\s*(?:너|넌|당신(?:은|이)?)\s*",
    "role_play": r"(?:역할극|롤플레이|롤\s*플레이)",
    "system_prompt_disclose": r"(?:시스템\s*프롬프트|system\s*prompt)(?:을|를)?\s*(?:알려|보여|출력|공개)",
    "jailbreak_developer_mode": r"(?:개발자\s*모드|developer\s*mode|DAN|탈옥|jailbreak)",
    "no_restrictions": r"(?:제한\s*없는|제약\s*없는|필터\s*없는)\s*(?:AI|인공지능|모델)",
    # New in llm-guard-ko v0.1 (listed in the original plan but never
    # landed in the old sibling package):
    "pretend_to_be": r"(?:인\s*척|처럼\s*행동|인\s*것처럼)",
    "bypass_filter": r"(?:필터|안전\s*장치|가드레일)(?:을|를)?\s*우회",
}

# ---------------------------------------------------------------------------
# pii_rule.json loader — converts user-supplied examples into patterns.
# ---------------------------------------------------------------------------

DEFAULT_RULES_PATH = Path(__file__).parent / "pii_rule.json"

_REGEX_INDICATOR_CHARS = frozenset(r"\[()*+?.{}^$|")


def _is_regex(value: str) -> bool:
    """Return True if *value* looks like a regex rather than a plain example."""
    return any(c in _REGEX_INDICATOR_CHARS for c in value)


def _infer_regex_from_example(example: str) -> str:
    """Build a regex pattern from a concrete example string.

    Consecutive characters of the same class are collapsed:
      digits       → ``\\d{n}``
      Korean chars → ``[가-힣]{n}``
      Latin letters → ``[A-Za-z]{n}``
      everything else → ``re.escape(char)`` (literal)

    Separators (hyphens and spaces) are made optional in the resulting pattern.

    Example::

        _infer_regex_from_example("971021-2333333") == r"\\d{6}-?\\d{7}"
        _infer_regex_from_example("010-1234-5678")  == r"\\d{3}-?\\d{4}-?\\d{4}"
    """
    parts: list[str] = []
    i = 0
    while i < len(example):
        c = example[i]
        if c.isdigit():
            j = i
            while j < len(example) and example[j].isdigit():
                j += 1
            parts.append(rf"\d{{{j - i}}}")
            i = j
        elif "가" <= c <= "힣":  # Hangul syllable block
            j = i
            while j < len(example) and "가" <= example[j] <= "힣":
                j += 1
            parts.append(f"[가-힣]{{{j - i}}}")
            i = j
        elif c.isalpha():
            j = i
            while j < len(example) and example[j].isalpha() and not ("가" <= example[j] <= "힣"):
                j += 1
            parts.append(f"[A-Za-z]{{{j - i}}}")
            i = j
        elif c in ("-", " "):
            parts.append(rf"{re.escape(c)}?")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return "".join(parts)


def load_pii_rules(path: str | Path | None = None) -> dict[str, str]:
    """Load PII rules from a JSON file and return a ``{label: regex}`` dict.

    Resolution order (first match wins):

    1. *path* argument — explicit per-call override.
    2. ``$LLM_GUARD_PII_RULES`` environment variable — per-process override.
    3. Local ``pii_rules.json`` or ``pii_rule.json`` in the current directory.
    4. :data:`DEFAULT_RULES_PATH` — bundled default shipped with the package.

    Each JSON entry should have the rule name as key and either:
    * a **plain example** (e.g. ``"971021-2333333"``) — a regex is inferred
      automatically via :func:`_infer_regex_from_example`,
    * a **list of plain examples** — multiple regexes are inferred and combined, or
    * a **regex pattern** (e.g. ``"\\\\d{6}-[1-8]\\\\d{6}"``) — used as-is.

    A value is treated as a regex when it contains meta-characters like ``\\``, ``[``,
    ``*``, etc.

    Example ``pii_rule.json``::

        {
            "주민등록번호": "971021-2333333",
            "휴대폰번호":  ["010-1234-5678", "01012345678"]
        }

    Args:
        path: Path to the JSON file. When ``None``, the resolution order above is followed.

    Returns:
        Mapping from rule label to compiled-ready regex string.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ValueError: If the JSON is not a flat ``{str: str|list[str]}`` object.
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

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"PII rules file must be a JSON object, got {type(data).__name__}")

    rules: dict[str, str] = {}
    for label, value in data.items():
        if isinstance(value, list):
            patterns = [v if _is_regex(v) else _infer_regex_from_example(v) for v in value]
            rules[label] = f"(?:{'|'.join(patterns)})"
        elif isinstance(value, str):
            rules[label] = value if _is_regex(value) else _infer_regex_from_example(value)
        else:
            raise ValueError(
                f"Value for rule {label!r} must be a string or list of strings, got {type(value).__name__}"
            )
    return rules
