"""Korean-specific regex patterns for PII and prompt-injection detection.

Each entry maps a label (used in logging/inspection) to a regex string.
Patterns are intentionally conservative: precision over recall for
Layer 1, because false positives here get escalated to Layer 2/3 by
``KoreanPipeline``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from regex_builder import infer_from_example

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
# Toxicity / Profanity patterns — match on Korean slang/swear words.
# ---------------------------------------------------------------------------
# [0-9^~_.,\s]* is used to prevent bypasses like inserting numbers or spaces.
KOREAN_TOXIC_PATTERNS: list[str] = [
    r"(씨|ㅆ)[0-9^~_.,\s]*(발|빨|팟|팔|팍|ㅂ|1ㅂ)",
    r"(개|ㄱㅐ)[0-9^~_.,\s]*(새|세)[0-9^~_.,\s]*(끼|기|ㄲl|ㄲ|ㄲ1)",
    r"(지|ㅈ)[0-9^~_.,\s]*(랄|럴|ㄹ)",
    r"(미|미췬|ㅁ)[0-9^~_.,\s]*(친)[0-9^~_.,\s]*(놈|년|새|끼|ㄴ)",
    r"(존|좃|ㅈ)[0-9^~_.,\s]*(나|잼|나게|니|ㄴ)",
    r"(좆|ㅈ)[0-9^~_.,\s]*(같|나|밥|까|됨|된|ㄱ)",
    r"(병|ㅂ)[0-9^~_.,\s]*(신|씬|ㅅ)",
    r"(호|ㅎ)[0-9^~_.,\s]*(구|로)[0-9^~_.,\s]*(새|끼|놈)",
    r"(썅|ㅆ)[0-9^~_.,\s]*(년|놈|새|끼)",
    r"(염|ㅇ)[0-9^~_.,\s]*(병|ㅂ)",
    r"(애|ㅇ)[0-9^~_.,\s]*(미|비|ㅁ|ㅂ)",
]

# ---------------------------------------------------------------------------
# Model Refusal phrases — match case-insensitively.
# ---------------------------------------------------------------------------
# Substrings commonly used by models to refuse fulfilling a prompt.
KOREAN_REFUSAL_PATTERNS: list[str] = [
    "죄송하지만",
    "죄송합니다만",
    "저는 AI 언어 모델로서",
    "AI 모델로서",
    "도와드릴 수 없습니다",
    "도움을 드릴 수 없습니다",
    "답변할 수 없습니다",
    "답변해 드릴 수 없습니다",
    "제공할 수 없습니다",
    "응답할 권한이 없습니다",
    "범위를 벗어납니다",
    "프로그래밍되어 있습니다",
    "처리할 수 없는 요청입니다",
    "불법적",
    "비윤리적",
    "윤리에 어긋나는",
    "도덕적이지 않은",
    "유해한 컨텐츠",
    "사용자 가이드라인",
    "지원할 수 없습니다",
    "할 수 없습니다",
    "어렵습니다",
    "제한되어 있습니다",
    "정책에 위배",
    "허용되지 않습니다",
]

__all__ = [
    "KOREAN_PII_PATTERNS",
    "KOREAN_INJECTION_PATTERNS",
    "KOREAN_TOXIC_PATTERNS",
    "KOREAN_REFUSAL_PATTERNS",
]
# pii_rule.json loader — converts user-supplied examples into patterns.
# ---------------------------------------------------------------------------

DEFAULT_RULES_PATH = Path(__file__).parent / "pii_rule.json"

_REGEX_INDICATOR_CHARS = frozenset(r"\[()*+?.{}^$|")


def _is_regex(value: str) -> bool:
    """Return True if *value* looks like a regex rather than a plain example."""
    return any(c in _REGEX_INDICATOR_CHARS for c in value)


def load_pii_rules(path: str | Path | None = None) -> dict[str, str]:
    """Load PII rules from a JSON file and return a ``{label: regex}`` dict.

    Resolution order (first match wins):

    1. *path* argument — explicit per-call override.
    2. ``$LLM_GUARD_PII_RULES`` environment variable — per-process override.
    3. Local ``pii_rules.json`` or ``pii_rule.json`` in the current directory.
    4. :data:`DEFAULT_RULES_PATH` — bundled default shipped with the package.

    Each JSON entry should have the rule name as key and either:
    * a **plain example** (e.g. ``"971021-2333333"``) — a regex is inferred
      automatically via :func:`regex_builder.infer_from_example`,
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
            patterns = [v if _is_regex(v) else infer_from_example(v) for v in value]
            rules[label] = f"(?:{'|'.join(patterns)})"
        elif isinstance(value, str):
            rules[label] = value if _is_regex(value) else infer_from_example(value)
        else:
            raise ValueError(
                f"Value for rule {label!r} must be a string or list of strings, got {type(value).__name__}"
            )
    return rules


# ---------------------------------------------------------------------------
# PII patterns — loaded from the bundled pii_rule.json.
# ---------------------------------------------------------------------------
# Note on RRN (주민등록번호): the 7th digit encodes sex + century
# (1-4 valid for modern records; 5-8 for foreign residents). We accept
# 1-8 to cover both citizens and foreign residents.
KOREAN_PII_PATTERNS: dict[str, str] = load_pii_rules(DEFAULT_RULES_PATH)
