"""Korean-specific regex patterns for PII and prompt-injection detection.

Each entry maps a label (used in logging/inspection) to a regex string.
Patterns are intentionally conservative: precision over recall for
Layer 1, because false positives here get escalated to Layer 2/3 by
``KoreanPipeline``.
"""

from __future__ import annotations

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
