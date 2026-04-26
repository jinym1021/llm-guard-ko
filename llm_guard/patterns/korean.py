"""Korean-specific regex patterns for PII, prompt-injection, toxicity and refusal detection.

These patterns are used across Layer 1 scanners to provide fast,
heuristic-based detection of common LLM risks in Korean.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# PII patterns — match on Korean-specific identifiers (LLM06).
# ---------------------------------------------------------------------------
KOREAN_PII_PATTERNS: dict[str, str] = {
    # 주민등록번호: 6 digits - [1-4 or 5-8 for foreigners] + 6 digits
    "resident_registration_number": r"\d{6}-[1-8]\d{6}",
    # 휴대폰 번호: 010, 011, etc.
    "phone_number_mobile": r"01[016789]-?\d{3,4}-?\d{4}",
    # 일반 전화 번호: 02 (Seoul), 031 (Gyeonggi), etc.
    "phone_number_landline": r"0(?:2|[3-6][1-5])-?\d{3,4}-?\d{4}",
    # 사업자 등록 번호: 3-2-5 digits
    "business_registration_number": r"\d{3}-\d{2}-\d{5}",
    # 계좌 번호: Variable formats, conservative match
    "bank_account_number": r"\d{3,6}-\d{2,6}-\d{4,7}",
    # 신용카드 번호: 4-4-4-4 digits
    "credit_card_number": r"\d{4}-\d{4}-\d{4}-\d{4}",
}

# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak phrases — match case-insensitively (LLM01).
# ---------------------------------------------------------------------------
KOREAN_INJECTION_PATTERNS: dict[str, str] = {
    "ignore_previous": r"이전\s*(?:지시|명령|프롬프트|규칙)(?:을|를)?\s*(?:모두\s*)?(?:무시|잊|지워)",
    "you_are_now": r"지금부터\s*(?:너|넌|당신(?:은|이)?)\s*",
    "role_play": r"(?:역할극|롤플레이|롤\s*플레이)",
    "system_prompt_disclose": r"(?:시스템\s*프롬프트|system\s*prompt)(?:을|를)?\s*(?:알려|보여|출력|공개)",
    "jailbreak_developer_mode": r"(?:개발자\s*모드|developer\s*mode|DAN|탈옥|jailbreak)",
    "no_restrictions": r"(?:제한\s*없는|제약\s*없는|필터\s*없는)\s*(?:AI|인공지능|모델)",
    "pretend_to_be": r"(?:인\s*척|처럼\s*행동|인\s*것처럼)",
    "bypass_filter": r"(?:필터|안전\s*장치|가드레일)(?:을|를)?\s*우회",
    "secret_key": r"(?:비밀\s*번호|비밀\s*키|api\s*key|api\s*키)(?:를|을)?\s*(?:알려|공개|출력)",
}

# ---------------------------------------------------------------------------
# Toxicity / Profanity patterns — match on Korean slang/swear words (LLM02).
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
