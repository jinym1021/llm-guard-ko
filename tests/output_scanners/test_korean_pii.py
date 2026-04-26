"""Tests for Layer 1: KoreanPII output scanner.

Contract (output Scanner protocol):
    scan(prompt, output) -> (sanitized_output, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score in [0.0, 1.0]
"""

from __future__ import annotations

import pytest

from llm_guard.output_scanners import KoreanPII

_DUMMY_PROMPT = "아무 프롬프트"


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = KoreanPII()
    result = scanner.scan(_DUMMY_PROMPT, "hello")
    assert isinstance(result, tuple) and len(result) == 3
    text, is_valid, risk = result
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_benign_output_is_valid_and_zero_risk():
    scanner = KoreanPII()
    text, is_valid, risk = scanner.scan(_DUMMY_PROMPT, "네, 도와드릴게요.")
    assert is_valid is True
    assert risk == 0.0
    assert text == "네, 도와드릴게요."


# ---------------------------------------------------------------------------
# PII detection — SHOULD_FLAG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output,label",
    [
        ("귀하의 주민번호는 901010-1234567입니다", "rrn"),
        ("연락처: 010-1234-5678", "phone_mobile"),
        ("전화: 02-123-4567 로 연락하세요", "phone_landline"),
        ("사업자등록번호는 123-45-67890입니다", "biz_reg"),
        ("계좌: 123-456-789012 로 송금하세요", "bank"),
        ("카드번호 1234-5678-9012-3456 확인됨", "card"),
    ],
)
def test_pii_in_output_is_flagged_and_redacted(output, label):
    scanner = KoreanPII(redact=True)
    text, is_valid, risk = scanner.scan(_DUMMY_PROMPT, output)
    assert is_valid is False, f"{label!r} should be flagged"
    assert risk == 1.0
    assert "[REDACTED]" in text
    assert text != output


def test_redact_false_keeps_original_output_but_still_flags():
    scanner = KoreanPII(redact=False)
    text, is_valid, risk = scanner.scan(_DUMMY_PROMPT, "주민번호 901010-1234567")
    assert is_valid is False
    assert risk == 1.0
    assert "901010-1234567" in text


def test_prompt_is_ignored_only_output_is_checked():
    """PII in prompt must not affect the result — only output is scanned."""
    scanner = KoreanPII(redact=True)
    # prompt contains PII, output is clean
    text, is_valid, risk = scanner.scan("내 번호 901010-1234567", "응, 알겠어요.")
    assert is_valid is True
    assert risk == 0.0


def test_overlapping_patterns_redact_once():
    scanner = KoreanPII(redact=True)
    text, is_valid, _ = scanner.scan(_DUMMY_PROMPT, "번호 010-1234-5678 확인됨")
    assert is_valid is False
    assert text.count("[REDACTED]") == 1
    assert "번호" in text
    assert "확인됨" in text


def test_multiple_distinct_pii_all_redacted():
    scanner = KoreanPII(redact=True)
    output = "주민번호 901010-1234567 전화 010-1234-5678"
    text, is_valid, _ = scanner.scan(_DUMMY_PROMPT, output)
    assert is_valid is False
    assert "901010-1234567" not in text
    assert "010-1234-5678" not in text
    assert text.count("[REDACTED]") == 2


# ---------------------------------------------------------------------------
# SHOULD_PASS — avoid false positives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output",
    [
        "네, 도와드릴게요.",
        "파이썬 3.11 버전을 사용하세요.",
        "2024년은 좋은 해였습니다.",
        "버전 1.2.3 업데이트가 있습니다.",
        "오늘 날씨는 맑습니다.",
    ],
)
def test_benign_outputs_pass(output):
    scanner = KoreanPII()
    _, is_valid, risk = scanner.scan(_DUMMY_PROMPT, output)
    assert is_valid is True, f"false positive on: {output!r}"
    assert risk == 0.0


# ---------------------------------------------------------------------------
# llm-guard integration
# ---------------------------------------------------------------------------

def test_integrates_with_llm_guard_scan_output():
    from llm_guard import scan_output

    scanner = KoreanPII()
    sanitized, valid, risk = scan_output(
        [scanner],
        _DUMMY_PROMPT,
        "귀하의 주민번호는 901010-1234567입니다",
    )
    assert valid == {"KoreanPII": False}
    assert risk == {"KoreanPII": 1.0}
    assert "[REDACTED]" in sanitized
