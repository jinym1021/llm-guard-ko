"""Tests for Layer 1: KoreanPII regex scanner.

Contract (matches llm-guard's Scanner protocol):
    scan(prompt) -> (sanitized_text, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score in [0.0, 1.0], 0 safe, 1 risky.
"""

from __future__ import annotations

import pytest

from llm_guard_ko import KoreanPII


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = KoreanPII()
    result = scanner.scan("hello")
    assert isinstance(result, tuple) and len(result) == 3
    text, is_valid, risk = result
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_benign_input_is_valid_and_zero_risk():
    scanner = KoreanPII()
    text, is_valid, risk = scanner.scan("오늘 날씨가 어때?")
    assert is_valid is True
    assert risk == 0.0
    assert text == "오늘 날씨가 어때?"


# ---------------------------------------------------------------------------
# PII detection — SHOULD_FLAG
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt,label",
    [
        ("내 주민번호는 901010-1234567이야", "rrn"),
        ("전화번호는 010-1234-5678", "phone_mobile"),
        ("연락처 02-123-4567 입니다", "phone_landline"),
        ("사업자등록번호 123-45-67890", "biz_reg"),
        ("계좌번호 123-456-789012", "bank"),
        ("카드번호 1234-5678-9012-3456", "card"),
    ],
)
def test_korean_pii_is_flagged_and_redacted(prompt, label):
    scanner = KoreanPII(redact=True)
    text, is_valid, risk = scanner.scan(prompt)
    assert is_valid is False, f"{label!r} should be flagged"
    assert risk == 1.0
    assert "[REDACTED]" in text
    # The original identifier should be gone from sanitized text.
    assert prompt != text


def test_redact_false_keeps_original_text_but_still_flags():
    scanner = KoreanPII(redact=False)
    text, is_valid, risk = scanner.scan("내 주민번호는 901010-1234567")
    assert is_valid is False
    assert risk == 1.0
    assert "901010-1234567" in text


def test_overlapping_patterns_redact_once_and_preserve_surrounding_text():
    """Regression: two patterns matched the same phone number span and
    redaction ran twice, corrupting text after the match."""
    scanner = KoreanPII(redact=True)
    text, is_valid, _ = scanner.scan("연락처 010-1234-5678 로 연락해줘")
    assert is_valid is False
    assert "연락처" in text
    assert "로" in text
    assert "연락해줘" in text
    assert text.count("[REDACTED]") == 1


def test_multiple_distinct_pii_in_one_prompt_all_redacted():
    scanner = KoreanPII(redact=True)
    text, is_valid, _ = scanner.scan(
        "홍길동 주민번호 901010-1234567 전화 010-1234-5678"
    )
    assert is_valid is False
    assert "901010-1234567" not in text
    assert "010-1234-5678" not in text
    assert text.count("[REDACTED]") == 2


# ---------------------------------------------------------------------------
# SHOULD_PASS — avoid false positives
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "오늘 날씨가 어때?",
        "파이썬 코드 리뷰 해줘",
        "한국 역사에 대해 알려줘",
        "2024년은 좋은 해였다",  # year numbers shouldn't match RRN
        "버전 1.2.3 을 사용해",
    ],
)
def test_benign_korean_inputs_pass(prompt):
    scanner = KoreanPII()
    _, is_valid, risk = scanner.scan(prompt)
    assert is_valid is True, f"false positive on: {prompt!r}"
    assert risk == 0.0


# ---------------------------------------------------------------------------
# llm-guard Scanner protocol duck-typing — the real integration test.
# ---------------------------------------------------------------------------

def test_conforms_to_llm_guard_scanner_protocol():
    """KoreanPII should be usable wherever llm-guard expects a Scanner."""
    pytest.importorskip(
        "llm_guard",
        reason="llm-guard not installed; integration test skipped",
    )
    from llm_guard.input_scanners.base import Scanner  # noqa: F401

    scanner = KoreanPII()
    # Scanner is a typing.Protocol; at minimum the scan signature must
    # exist, be callable, and return the 3-tuple shape.
    assert callable(getattr(scanner, "scan", None))
    result = scanner.scan("test")
    assert len(result) == 3
    text, is_valid, risk = result
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)
