"""Unit tests for the raw pattern dicts in patterns.korean."""

import re

from llm_guard_ko.patterns.korean import (
    KOREAN_INJECTION_PATTERNS,
    KOREAN_PII_PATTERNS,
)


def test_all_pii_patterns_compile():
    for label, pattern in KOREAN_PII_PATTERNS.items():
        re.compile(pattern), f"PII pattern {label!r} failed to compile"


def test_all_injection_patterns_compile():
    for label, pattern in KOREAN_INJECTION_PATTERNS.items():
        re.compile(pattern), f"Injection pattern {label!r} failed to compile"


def test_rrn_pattern_sanity():
    rrn = KOREAN_PII_PATTERNS["resident_registration_number"]
    assert re.search(rrn, "901010-1234567")
    assert not re.search(rrn, "90101-1234567")     # too few digits
    assert not re.search(rrn, "901010-9234567")    # 7th digit out of range
