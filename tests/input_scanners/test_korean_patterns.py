"""Unit tests for the raw pattern dicts in ``llm_guard.input_scanners.korean_patterns``."""

from __future__ import annotations

import re

from llm_guard.input_scanners.korean_patterns import (
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
    rrn = KOREAN_PII_PATTERNS["주민등록번호"]
    assert re.search(rrn, "901010-1234567")
    assert not re.search(rrn, "90101-1234567")     # too few digits


def test_new_injection_patterns_present():
    """v0.1 added two patterns the old sibling package never shipped."""
    assert "pretend_to_be" in KOREAN_INJECTION_PATTERNS
    assert "bypass_filter" in KOREAN_INJECTION_PATTERNS
