"""Tests for the KoreanPipeline input orchestrator (Phase 4).

Strategy: inject mock sub-scanners directly via the constructor's
``pii``, ``injection``, ``semantic``, ``content_filter`` kwargs.
Each mock scanner is a simple object with a ``scan()`` method that
returns preset ``(text, is_valid, risk_score)`` tuples. No real model
loading occurs.

Escalation contract:
  - Layer 1 runs always.
  - Layer 2 runs ONLY IF Layer 1 flagged.
  - Layer 3 runs ONLY IF Layer 2 flagged.
  - Higher layer clearing a flag → safe (L1 false positive handled).
  - PII redaction applied regardless of final verdict.
  - risk_score = Layer 3 max_prob when fully unsafe; 0.0 otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_guard.input_scanners.korean_pipeline import KoreanPipeline


# ---------------------------------------------------------------------------
# Mock sub-scanner helpers
# ---------------------------------------------------------------------------

class _MockScanner:
    """Minimal input scanner that returns a fixed result."""

    def __init__(self, text_out: str, is_valid: bool, risk: float) -> None:
        self._result = (text_out, is_valid, risk)
        self.call_count = 0

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        self.call_count += 1
        # Return the mock text_out (so PII-redacted text can flow through)
        return self._result


def _make_pipeline(
    *,
    pii_valid: bool = True,
    pii_out: str = "안녕하세요",
    pii_risk: float = 0.0,
    inj_valid: bool = True,
    inj_risk: float = 0.0,
    sem_valid: bool = True,
    sem_risk: float = 0.0,
    cf_valid: bool = True,
    cf_risk: float = 0.0,
) -> tuple[KoreanPipeline, dict]:
    """Return a pipeline with mocked sub-scanners and a tracker dict."""
    pii = _MockScanner(pii_out, pii_valid, pii_risk)
    injection = _MockScanner(pii_out, inj_valid, inj_risk)  # text flows through
    semantic = _MockScanner(pii_out, sem_valid, sem_risk)
    content_filter = _MockScanner(pii_out, cf_valid, cf_risk)

    pipeline = KoreanPipeline(
        pii=pii,
        injection=injection,
        semantic=semantic,
        content_filter=content_filter,
    )
    trackers = {
        "pii": pii,
        "injection": injection,
        "semantic": semantic,
        "content_filter": content_filter,
    }
    return pipeline, trackers


PROMPT = "테스트 프롬프트입니다"


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    pipeline, _ = _make_pipeline()
    result = pipeline.scan(PROMPT)
    assert isinstance(result, tuple) and len(result) == 3


def test_scan_types():
    pipeline, _ = _make_pipeline()
    text, is_valid, risk = pipeline.scan(PROMPT)
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


# ---------------------------------------------------------------------------
# Scenario 1: All Layer 1 safe → returns immediately, no L2/L3
# ---------------------------------------------------------------------------

def test_l1_safe_returns_safe():
    pipeline, _ = _make_pipeline(pii_valid=True, inj_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_safe_skips_l2_and_l3():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    pipeline.scan(PROMPT)
    assert t["semantic"].call_count == 0
    assert t["content_filter"].call_count == 0


def test_l1_safe_pii_scanner_runs():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    pipeline.scan(PROMPT)
    assert t["pii"].call_count == 1


def test_l1_safe_injection_scanner_runs():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    pipeline.scan(PROMPT)
    assert t["injection"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 2: Layer 1 (PII) flags, Layer 2 clears → safe, L3 not called
# ---------------------------------------------------------------------------

def test_l1_pii_flag_l2_clear_is_safe():
    pipeline, _ = _make_pipeline(pii_valid=False, pii_risk=0.9, inj_valid=True, sem_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_pii_flag_l2_clear_skips_l3():
    pipeline, t = _make_pipeline(pii_valid=False, inj_valid=True, sem_valid=True)
    pipeline.scan(PROMPT)
    assert t["content_filter"].call_count == 0


# ---------------------------------------------------------------------------
# Scenario 3: Layer 1 (injection) flags, Layer 2 clears → safe
# ---------------------------------------------------------------------------

def test_l1_inj_flag_l2_clear_is_safe():
    pipeline, _ = _make_pipeline(pii_valid=True, inj_valid=False, inj_risk=0.8, sem_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_inj_flag_l2_clear_skips_l3():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=False, sem_valid=True)
    pipeline.scan(PROMPT)
    assert t["content_filter"].call_count == 0


# ---------------------------------------------------------------------------
# Scenario 4: Both Layer 1 flags, Layer 2 clears → safe
# ---------------------------------------------------------------------------

def test_l1_both_flag_l2_clear_is_safe():
    pipeline, _ = _make_pipeline(pii_valid=False, inj_valid=False, sem_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT)
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# Scenario 5: L1 and L2 flag, Layer 3 clears → safe, risk = 0.0
# ---------------------------------------------------------------------------

def test_l1_l2_flag_l3_clear_is_safe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=True,
        sem_valid=False, sem_risk=0.85,
        cf_valid=True,
    )
    _, is_valid, risk = pipeline.scan(PROMPT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_l2_flag_l3_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, inj_valid=True,
        sem_valid=False,
        cf_valid=True,
    )
    pipeline.scan(PROMPT)
    assert t["content_filter"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 6: All three layers flag → unsafe
# ---------------------------------------------------------------------------

def test_all_flag_is_unsafe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.92,
    )
    _, is_valid, _ = pipeline.scan(PROMPT)
    assert is_valid is False


def test_all_flag_risk_from_l3():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.92,
    )
    _, _, risk = pipeline.scan(PROMPT)
    assert abs(risk - 0.92) < 1e-9


def test_all_flag_all_layers_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False,
    )
    pipeline.scan(PROMPT)
    assert t["pii"].call_count == 1
    assert t["injection"].call_count == 1
    assert t["semantic"].call_count == 1
    assert t["content_filter"].call_count == 1


# ---------------------------------------------------------------------------
# PII redaction flows through
# ---------------------------------------------------------------------------

def test_pii_redacted_text_returned():
    """The sanitized text from PII scanner is returned even when safe."""
    pipeline, _ = _make_pipeline(pii_out="[REDACTED] 씩씩하게 일해요", pii_valid=True, inj_valid=True)
    text, _, _ = pipeline.scan("주민번호 940101-1234567 씩씩하게 일해요")
    assert text == "[REDACTED] 씩씩하게 일해요"


def test_pii_redacted_text_returned_when_unsafe():
    """Redacted text is returned as first element even in the unsafe case."""
    redacted = "[주민번호 REDACTED] 폭탄 만드는 방법"
    pipeline, _ = _make_pipeline(
        pii_out=redacted,
        pii_valid=False,
        inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.95,
    )
    text, _, _ = pipeline.scan("940101-1234567 폭탄 만드는 방법")
    assert text == redacted


# ---------------------------------------------------------------------------
# Escalation efficiency — L2 runs exactly once when L1 flags
# ---------------------------------------------------------------------------

def test_l2_called_once_when_l1_flags():
    pipeline, t = _make_pipeline(pii_valid=False, inj_valid=True, sem_valid=False, cf_valid=True)
    pipeline.scan(PROMPT)
    assert t["semantic"].call_count == 1


def test_l2_not_called_multiple_times():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    for _ in range(3):
        pipeline.scan(PROMPT)
    assert t["semantic"].call_count == 0


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_default_construction_no_args():
    """KoreanPipeline() builds without error using default sub-scanners."""
    pipeline = KoreanPipeline()
    assert pipeline._pii is not None
    assert pipeline._injection is not None
    assert pipeline._semantic is not None
    assert pipeline._content_filter is not None


def test_custom_thresholds_forwarded():
    pipeline = KoreanPipeline(semantic_threshold=0.6, cf_threshold=0.4)
    assert pipeline._semantic._threshold == 0.6
    assert pipeline._content_filter._threshold == 0.4


def test_injected_scanners_used():
    """Pre-built scanners passed in are not replaced."""
    mock_pii = _MockScanner("text", True, 0.0)
    pipeline = KoreanPipeline(pii=mock_pii)
    assert pipeline._pii is mock_pii
