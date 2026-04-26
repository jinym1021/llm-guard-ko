"""Tests for the KoreanPipeline output orchestrator (Phase 4).

Strategy: inject mock sub-scanners via constructor kwargs so no real
model is loaded. Output-scanner mocks accept ``(prompt, output)`` and
return preset ``(text, is_valid, risk_score)`` tuples.

Output-scanner contract:
    scan(prompt, output) -> (sanitized_output, is_valid, risk_score)
    - Classifies the LLM *response* (output), not the prompt.
    - PII redacted from response regardless of final verdict.
    - risk_score == 0.0 when safe; == Layer 3 max_prob when fully unsafe.
"""

from __future__ import annotations

import pytest

from llm_guard.output_scanners.korean_pipeline import KoreanPipeline


# ---------------------------------------------------------------------------
# Mock sub-scanner helpers
# ---------------------------------------------------------------------------

class _MockOutputScanner:
    """Minimal output scanner returning a fixed result."""

    def __init__(self, text_out: str, is_valid: bool, risk: float) -> None:
        self._result = (text_out, is_valid, risk)
        self.call_count = 0

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        self.call_count += 1
        return self._result


def _make_pipeline(
    *,
    pii_valid: bool = True,
    pii_out: str = "안전한 응답입니다",
    pii_risk: float = 0.0,
    inj_valid: bool = True,
    inj_risk: float = 0.0,
    sem_valid: bool = True,
    sem_risk: float = 0.0,
    cf_valid: bool = True,
    cf_risk: float = 0.0,
) -> tuple[KoreanPipeline, dict]:
    pii = _MockOutputScanner(pii_out, pii_valid, pii_risk)
    injection = _MockOutputScanner(pii_out, inj_valid, inj_risk)
    semantic = _MockOutputScanner(pii_out, sem_valid, sem_risk)
    content_filter = _MockOutputScanner(pii_out, cf_valid, cf_risk)

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


PROMPT = "무엇을 도와드릴까요?"
OUTPUT = "위험한 내용이 담긴 응답입니다"


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    pipeline, _ = _make_pipeline()
    result = pipeline.scan(PROMPT, OUTPUT)
    assert isinstance(result, tuple) and len(result) == 3


def test_scan_types():
    pipeline, _ = _make_pipeline()
    text, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_scan_returns_output_not_prompt():
    """First element is the (sanitized) response, not the prompt."""
    pipeline, _ = _make_pipeline(pii_out="응답 텍스트")
    text, _, _ = pipeline.scan(PROMPT, OUTPUT)
    assert text != PROMPT


# ---------------------------------------------------------------------------
# Scenario 1: All Layer 1 safe → no L2/L3
# ---------------------------------------------------------------------------

def test_l1_safe_is_valid():
    pipeline, _ = _make_pipeline(pii_valid=True, inj_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_safe_skips_l2_l3():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    pipeline.scan(PROMPT, OUTPUT)
    assert t["semantic"].call_count == 0
    assert t["content_filter"].call_count == 0


def test_l1_always_runs_both_scanners():
    pipeline, t = _make_pipeline(pii_valid=True, inj_valid=True)
    pipeline.scan(PROMPT, OUTPUT)
    assert t["pii"].call_count == 1
    assert t["injection"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 2: L1 PII flags, L2 clears → safe
# ---------------------------------------------------------------------------

def test_l1_pii_l2_clear_safe():
    pipeline, _ = _make_pipeline(pii_valid=False, pii_risk=0.9, inj_valid=True, sem_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_pii_l2_clear_skips_l3():
    pipeline, t = _make_pipeline(pii_valid=False, inj_valid=True, sem_valid=True)
    pipeline.scan(PROMPT, OUTPUT)
    assert t["content_filter"].call_count == 0


# ---------------------------------------------------------------------------
# Scenario 3: L1 injection flags, L2 clears → safe
# ---------------------------------------------------------------------------

def test_l1_inj_l2_clear_safe():
    pipeline, _ = _make_pipeline(pii_valid=True, inj_valid=False, inj_risk=0.7, sem_valid=True)
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# Scenario 4: L1 and L2 flag, L3 clears → safe
# ---------------------------------------------------------------------------

def test_l1_l2_flag_l3_clear_safe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=True,
        sem_valid=False, sem_risk=0.8,
        cf_valid=True,
    )
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_l2_flag_l3_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, inj_valid=True,
        sem_valid=False,
        cf_valid=True,
    )
    pipeline.scan(PROMPT, OUTPUT)
    assert t["content_filter"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 5: All three flag → unsafe
# ---------------------------------------------------------------------------

def test_all_flag_is_unsafe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.95,
    )
    _, is_valid, _ = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is False


def test_all_flag_risk_from_l3():
    pipeline, _ = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.95,
    )
    _, _, risk = pipeline.scan(PROMPT, OUTPUT)
    assert abs(risk - 0.95) < 1e-9


def test_all_flag_all_layers_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, inj_valid=False,
        sem_valid=False,
        cf_valid=False,
    )
    pipeline.scan(PROMPT, OUTPUT)
    for name, scanner in t.items():
        assert scanner.call_count == 1, f"{name} should have been called once"


# ---------------------------------------------------------------------------
# Prompt is passed to sub-scanners (not ignored by orchestrator)
# ---------------------------------------------------------------------------

def test_prompt_forwarded_to_sub_scanners():
    """Sub-scanners must receive the original prompt."""
    received_prompts = []

    class _TrackingScanner:
        def scan(self, prompt, output):
            received_prompts.append(prompt)
            return (output, True, 0.0)

    tracker = _TrackingScanner()
    pipeline = KoreanPipeline(
        pii=tracker,
        injection=_TrackingScanner(),
    )
    pipeline.scan("특정 프롬프트", "응답")
    assert received_prompts[0] == "특정 프롬프트"


# ---------------------------------------------------------------------------
# PII redaction flows through
# ---------------------------------------------------------------------------

def test_pii_redacted_output_returned():
    redacted = "[주민번호 REDACTED] 안전한 내용"
    pipeline, _ = _make_pipeline(pii_out=redacted, pii_valid=True, inj_valid=True)
    text, _, _ = pipeline.scan(PROMPT, "940101-1234567 안전한 내용")
    assert text == redacted


def test_pii_redacted_returned_when_unsafe():
    redacted = "[REDACTED] 위험한 내용"
    pipeline, _ = _make_pipeline(
        pii_out=redacted,
        pii_valid=False,
        inj_valid=False,
        sem_valid=False,
        cf_valid=False, cf_risk=0.91,
    )
    text, _, _ = pipeline.scan(PROMPT, "940101-1234567 위험한 내용")
    assert text == redacted


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_default_construction():
    pipeline = KoreanPipeline()
    assert pipeline._pii is not None
    assert pipeline._injection is not None
    assert pipeline._semantic is not None
    assert pipeline._content_filter is not None


def test_custom_thresholds_forwarded():
    pipeline = KoreanPipeline(semantic_threshold=0.65, cf_threshold=0.45)
    assert pipeline._semantic._threshold == 0.65          # KoreanSemantic direct attribute
    assert pipeline._content_filter._inner._threshold == 0.45  # KoreanContentFilter wraps _inner


def test_injected_scanners_used():
    mock_pii = _MockOutputScanner("text", True, 0.0)
    pipeline = KoreanPipeline(pii=mock_pii)
    assert pipeline._pii is mock_pii
