"""Tests for the KoreanPipeline output orchestrator.

Strategy: inject mock sub-scanners via constructor kwargs so no real
model is loaded. Output-scanner mocks accept ``(prompt, output)`` and
return preset ``(text, is_valid, risk_score)`` tuples.

Output-scanner contract:
    scan(prompt, output) -> (sanitized_output, is_valid, risk_score)
    - Classifies the LLM *response* (output), not the prompt.
    - PII is redacted from the response regardless of the final verdict.
    - risk_score == 0.0 when safe; == max Layer-3 risk when fully unsafe.

Layer layout (matches llm_guard.output_scanners.korean_pipeline):
    L1: pii, toxicity, no_refusal      (regex/heuristic)
    L2: sensitive, semantic            (light AI)
    L3: factual_consistency?, llm_judge (deep / LLM)
factual_consistency is optional: when None, only llm_judge runs in L3.
"""

from __future__ import annotations

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
    pii_valid: bool = True, pii_out: str = "안전한 응답입니다", pii_risk: float = 0.0,
    tox_valid: bool = True, tox_risk: float = 0.0,
    ref_valid: bool = True, ref_risk: float = 0.0,
    sens_valid: bool = True, sens_risk: float = 0.0,
    sem_valid: bool = True, sem_risk: float = 0.0,
    fac_valid: bool = True, fac_risk: float = 0.0,
    judge_valid: bool = True, judge_risk: float = 0.0,
    use_factual: bool = True,
) -> tuple[KoreanPipeline, dict]:
    pii = _MockOutputScanner(pii_out, pii_valid, pii_risk)
    toxicity = _MockOutputScanner(pii_out, tox_valid, tox_risk)
    no_refusal = _MockOutputScanner(pii_out, ref_valid, ref_risk)
    sensitive = _MockOutputScanner(pii_out, sens_valid, sens_risk)
    semantic = _MockOutputScanner(pii_out, sem_valid, sem_risk)
    factual = _MockOutputScanner(pii_out, fac_valid, fac_risk) if use_factual else None
    judge = _MockOutputScanner(pii_out, judge_valid, judge_risk)

    pipeline = KoreanPipeline(
        pii=pii,
        toxicity=toxicity,
        no_refusal=no_refusal,
        sensitive=sensitive,
        semantic=semantic,
        factual_consistency=factual,
        llm_judge=judge,
    )
    trackers = {
        "pii": pii, "toxicity": toxicity, "no_refusal": no_refusal,
        "sensitive": sensitive, "semantic": semantic,
        "factual_consistency": factual, "llm_judge": judge,
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


def test_scan_returns_sanitized_output():
    pipeline, _ = _make_pipeline(pii_out="응답 텍스트")
    text, _, _ = pipeline.scan(PROMPT, OUTPUT)
    assert text == "응답 텍스트"


# ---------------------------------------------------------------------------
# Scenario 1: All Layer 1 safe → no L2/L3
# ---------------------------------------------------------------------------

def test_l1_safe_is_valid():
    pipeline, _ = _make_pipeline()
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert risk == 0.0


def test_l1_safe_skips_l2_l3():
    pipeline, t = _make_pipeline()
    pipeline.scan(PROMPT, OUTPUT)
    assert t["sensitive"].call_count == 0
    assert t["semantic"].call_count == 0
    assert t["factual_consistency"].call_count == 0
    assert t["llm_judge"].call_count == 0


def test_l1_always_runs_all_three_scanners():
    pipeline, t = _make_pipeline()
    pipeline.scan(PROMPT, OUTPUT)
    assert t["pii"].call_count == 1
    assert t["toxicity"].call_count == 1
    assert t["no_refusal"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 2: L1 flags, L2 clears → safe, L3 not called
# ---------------------------------------------------------------------------

def test_l1_pii_flag_l2_clear_safe():
    pipeline, t = _make_pipeline(pii_valid=False, pii_risk=0.9)
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True and risk == 0.0
    assert t["llm_judge"].call_count == 0


def test_l1_tox_flag_l2_clear_safe():
    pipeline, t = _make_pipeline(tox_valid=False, tox_risk=0.7)
    _, is_valid, _ = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert t["llm_judge"].call_count == 0


def test_l1_refusal_flag_l2_clear_safe():
    pipeline, t = _make_pipeline(ref_valid=False, ref_risk=0.6)
    _, is_valid, _ = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert t["llm_judge"].call_count == 0


# ---------------------------------------------------------------------------
# Scenario 3: L1 + L2 flag, L3 clears → safe
# ---------------------------------------------------------------------------

def test_l1_l2_flag_l3_clear_safe():
    pipeline, t = _make_pipeline(
        pii_valid=False,
        sens_valid=False, sem_valid=True,
        fac_valid=True, judge_valid=True,
    )
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True and risk == 0.0
    assert t["llm_judge"].call_count == 1


def test_l1_l2_flag_l3_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, sem_valid=False,
        fac_valid=True, judge_valid=True,
    )
    pipeline.scan(PROMPT, OUTPUT)
    assert t["factual_consistency"].call_count == 1
    assert t["llm_judge"].call_count == 1


# ---------------------------------------------------------------------------
# Scenario 4: All three layers flag → unsafe
# ---------------------------------------------------------------------------

def test_all_flag_unsafe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, tox_valid=False, ref_valid=False,
        sens_valid=False, sem_valid=False,
        fac_valid=False, fac_risk=0.7,
        judge_valid=False, judge_risk=0.95,
    )
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is False
    assert abs(risk - 0.95) < 1e-9


def test_all_flag_all_layers_called():
    pipeline, t = _make_pipeline(
        pii_valid=False, tox_valid=False, ref_valid=False,
        sens_valid=False, sem_valid=False,
        fac_valid=False, judge_valid=False,
    )
    pipeline.scan(PROMPT, OUTPUT)
    for name, scanner in t.items():
        assert scanner.call_count == 1, f"{name} should have been called once"


# ---------------------------------------------------------------------------
# Optional factual_consistency
# ---------------------------------------------------------------------------

def test_factual_consistency_optional_skips_when_none():
    """When factual_consistency=None, only llm_judge runs in L3."""
    pipeline, t = _make_pipeline(
        pii_valid=False, sem_valid=False,
        judge_valid=True,
        use_factual=False,
    )
    _, is_valid, _ = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is True
    assert t["factual_consistency"] is None
    assert t["llm_judge"].call_count == 1


def test_factual_consistency_none_judge_flags_unsafe():
    pipeline, _ = _make_pipeline(
        pii_valid=False, sem_valid=False,
        judge_valid=False, judge_risk=0.88,
        use_factual=False,
    )
    _, is_valid, risk = pipeline.scan(PROMPT, OUTPUT)
    assert is_valid is False
    assert abs(risk - 0.88) < 1e-9


# ---------------------------------------------------------------------------
# Prompt forwarded
# ---------------------------------------------------------------------------

def test_prompt_forwarded_to_sub_scanners():
    received = []

    class _Tracking:
        def scan(self, prompt, output):
            received.append(prompt)
            return (output, True, 0.0)

    pipeline = KoreanPipeline(
        pii=_Tracking(), toxicity=_Tracking(), no_refusal=_Tracking(),
    )
    pipeline.scan("특정 프롬프트", "응답")
    assert all(p == "특정 프롬프트" for p in received)


# ---------------------------------------------------------------------------
# PII redaction flows through
# ---------------------------------------------------------------------------

def test_pii_redacted_output_returned_when_safe():
    redacted = "[주민번호 REDACTED] 안전한 내용"
    pipeline, _ = _make_pipeline(pii_out=redacted)
    text, _, _ = pipeline.scan(PROMPT, "940101-1234567 안전한 내용")
    assert text == redacted


def test_pii_redacted_output_returned_when_unsafe():
    redacted = "[REDACTED] 위험한 내용"
    pipeline, _ = _make_pipeline(
        pii_out=redacted,
        pii_valid=False, sem_valid=False,
        judge_valid=False, judge_risk=0.91,
    )
    text, _, _ = pipeline.scan(PROMPT, "940101-1234567 위험한 내용")
    assert text == redacted


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------

def test_default_construction():
    pipeline = KoreanPipeline()
    assert pipeline._pii is not None
    assert pipeline._toxicity is not None
    assert pipeline._no_refusal is not None
    assert pipeline._sensitive is not None
    assert pipeline._semantic is not None
    # factual_consistency requires explicit model — should default to None.
    assert pipeline._factual_consistency is None
    assert pipeline._llm_judge is not None


def test_injected_scanners_used():
    mock = _MockOutputScanner("text", True, 0.0)
    pipeline = KoreanPipeline(pii=mock)
    assert pipeline._pii is mock
