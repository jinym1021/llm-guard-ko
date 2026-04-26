"""Tests for Layer 3: KoreanContentFilter output scanner.

All model calls are mocked — no transformers download required.

Output-scanner contract:
    scan(prompt, output) -> (output_text, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score == 0.0 when safe, max_prob when unsafe.

The scanner classifies the LLM *response* (``output``), not the prompt.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from llm_guard.input_scanners.korean_content_filter import CATEGORIES, DEFAULT_THRESHOLD
from llm_guard.output_scanners.korean_content_filter import KoreanContentFilter


# ---------------------------------------------------------------------------
# Helpers (mirrors input test helpers, operates on _inner)
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_scanner(logits: list[float], threshold: float = DEFAULT_THRESHOLD) -> KoreanContentFilter:
    """Return a KoreanContentFilter (output) with mocked inner classifier."""
    assert len(logits) == len(CATEGORIES)

    logit_tensor = torch.tensor([logits])

    mock_output = MagicMock()
    mock_output.logits = logit_tensor

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_inputs

    scanner = KoreanContentFilter(threshold=threshold)
    scanner._inner._tokenizer = mock_tokenizer
    scanner._inner._model = mock_model
    scanner._inner._device = "cpu"
    return scanner


def _safe_logits() -> list[float]:
    return [-2.0, -3.0, -2.5, -4.0, -1.5]


def _unsafe_logits(hot_idx: int = 2, hot_logit: float = 2.0) -> list[float]:
    base = [-2.0, -3.0, -2.5, -4.0, -1.5]
    base[hot_idx] = hot_logit
    return base


DUMMY_PROMPT = "이건 테스트 프롬프트입니다."


# ---------------------------------------------------------------------------
# Contract conformance — scan(prompt, output)
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan(DUMMY_PROMPT, "안녕하세요")
    assert isinstance(result, tuple) and len(result) == 3


def test_scan_tuple_types():
    scanner = _make_scanner(_safe_logits())
    text, is_valid, risk = scanner.scan(DUMMY_PROMPT, "응답 텍스트")
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_scan_returns_output_not_prompt():
    """First element must be the LLM *output*, not the prompt."""
    scanner = _make_scanner(_safe_logits())
    output_text = "이것은 모델의 응답입니다."
    returned_text, _, _ = scanner.scan(DUMMY_PROMPT, output_text)
    assert returned_text == output_text
    assert returned_text != DUMMY_PROMPT


def test_scan_preserves_output_text():
    scanner = _make_scanner(_safe_logits())
    output = "오늘 날씨는 맑습니다."
    text, _, _ = scanner.scan(DUMMY_PROMPT, output)
    assert text == output


# ---------------------------------------------------------------------------
# Prompt is ignored — classification uses output only
# ---------------------------------------------------------------------------

def test_prompt_does_not_affect_result():
    """Same output, different prompts → identical result."""
    scanner = _make_scanner(_unsafe_logits())
    output = "해킹 도구를 만드는 방법입니다"

    _, valid1, risk1 = scanner.scan("무해한 질문", output)
    _, valid2, risk2 = scanner.scan("다른 질문", output)

    assert valid1 == valid2
    assert abs(risk1 - risk2) < 1e-9


# ---------------------------------------------------------------------------
# Safe outputs
# ---------------------------------------------------------------------------

def test_safe_output_is_valid():
    scanner = _make_scanner(_safe_logits())
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "안전한 응답입니다.")
    assert is_valid is True


def test_safe_output_risk_zero():
    scanner = _make_scanner(_safe_logits())
    _, _, risk = scanner.scan(DUMMY_PROMPT, "도움이 되는 내용입니다.")
    assert risk == 0.0


@pytest.mark.parametrize("logits,threshold", [
    ([-3.0, -3.0, -3.0, -3.0, -3.0], 0.5),
    ([-0.5, -0.5, -0.5, -0.5, -0.5], 0.5),
    ([0.0, 0.0, 0.0, 0.0, 0.0], 0.6),
])
def test_safe_parametrized(logits, threshold):
    scanner = _make_scanner(logits, threshold=threshold)
    _, is_valid, risk = scanner.scan(DUMMY_PROMPT, "평범한 응답")
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# Unsafe outputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hot_idx", range(5))
def test_each_category_can_flag_output(hot_idx):
    logits = _unsafe_logits(hot_idx=hot_idx, hot_logit=3.0)
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "위험한 응답 내용")
    assert is_valid is False


def test_unsafe_risk_equals_max_prob():
    hot_logit = 2.0
    expected = _sigmoid(hot_logit)
    logits = _unsafe_logits(hot_idx=1, hot_logit=hot_logit)
    scanner = _make_scanner(logits, threshold=0.5)
    _, _, risk = scanner.scan(DUMMY_PROMPT, "위험 응답")
    assert abs(risk - expected) < 1e-5


def test_multi_category_uses_max():
    logits = [1.0, 1.5, 2.0, 0.8, 0.5]
    expected_max = _sigmoid(2.0)
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, risk = scanner.scan(DUMMY_PROMPT, "복합 위험 응답")
    assert is_valid is False
    assert abs(risk - expected_max) < 1e-5


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------

def test_at_threshold_is_flagged():
    logits = [0.0, -3.0, -3.0, -3.0, -3.0]  # sigmoid(0) == 0.5
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "경계값 응답")
    assert is_valid is False


def test_just_below_threshold_is_safe():
    logits = [-0.01, -3.0, -3.0, -3.0, -3.0]
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "경계값 바로 아래 응답")
    assert is_valid is True


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

def test_strict_threshold():
    logits = [-0.5, -3.0, -3.0, -3.0, -3.0]  # sigmoid(-0.5) ≈ 0.38
    scanner = _make_scanner(logits, threshold=0.3)
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "보통 응답")
    assert is_valid is False


def test_lenient_threshold():
    logits = [1.0, -3.0, -3.0, -3.0, -3.0]  # sigmoid(1.0) ≈ 0.73
    scanner = _make_scanner(logits, threshold=0.9)
    _, is_valid, _ = scanner.scan(DUMMY_PROMPT, "보통 응답")
    assert is_valid is True


# ---------------------------------------------------------------------------
# scan_detailed() — output variant
# ---------------------------------------------------------------------------

def test_scan_detailed_returns_all_categories():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan_detailed("안녕하세요")
    assert set(result.keys()) == set(CATEGORIES)


def test_scan_detailed_entry_shape():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan_detailed("테스트 응답")
    for cat in CATEGORIES:
        assert "unsafe" in result[cat]
        assert "prob" in result[cat]
        assert isinstance(result[cat]["unsafe"], bool)
        assert isinstance(result[cat]["prob"], float)


def test_scan_detailed_prob_matches_sigmoid():
    logits = [1.0, -1.0, 2.0, -2.0, 0.0]
    scanner = _make_scanner(logits)
    result = scanner.scan_detailed("응답 텍스트")
    for cat, logit in zip(CATEGORIES, logits):
        expected = _sigmoid(logit)
        assert abs(result[cat]["prob"] - expected) < 1e-5


def test_scan_detailed_unsafe_flag():
    logits = [3.0, -3.0, -3.0, -3.0, -3.0]
    scanner = _make_scanner(logits, threshold=0.5)
    result = scanner.scan_detailed("응답")
    assert result["violent_crimes"]["unsafe"] is True
    for cat in list(CATEGORIES)[1:]:
        assert result[cat]["unsafe"] is False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_model_lazily_initialised():
    scanner = KoreanContentFilter()
    assert scanner._inner._model is None
    assert scanner._inner._tokenizer is None


def test_default_threshold_forwarded():
    scanner = KoreanContentFilter()
    assert scanner._inner._threshold == DEFAULT_THRESHOLD


def test_custom_threshold_forwarded():
    scanner = KoreanContentFilter(threshold=0.7)
    assert scanner._inner._threshold == 0.7


def test_custom_model_forwarded():
    scanner = KoreanContentFilter(model_name="my-org/model")
    assert scanner._inner._model_name == "my-org/model"
