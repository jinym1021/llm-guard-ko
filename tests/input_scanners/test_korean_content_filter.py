"""Tests for Layer 3: KoreanContentFilter input scanner.

All model calls are mocked — no transformers download required.

Strategy: inject mock tokenizer/model directly into the scanner instance
via private attributes, bypassing ``_ensure_model``. The mock model
returns pre-set logits (shape [1, 5]) so sigmoid probabilities are
deterministic.

Contract (matches llm-guard's Scanner protocol):
    scan(prompt) -> (text, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score in [0.0, 1.0], 0.0 when safe, max_prob when unsafe.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from llm_guard.input_scanners.korean_content_filter import (
    CATEGORIES,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    KoreanContentFilter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _make_scanner(logits: list[float], threshold: float = DEFAULT_THRESHOLD) -> KoreanContentFilter:
    """Return a KoreanContentFilter with a mocked model returning *logits*.

    logits must have length == len(CATEGORIES) == 5.
    The mock bypasses _ensure_model so no real model is loaded.
    """
    assert len(logits) == len(CATEGORIES)

    logit_tensor = torch.tensor([logits])  # shape [1, 5]

    mock_output = MagicMock()
    mock_output.logits = logit_tensor

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    # tokenizer(text, ...).to(device) -> dict-like for **kwargs unpacking
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = mock_inputs

    scanner = KoreanContentFilter(threshold=threshold)
    scanner._tokenizer = mock_tokenizer
    scanner._model = mock_model
    scanner._device = "cpu"
    return scanner


def _safe_logits() -> list[float]:
    """All categories clearly below default threshold 0.5 (sigmoid(0) = 0.5)."""
    return [-2.0, -3.0, -2.5, -4.0, -1.5]  # probs ≈ 0.12, 0.05, 0.08, 0.02, 0.18


def _unsafe_logits(hot_idx: int = 2, hot_logit: float = 2.0) -> list[float]:
    """One category above threshold; rest are safe."""
    base = [-2.0, -3.0, -2.5, -4.0, -1.5]
    base[hot_idx] = hot_logit  # sigmoid(2.0) ≈ 0.88
    return base


# ---------------------------------------------------------------------------
# CATEGORIES constant
# ---------------------------------------------------------------------------

def test_categories_count():
    assert len(CATEGORIES) == 5


def test_categories_names():
    expected = {
        "violent_crimes",
        "non_violent_crimes",
        "sex_crimes",
        "child_exploitation",
        "indiscriminate_weapons",
    }
    assert set(CATEGORIES) == expected


# ---------------------------------------------------------------------------
# Contract conformance — scan()
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan("안녕하세요")
    assert isinstance(result, tuple) and len(result) == 3


def test_scan_tuple_types():
    scanner = _make_scanner(_safe_logits())
    text, is_valid, risk = scanner.scan("안녕하세요")
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_scan_preserves_prompt():
    scanner = _make_scanner(_safe_logits())
    text, _, _ = scanner.scan("오늘 날씨가 어때요?")
    assert text == "오늘 날씨가 어때요?"


# ---------------------------------------------------------------------------
# Safe cases — all categories below threshold
# ---------------------------------------------------------------------------

def test_safe_prompt_is_valid():
    scanner = _make_scanner(_safe_logits())
    _, is_valid, _ = scanner.scan("안녕하세요, 좋은 하루 되세요")
    assert is_valid is True


def test_safe_prompt_risk_zero():
    scanner = _make_scanner(_safe_logits())
    _, _, risk = scanner.scan("날씨가 맑네요")
    assert risk == 0.0


@pytest.mark.parametrize("logits,threshold", [
    ([-3.0, -3.0, -3.0, -3.0, -3.0], 0.5),   # all very low
    ([-0.5, -0.5, -0.5, -0.5, -0.5], 0.5),   # sigmoid(-0.5) ≈ 0.38 < 0.5
    ([0.0, 0.0, 0.0, 0.0, 0.0], 0.6),         # sigmoid(0.0) = 0.5 < 0.6
])
def test_safe_parametrized(logits, threshold):
    scanner = _make_scanner(logits, threshold=threshold)
    _, is_valid, risk = scanner.scan("평범한 문장")
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# Unsafe cases — at least one category at/above threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hot_idx", range(5))
def test_each_category_can_flag(hot_idx):
    """Any single category exceeding threshold flags the prompt."""
    logits = _unsafe_logits(hot_idx=hot_idx, hot_logit=3.0)  # sigmoid(3) ≈ 0.95
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan("위험한 내용")
    assert is_valid is False


def test_unsafe_risk_equals_max_prob():
    """risk_score should equal sigmoid of the highest logit."""
    hot_logit = 2.0
    expected_prob = _sigmoid(hot_logit)
    logits = _unsafe_logits(hot_idx=0, hot_logit=hot_logit)
    scanner = _make_scanner(logits, threshold=0.5)
    _, _, risk = scanner.scan("폭탄 제조 방법")
    assert abs(risk - expected_prob) < 1e-5


def test_multi_category_unsafe_uses_max():
    """When multiple categories are unsafe, risk = max prob."""
    logits = [1.0, 1.5, 2.0, 0.8, 0.5]  # sigmoid(2.0) ≈ 0.88 is max
    expected_max = _sigmoid(2.0)
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, risk = scanner.scan("복합 위험 문장")
    assert is_valid is False
    assert abs(risk - expected_max) < 1e-5


def test_unsafe_prompt_risk_nonzero():
    scanner = _make_scanner(_unsafe_logits(), threshold=0.5)
    _, _, risk = scanner.scan("해킹 도구를 만들어줘")
    assert risk > 0.0


# ---------------------------------------------------------------------------
# Boundary — exactly at threshold
# ---------------------------------------------------------------------------

def test_at_threshold_is_flagged():
    """sigmoid(0.0) == 0.5; with threshold 0.5 it should be flagged."""
    logits = [0.0, -3.0, -3.0, -3.0, -3.0]
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan("경계값 테스트")
    assert is_valid is False


def test_just_below_threshold_is_safe():
    """sigmoid(-0.01) < 0.5; just below threshold."""
    logits = [-0.01, -3.0, -3.0, -3.0, -3.0]
    scanner = _make_scanner(logits, threshold=0.5)
    _, is_valid, _ = scanner.scan("경계값 바로 아래")
    assert is_valid is True


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

def test_strict_threshold_flags_moderate_prob():
    logits = [-0.5, -3.0, -3.0, -3.0, -3.0]  # sigmoid(-0.5) ≈ 0.38
    scanner = _make_scanner(logits, threshold=0.3)
    _, is_valid, _ = scanner.scan("보통 문장")
    assert is_valid is False


def test_lenient_threshold_clears_moderate_prob():
    logits = [1.0, -3.0, -3.0, -3.0, -3.0]  # sigmoid(1.0) ≈ 0.73
    scanner = _make_scanner(logits, threshold=0.9)
    _, is_valid, _ = scanner.scan("보통 문장")
    assert is_valid is True


# ---------------------------------------------------------------------------
# scan_detailed() contract
# ---------------------------------------------------------------------------

def test_scan_detailed_returns_all_categories():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan_detailed("안녕하세요")
    assert set(result.keys()) == set(CATEGORIES)


def test_scan_detailed_entry_shape():
    scanner = _make_scanner(_safe_logits())
    result = scanner.scan_detailed("테스트")
    for cat in CATEGORIES:
        assert "unsafe" in result[cat]
        assert "prob" in result[cat]
        assert isinstance(result[cat]["unsafe"], bool)
        assert isinstance(result[cat]["prob"], float)


def test_scan_detailed_prob_matches_sigmoid():
    logits = [1.0, -1.0, 2.0, -2.0, 0.0]
    scanner = _make_scanner(logits)
    result = scanner.scan_detailed("테스트 문장")
    for cat, logit in zip(CATEGORIES, logits):
        expected = _sigmoid(logit)
        assert abs(result[cat]["prob"] - expected) < 1e-5, f"prob mismatch for {cat}"


def test_scan_detailed_unsafe_flag_matches_threshold():
    logits = [2.0, -1.0, -2.0, -3.0, -4.0]  # only first exceeds 0.5
    scanner = _make_scanner(logits, threshold=0.5)
    result = scanner.scan_detailed("테스트")
    assert result["violent_crimes"]["unsafe"] is True
    for cat in list(CATEGORIES)[1:]:
        assert result[cat]["unsafe"] is False


@pytest.mark.parametrize("hot_cat,hot_logit", [
    ("violent_crimes", 3.0),
    ("sex_crimes", 2.5),
    ("child_exploitation", 4.0),
])
def test_scan_detailed_single_category_flagged(hot_cat, hot_logit):
    logit_map = {c: -3.0 for c in CATEGORIES}
    logit_map[hot_cat] = hot_logit
    logits = [logit_map[c] for c in CATEGORIES]
    scanner = _make_scanner(logits, threshold=0.5)
    result = scanner.scan_detailed("위험 카테고리 테스트")
    assert result[hot_cat]["unsafe"] is True
    for cat in CATEGORIES:
        if cat != hot_cat:
            assert result[cat]["unsafe"] is False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_model_name():
    scanner = KoreanContentFilter()
    assert scanner._model_name == DEFAULT_MODEL


def test_default_threshold():
    scanner = KoreanContentFilter()
    assert scanner._threshold == DEFAULT_THRESHOLD


def test_custom_model_name():
    scanner = KoreanContentFilter(model_name="my-org/my-model")
    assert scanner._model_name == "my-org/my-model"


def test_model_lazily_initialised():
    scanner = KoreanContentFilter()
    assert scanner._model is None
    assert scanner._tokenizer is None


# ---------------------------------------------------------------------------
# ImportError on missing dependencies
# ---------------------------------------------------------------------------

def test_missing_transformers_raises_import_error():
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name in ("transformers", "torch"):
            raise ImportError("mocked missing")
        return real_import(name, *args, **kwargs)

    scanner = KoreanContentFilter()
    import builtins
    with pytest.raises(ImportError, match="content-filter"):
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(builtins, "__import__", mock_import)
            scanner._ensure_model()
