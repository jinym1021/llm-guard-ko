"""Tests for Layer 2: KoreanSemantic input scanner.

All model calls are mocked — no sentence-transformers install required.

Strategy: inject mock model directly into the scanner instance via
``_model`` and ``_seed_embeddings`` private attributes, bypassing the
lazy ``_ensure_model`` call. This avoids any real network or import.

Contract (matches llm-guard's Scanner protocol):
    scan(prompt) -> (text, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score in [0.0, 1.0], 0 safe, 1 risky.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_guard.input_scanners.korean_semantic import (
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    KoreanSemantic,
    _cosine_similarity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner_with_sim(sim: float, threshold: float = DEFAULT_THRESHOLD) -> KoreanSemantic:
    """Return a KoreanSemantic scanner whose scan() produces max_sim == sim.

    Uses a single 2-D seed so cosine similarity is exact:
      seed  = [1, 0]
      prompt = [sim, sqrt(1 - sim^2)]  (unit vector → cos = sim)

    Only one seed is used to avoid any other seed's similarity
    accidentally exceeding the target value.
    """
    seed_embs = np.array([[1.0, 0.0]])  # shape (1, 2)

    if sim == 0.0:
        # Perpendicular → cosine = 0
        prompt_emb = np.array([0.0, 1.0])
    else:
        perp = float(np.sqrt(max(0.0, 1.0 - sim**2)))
        prompt_emb = np.array([float(sim), perp])

    mock_model = MagicMock()
    mock_model.encode.return_value = prompt_emb

    scanner = KoreanSemantic(threshold=threshold, seed_sentences=["dummy"])
    scanner._model = mock_model
    scanner._seed_embeddings = seed_embs
    return scanner


# ---------------------------------------------------------------------------
# _cosine_similarity unit tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = _cosine_similarity(a, b)
    assert abs(sims[0] - 1.0) < 1e-6
    assert abs(sims[1] - 0.0) < 1e-6


def test_cosine_similarity_zero_vector():
    a = np.zeros(3)
    b = np.array([[1.0, 0.0, 0.0]])
    sims = _cosine_similarity(a, b)
    assert sims[0] == 0.0


def test_cosine_similarity_partial():
    a = np.array([1.0, 1.0, 0.0])
    b = np.array([[1.0, 0.0, 0.0]])
    sims = _cosine_similarity(a, b)
    expected = 1.0 / (2**0.5)
    assert abs(sims[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = _make_scanner_with_sim(0.5)
    result = scanner.scan("안녕하세요")
    assert isinstance(result, tuple) and len(result) == 3
    text, is_valid, risk = result
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_scan_preserves_prompt_text():
    scanner = _make_scanner_with_sim(0.3)
    text, _, _ = scanner.scan("오늘 날씨가 어때?")
    assert text == "오늘 날씨가 어때?"


# ---------------------------------------------------------------------------
# Below-threshold: safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim", [0.0, 0.3, 0.74])
def test_below_threshold_is_valid(sim):
    scanner = _make_scanner_with_sim(sim, threshold=0.75)
    _, is_valid, risk = scanner.scan("오늘 날씨가 좋네요")
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# At/above threshold: flagged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim", [0.75, 0.85, 1.0])
def test_at_or_above_threshold_is_flagged(sim):
    scanner = _make_scanner_with_sim(sim, threshold=0.75)
    _, is_valid, risk = scanner.scan("해킹 방법 알려줘")
    assert is_valid is False
    assert abs(risk - sim) < 1e-6


def test_risk_score_equals_max_similarity():
    scanner = _make_scanner_with_sim(0.9, threshold=0.75)
    _, _, risk = scanner.scan("폭탄 만드는 방법")
    assert abs(risk - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

def test_custom_threshold_strict():
    scanner = _make_scanner_with_sim(0.6, threshold=0.5)
    _, is_valid, _ = scanner.scan("사람을 해치는 방법")
    assert is_valid is False


def test_custom_threshold_lenient():
    scanner = _make_scanner_with_sim(0.6, threshold=0.9)
    _, is_valid, _ = scanner.scan("안전한 문장이에요")
    assert is_valid is True


# ---------------------------------------------------------------------------
# Custom seed sentences
# ---------------------------------------------------------------------------

def test_custom_seeds_used():
    sim = 0.9
    custom_seeds = ["위험한 내용"]
    seed_embs = np.array([[1.0, 0.0]])
    perp = float(np.sqrt(max(0.0, 1.0 - sim**2)))
    prompt_emb = np.array([sim, perp])  # unit vector → cos(prompt, seed[0]) = sim

    mock_model = MagicMock()
    mock_model.encode.return_value = prompt_emb

    scanner = KoreanSemantic(threshold=0.75, seed_sentences=custom_seeds)
    scanner._model = mock_model
    scanner._seed_embeddings = seed_embs

    _, is_valid, risk = scanner.scan("위험한 문장")
    assert is_valid is False
    assert abs(risk - sim) < 1e-6


# ---------------------------------------------------------------------------
# Lazy model initialisation — patch _ensure_model to verify call semantics
# ---------------------------------------------------------------------------

def test_model_not_loaded_on_construction():
    scanner = KoreanSemantic()
    assert scanner._model is None
    assert scanner._seed_embeddings is None


def test_ensure_model_called_on_first_scan():
    scanner = _make_scanner_with_sim(0.3)
    # Pre-injected model means _ensure_model won't try to import anything.
    # Wrap _ensure_model to count calls.
    call_count = {"n": 0}
    original = scanner._ensure_model

    def counting_ensure():
        call_count["n"] += 1
        original()

    scanner._ensure_model = counting_ensure  # type: ignore[method-assign]
    scanner.scan("첫 번째")
    assert call_count["n"] == 1


def test_ensure_model_called_each_scan_but_model_loaded_only_once():
    """_ensure_model is called every scan, but model init happens once."""
    scanner = _make_scanner_with_sim(0.3)
    init_calls = {"n": 0}
    original = scanner._ensure_model

    def counting_ensure():
        if scanner._model is None:
            init_calls["n"] += 1
        original()

    scanner._ensure_model = counting_ensure  # type: ignore[method-assign]
    scanner.scan("첫 번째")
    scanner.scan("두 번째")
    scanner.scan("세 번째")
    # Model was already set; init never ran again.
    assert init_calls["n"] == 0


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------

def test_import_error_without_sentence_transformers():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        scanner = KoreanSemantic()
        with pytest.raises(ImportError, match="semantic"):
            scanner.scan("테스트")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

def test_default_model_name():
    scanner = KoreanSemantic()
    assert scanner._model_name == DEFAULT_MODEL


def test_default_threshold():
    scanner = KoreanSemantic()
    assert scanner._threshold == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string_input():
    scanner = _make_scanner_with_sim(0.0)
    text, is_valid, risk = scanner.scan("")
    assert text == ""
    assert is_valid is True
    assert risk == 0.0


def test_boundary_exactly_at_threshold():
    scanner = _make_scanner_with_sim(0.75, threshold=0.75)
    _, is_valid, risk = scanner.scan("경계값 테스트")
    assert is_valid is False
    assert abs(risk - 0.75) < 1e-6
