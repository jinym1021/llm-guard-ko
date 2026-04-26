"""Tests for Layer 2: KoreanSemantic output scanner.

All model calls are mocked — no sentence-transformers install required.

Strategy: inject mock model directly into the scanner instance via
``_model`` and ``_seed_embeddings`` private attributes, bypassing the
lazy ``_ensure_model`` call.

Contract (matches llm-guard's output Scanner protocol):
    scan(prompt, output) -> (output, is_valid, risk_score)
    is_valid == True  -> safe
    is_valid == False -> unsafe / flagged
    risk_score in [0.0, 1.0], 0 safe, 1 risky.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llm_guard.output_scanners.korean_semantic import KoreanSemantic
from llm_guard.input_scanners.korean_semantic import DEFAULT_MODEL, DEFAULT_THRESHOLD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner_with_sim(sim: float, threshold: float = DEFAULT_THRESHOLD) -> KoreanSemantic:
    """Return a KoreanSemantic output scanner whose scan() produces max_sim == sim.

    Uses a single 2-D seed so cosine similarity is exact:
      seed   = [1, 0]
      output = [sim, sqrt(1 - sim^2)]  (unit vector → cos = sim)
    """
    seed_embs = np.array([[1.0, 0.0]])

    if sim == 0.0:
        output_emb = np.array([0.0, 1.0])
    else:
        perp = float(np.sqrt(max(0.0, 1.0 - sim**2)))
        output_emb = np.array([float(sim), perp])

    mock_model = MagicMock()
    mock_model.encode.return_value = output_emb

    scanner = KoreanSemantic(threshold=threshold, seed_sentences=["dummy"])
    scanner._model = mock_model
    scanner._seed_embeddings = seed_embs
    return scanner


# ---------------------------------------------------------------------------
# Contract conformance
# ---------------------------------------------------------------------------

def test_scan_returns_three_tuple():
    scanner = _make_scanner_with_sim(0.5)
    result = scanner.scan("질문입니다", "응답입니다")
    assert isinstance(result, tuple) and len(result) == 3
    text, is_valid, risk = result
    assert isinstance(text, str)
    assert isinstance(is_valid, bool)
    assert isinstance(risk, float)


def test_scan_returns_output_not_prompt():
    """First element of the tuple must be the output, not the prompt."""
    scanner = _make_scanner_with_sim(0.3)
    text, _, _ = scanner.scan("프롬프트", "LLM 응답")
    assert text == "LLM 응답"


def test_prompt_is_ignored_for_risk():
    """Risk is assessed on output; same output with different prompts = same result."""
    scanner = _make_scanner_with_sim(0.8, threshold=0.75)
    _, is_valid_a, risk_a = scanner.scan("프롬프트 A", "동일한 응답")
    _, is_valid_b, risk_b = scanner.scan("프롬프트 B", "동일한 응답")
    assert is_valid_a == is_valid_b
    assert abs(risk_a - risk_b) < 1e-9


# ---------------------------------------------------------------------------
# Below-threshold: safe
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim", [0.0, 0.3, 0.74])
def test_below_threshold_is_valid(sim):
    scanner = _make_scanner_with_sim(sim, threshold=0.75)
    _, is_valid, risk = scanner.scan("질문", "안전한 응답입니다")
    assert is_valid is True
    assert risk == 0.0


# ---------------------------------------------------------------------------
# At/above threshold: flagged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sim", [0.75, 0.85, 1.0])
def test_at_or_above_threshold_is_flagged(sim):
    scanner = _make_scanner_with_sim(sim, threshold=0.75)
    _, is_valid, risk = scanner.scan("질문", "위험한 응답 내용")
    assert is_valid is False
    assert abs(risk - sim) < 1e-6


def test_risk_score_equals_max_similarity():
    scanner = _make_scanner_with_sim(0.92, threshold=0.75)
    _, _, risk = scanner.scan("질문", "폭탄 만드는 방법이에요")
    assert abs(risk - 0.92) < 1e-6


# ---------------------------------------------------------------------------
# Custom threshold
# ---------------------------------------------------------------------------

def test_custom_threshold_strict():
    scanner = _make_scanner_with_sim(0.6, threshold=0.5)
    _, is_valid, _ = scanner.scan("질문", "응답")
    assert is_valid is False


def test_custom_threshold_lenient():
    scanner = _make_scanner_with_sim(0.6, threshold=0.9)
    _, is_valid, _ = scanner.scan("질문", "안전한 응답")
    assert is_valid is True


# ---------------------------------------------------------------------------
# Custom seed sentences
# ---------------------------------------------------------------------------

def test_custom_seeds_used():
    sim = 0.88
    custom_seeds = ["위험한 응답 내용"]
    seed_embs = np.array([[1.0, 0.0]])
    perp = float(np.sqrt(max(0.0, 1.0 - sim**2)))
    output_emb = np.array([sim, perp])  # unit vector → cos(output, seed[0]) = sim

    mock_model = MagicMock()
    mock_model.encode.return_value = output_emb

    scanner = KoreanSemantic(threshold=0.75, seed_sentences=custom_seeds)
    scanner._model = mock_model
    scanner._seed_embeddings = seed_embs

    _, is_valid, risk = scanner.scan("질문", "응답")
    assert is_valid is False
    assert abs(risk - sim) < 1e-6


# ---------------------------------------------------------------------------
# Lazy model initialisation
# ---------------------------------------------------------------------------

def test_model_not_loaded_on_construction():
    scanner = KoreanSemantic()
    assert scanner._model is None
    assert scanner._seed_embeddings is None


def test_ensure_model_called_on_first_scan():
    scanner = _make_scanner_with_sim(0.3)
    call_count = {"n": 0}
    original = scanner._ensure_model

    def counting_ensure():
        call_count["n"] += 1
        original()

    scanner._ensure_model = counting_ensure  # type: ignore[method-assign]
    scanner.scan("질문", "응답")
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Missing dependency
# ---------------------------------------------------------------------------

def test_import_error_without_sentence_transformers():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        scanner = KoreanSemantic()
        with pytest.raises(ImportError, match="semantic"):
            scanner.scan("질문", "응답")


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

def test_empty_output():
    scanner = _make_scanner_with_sim(0.0)
    text, is_valid, risk = scanner.scan("질문", "")
    assert text == ""
    assert is_valid is True
    assert risk == 0.0


def test_empty_prompt_with_unsafe_output():
    scanner = _make_scanner_with_sim(0.9, threshold=0.75)
    _, is_valid, risk = scanner.scan("", "위험한 응답")
    assert is_valid is False


def test_boundary_exactly_at_threshold():
    scanner = _make_scanner_with_sim(0.75, threshold=0.75)
    _, is_valid, risk = scanner.scan("질문", "경계값 응답")
    assert is_valid is False
    assert abs(risk - 0.75) < 1e-6
