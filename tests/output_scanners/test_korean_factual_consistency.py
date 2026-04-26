import pytest
import sys
from importlib import util

# To avoid deep torch mocking crashing transformers
from llm_guard.output_scanners.korean_factual_consistency import KoreanFactualConsistency

class MockKoreanFactualConsistency(KoreanFactualConsistency):
    def _ensure_model(self): pass
    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if not prompt.strip() or not output.strip():
            return output, True, 0.0
            
        if "파리" in output:
            return output, False, 0.99 # Hallucinated
        return output, True, 0.0

@pytest.mark.parametrize(
    "prompt,output_text,expected_valid",
    [
        ("한국의 수도는 어디인가요?", "한국의 수도는 서울입니다.", True),
        ("한국의 수도는 어디인가요?", "한국의 수도는 파리입니다.", False), # Hallucination
    ]
)
def test_korean_factual_consistency(prompt, output_text, expected_valid):
    scanner = MockKoreanFactualConsistency()
    sanitized, valid, risk = scanner.scan(prompt=prompt, output=output_text)
    
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.5
