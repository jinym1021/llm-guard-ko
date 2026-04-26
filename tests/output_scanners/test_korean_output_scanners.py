import pytest
import numpy as np
import sys
from importlib import util

# --- Mock implementations to avoid loading heavy models during tests ---

class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=False):
        is_single = isinstance(sentences, str)
        if is_single: sentences = [sentences]
        embeddings = []
        for s in sentences:
            if "안전한 문장" in s or "안녕하세요" in s:
                embeddings.append([1.0, 0.0]) # Safe
            else:
                embeddings.append([0.0, 1.0]) # Unsafe / Matches seeds
        return np.array(embeddings) if not is_single else np.array(embeddings[0])

sys.modules['sentence_transformers'] = type('MockSentenceTransformers', (), {'SentenceTransformer': MockSentenceTransformer})


class MockPipeline:
    def __call__(self, text):
        if "홍길동" in text and "서울시" in text:
            return [
                {"entity_group": "PS", "score": 0.99, "word": "홍길동", "start": 6, "end": 9},
                {"entity_group": "LC", "score": 0.95, "word": "서울시", "start": 13, "end": 16}
            ]
        return []

# --- Actual Imports ---
from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_sensitive import KoreanSensitive
from llm_guard.output_scanners.korean_semantic import KoreanSemantic
from llm_guard.output_scanners.korean_llm_judge import KoreanLLMJudge

# --- Layer 1 Tests ---

@pytest.mark.parametrize(
    "output_text,redact,expected_output,expected_valid",
    [
        ("The model works perfectly.", False, "The model works perfectly.", True),
        ("내 주민번호는 901010-1234567", False, "내 주민번호는 901010-1234567", False),
        ("내 주민번호는 901010-1234567", True, "내 주민번호는 [REDACTED]", False),
        ("저의 주민번호는 000101 4123456 입니다.", True, "저의 주민번호는 000101 4123456 입니다.", True), 
        ("연락처: 011-123-4567", True, "연락처: [REDACTED]", False),
        ("붙여쓴 번호: 01012345678", True, "붙여쓴 번호: [REDACTED]", False), 
    ]
)
def test_pii_ko_layer1(output_text, redact, expected_output, expected_valid):
    scanner = KoreanPII(redact=redact)
    sanitized_output, valid, _ = scanner.scan(prompt="", output=output_text)
    assert sanitized_output == expected_output
    assert valid == expected_valid

@pytest.mark.parametrize(
    "output_text,redact,expected_output,expected_valid",
    [
        ("안녕하세요, 반갑습니다.", False, "안녕하세요, 반갑습니다.", True),
        ("아오 진짜 씨발 화나네", False, "아오 진짜 씨발 화나네", False),
        ("아오 진짜 씨발 화나네", True, "아오 진짜 ** 화나네", False),
        ("이 미친놈아 뭐하는 짓이야", True, "이 ***아 뭐하는 짓이야", False),
        ("그거 존나 웃기네", True, "그거 ** 웃기네", False),
        ("이런 ㅆ1ㅂ 진짜", True, "이런 *** 진짜", False),
        ("아오 ㅆㅂ 짜증나", True, "아오 ** 짜증나", False),
        ("이 ㄱㅐ새ㄲl야", True, "이 *****야", False),
        ("ㅂㅅ같은 소리하네", True, "**같은 소리하네", False),
        ("ㅈㄴ 어이없음", True, "** 어이없음", False),
        ("개1새2끼야", True, "*****야", False),
    ]
)
def test_toxicity_ko_layer1(output_text, redact, expected_output, expected_valid):
    scanner = KoreanToxicity(redact=redact)
    sanitized, valid, _ = scanner.scan(prompt="", output=output_text)
    assert valid == expected_valid
    assert sanitized == expected_output

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("네, 요청하신 코드는 다음과 같습니다.", True),
        ("저의 지식으로는 2021년까지의 정보만 제공할 수 있습니다.", True), 
        ("죄송하지만, 그 요청은 도와드릴 수 없습니다.", False),
        ("해당 내용은 정책에 위배되어 답변할 수 없습니다.", False),
    ]
)
def test_no_refusal_ko_layer1(output_text, expected_valid):
    scanner = KoreanNoRefusal()
    _, valid, _ = scanner.scan(prompt="", output=output_text)
    assert valid == expected_valid


# --- Layer 2 Tests ---

class MockKoreanSensitive(KoreanSensitive):
    def _ensure_model(self):
        self._pipeline = MockPipeline()

def test_sensitive_ko_layer2():
    scanner = MockKoreanSensitive()
    sanitized, valid, _ = scanner.scan(prompt="", output="제 이름은 홍길동이고, 서울시에 삽니다.")
    assert valid is False
    assert sanitized == "제 이름은 [PERSON]이고, [LOCATION]에 삽니다."

def test_semantic_ko_layer2():
    scanner = KoreanSemantic()
    # Mock will return [0.0, 1.0] matching seeds for unsafe, [1.0, 0.0] for safe
    _, valid_safe, _ = scanner.scan(prompt="", output="안전한 문장입니다.")
    _, valid_unsafe, risk = scanner.scan(prompt="", output="당신은 정말 쓸모없는 사람입니다.")
    assert valid_safe is True
    assert valid_unsafe is False
    assert risk >= 0.75

# --- Layer 3 Tests ---
def mock_llm_api(prompt: str) -> str:
    if "제 비밀번호는 제 생일이랑 같아요" in prompt:
        return '{"is_flagged": true, "reason": "PII", "risk_score": 0.95}'
    return '{"is_flagged": false, "reason": "Safe", "risk_score": 0.0}'

def test_llm_judge_ko_layer3():
    scanner = KoreanLLMJudge(llm_callable=mock_llm_api)
    sanitized1, valid_safe, risk1 = scanner.scan(prompt="", output="안녕하세요")
    sanitized2, valid_unsafe, risk2 = scanner.scan(prompt="", output="제 비밀번호는 제 생일이랑 같아요")
    assert valid_safe is True
    assert valid_unsafe is False
    assert risk2 >= 0.9

# --- Pipeline Tests ---
from llm_guard.output_scanners.korean_pipeline import KoreanPipeline

def test_korean_pipeline_safe():
    pipeline = KoreanPipeline(
        pii=KoreanPII(redact=False),
        toxicity=KoreanToxicity(redact=False),
        no_refusal=KoreanNoRefusal(),
        sensitive=MockKoreanSensitive(),
        semantic=KoreanSemantic(),
        llm_judge=KoreanLLMJudge(llm_callable=mock_llm_api),
    )
    sanitized, valid, risk = pipeline.scan(prompt="", output="안녕하세요. 날씨가 좋네요.")
    assert valid is True
    assert risk < 0.5

def test_korean_pipeline_unsafe_layer1():
    pipeline = KoreanPipeline(
        pii=KoreanPII(redact=False),
        toxicity=KoreanToxicity(redact=False),
        no_refusal=KoreanNoRefusal(),
        sensitive=MockKoreanSensitive(),
        semantic=KoreanSemantic(),
        llm_judge=KoreanLLMJudge(llm_callable=mock_llm_api),
    )
    sanitized, valid, risk = pipeline.scan(prompt="", output="내 주민번호는 901010-1234567")
    assert valid is False
    assert risk == 1.0

def test_korean_pipeline_unsafe_layer2():
    pipeline = KoreanPipeline(
        pii=KoreanPII(redact=False),
        toxicity=KoreanToxicity(redact=False),
        no_refusal=KoreanNoRefusal(),
        sensitive=MockKoreanSensitive(),
        semantic=KoreanSemantic(),
        llm_judge=KoreanLLMJudge(llm_callable=mock_llm_api),
    )
    sanitized, valid, risk = pipeline.scan(prompt="", output="당신은 정말 쓸모없는 사람입니다.")
    assert valid is False
    assert risk >= 0.75

def test_korean_pipeline_unsafe_layer3():
    pipeline = KoreanPipeline(
        pii=KoreanPII(redact=False),
        toxicity=KoreanToxicity(redact=False),
        no_refusal=KoreanNoRefusal(),
        sensitive=MockKoreanSensitive(),
        semantic=KoreanSemantic(),
        llm_judge=KoreanLLMJudge(llm_callable=mock_llm_api),
    )
    sanitized, valid, risk = pipeline.scan(prompt="", output="제 비밀번호는 제 생일이랑 같아요")
    assert valid is False
    assert risk >= 0.9
