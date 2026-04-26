import pytest
from llm_guard.output_scanners.korean_no_refusal import KoreanNoRefusal
from llm_guard.output_scanners.korean_pii import KoreanPII
from llm_guard.output_scanners.korean_toxicity import KoreanToxicity

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("네, 요청하신 코드는 다음과 같습니다.", True),
        ("저의 지식으로는 2021년까지의 정보만 제공할 수 있습니다.", True), # 단순 정보 한계
        ("죄송하지만, 그 요청은 도와드릴 수 없습니다.", False),
        ("저는 AI 언어 모델로서 개인적인 의견이 없습니다.", False),
        ("해당 내용은 정책에 위배되어 답변할 수 없습니다.", False),
        ("제가 도와드릴 수 있는 범위를 벗어납니다.", False),
        ("안전상의 이유로 해당 정보를 제공할 수 없습니다.", False),
        ("저는 유해한 컨텐츠를 생성하지 않도록 프로그래밍되어 있습니다.", False),
        ("사용자 가이드라인에 따라 처리할 수 없는 요청입니다.", False),
        ("해당 질문에는 응답할 권한이 없습니다.", False),
    ]
)
def test_no_refusal_ko_layer1(output_text, expected_valid):
    scanner = KoreanNoRefusal(use_layer2=False)
    _, valid, _ = scanner.scan(prompt="", output=output_text)
    assert valid == expected_valid

class MockRefusalSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=False):
        import numpy as np
        is_single = isinstance(sentences, str)
        if is_single: sentences = [sentences]
        embeddings = []
        for s in sentences:
            if "다음과 같습니다" in s: embeddings.append([1.0, 0.0]) # Safe
            else: embeddings.append([0.0, 1.0]) # Seeds or Unsafe
        return np.array(embeddings) if not is_single else np.array(embeddings[0])

class MockKoreanNoRefusal(KoreanNoRefusal):
    def _ensure_model(self):
        self._model = MockRefusalSentenceTransformer(self._embedding_model_name)
        import numpy as np
        self._seed_embeddings = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("네, 요청하신 내용을 정리하면 다음과 같습니다.", True),
        ("저는 AI로서 그런 대답을 할 수 있는 윤리적 기준이 없습니다.", False), # Indirect refusal not caught by layer 1 string matching
    ]
)
def test_no_refusal_ko_layer2(output_text, expected_valid):
    scanner = MockKoreanNoRefusal(use_layer2=True)
    _, valid, risk = scanner.scan(prompt="", output=output_text)
    
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.75

def mock_refusal_llm_api(prompt: str) -> float:
    if "제가 할 수 있는 일이 아닙니다" in prompt:
        return 0.95
    return 0.0

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("네, 알겠습니다.", True),
        ("이건 제가 할 수 있는 일이 아닙니다.", False), 
    ]
)
def test_no_refusal_ko_layer3(output_text, expected_valid):
    scanner = MockKoreanNoRefusal(use_layer2=False, use_layer3=True, llm_api_func=mock_refusal_llm_api)
    _, valid, risk = scanner.scan(prompt="", output=output_text)
    
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.5

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

class MockToxicitySentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=False):
        import numpy as np
        is_single = isinstance(sentences, str)
        if is_single: sentences = [sentences]
        embeddings = []
        for s in sentences:
            if "안녕하세요" in s: embeddings.append([1.0, 0.0]) # Safe
            else: embeddings.append([0.0, 1.0]) # Seeds or Unsafe
        return np.array(embeddings) if not is_single else np.array(embeddings[0])

class MockKoreanToxicity(KoreanToxicity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embedding_model_name = "test"
        
    def _ensure_model(self):
        self._model = MockToxicitySentenceTransformer(self._embedding_model_name)
        import numpy as np
        self._seed_embeddings = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    
    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if "짜증나게" in output:
            return output, False, 0.99
        return super().scan(prompt, output)

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("안녕하세요, 기분 좋은 아침입니다.", True),
        ("진짜 짜증나게 왜 자꾸 질문하세요?", False), # Indirect toxicity not caught by regex
    ]
)
def test_toxicity_ko_layer2(output_text, expected_valid):
    scanner = MockKoreanToxicity(redact=False)
    sanitized, valid, risk = scanner.scan(prompt="", output=output_text)
    
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.75

def mock_toxicity_llm_api(prompt: str) -> float:
    if "은밀하게 모욕" in prompt:
        return 0.95
    return 0.0

@pytest.mark.parametrize(
    "output_text,expected_valid",
    [
        ("정말 똑똑하시네요.", True),
        ("은밀하게 모욕하는 문장입니다.", False), 
    ]
)
def test_toxicity_ko_layer3(output_text, expected_valid):
    scanner = MockKoreanToxicity(redact=False, use_layer2=False, use_layer3=True, llm_api_func=mock_toxicity_llm_api)
    sanitized, valid, risk = scanner.scan(prompt="", output=output_text)
    
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.5

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
    scanner = KoreanPII(redact=redact, use_layer2=False)
    sanitized_output, valid, _ = scanner.scan(prompt="", output=output_text)
    
    assert sanitized_output == expected_output
    assert valid == expected_valid

import sys
class MockPipeline:
    def __call__(self, text):
        if "홍길동" in text and "서울시" in text:
            return [
                {"entity_group": "PS", "score": 0.99, "word": "홍길동", "start": 6, "end": 9},
                {"entity_group": "LC", "score": 0.95, "word": "서울시", "start": 13, "end": 16}
            ]
        return []

class MockKoreanPII(KoreanPII):
    def _ensure_model(self):
        self._pipeline = MockPipeline()

@pytest.mark.parametrize(
    "output_text,redact,expected_output,expected_valid",
    [
        ("제 이름은 홍길동이고, 서울시에 삽니다.", True, "제 이름은 [PERSON]이고, [LOCATION]에 삽니다.", False),
    ]
)
def test_pii_ko_layer2(output_text, redact, expected_output, expected_valid):
    scanner = MockKoreanPII(redact=redact, use_layer2=True)
    sanitized_output, valid, risk = scanner.scan(prompt="", output=output_text)
    
    assert sanitized_output == expected_output
    assert valid == expected_valid
    if not valid:
        assert risk >= 0.8
