from typing import Dict, Optional

from .ban_code import BanCode
from .ban_competitors import BanCompetitors
from .ban_substrings import BanSubstrings
from .ban_topics import BanTopics
from .base import Scanner
from .bias import Bias
from .code import Code
from .deanonymize import Deanonymize
from .factual_consistency import FactualConsistency
from .gibberish import Gibberish
from .json import JSON
from .korean_content_filter import KoreanContentFilter
from .korean_factual_consistency import KoreanFactualConsistency
from .korean_llm_judge import KoreanLLMJudge
from .korean_no_refusal import KoreanNoRefusal
from .korean_pii import KoreanPII
from .korean_pipeline import KoreanPipeline
from .korean_semantic import KoreanSemantic
from .korean_sensitive import KoreanSensitive
from .korean_toxicity import KoreanToxicity
from .language import Language
from .language_same import LanguageSame
from .malicious_urls import MaliciousURLs
from .no_refusal import NoRefusal, NoRefusalLight
from .reading_time import ReadingTime
from .regex import Regex
from .relevance import Relevance
from .sensitive import Sensitive
from .sentiment import Sentiment
from .toxicity import Toxicity
from .url_reachabitlity import URLReachability


_SCANNER_REGISTRY = {
    "BanCode": BanCode,
    "BanCompetitors": BanCompetitors,
    "BanSubstrings": BanSubstrings,
    "BanTopics": BanTopics,
    "Bias": Bias,
    "Code": Code,
    "Deanonymize": Deanonymize,
    "FactualConsistency": FactualConsistency,
    "Gibberish": Gibberish,
    "JSON": JSON,
    "KoreanContentFilter": KoreanContentFilter,
    "KoreanFactualConsistency": KoreanFactualConsistency,
    "KoreanLLMJudge": KoreanLLMJudge,
    "KoreanNoRefusal": KoreanNoRefusal,
    "KoreanPII": KoreanPII,
    "KoreanPipeline": KoreanPipeline,
    "KoreanSemantic": KoreanSemantic,
    "KoreanSensitive": KoreanSensitive,
    "KoreanToxicity": KoreanToxicity,
    "Language": Language,
    "LanguageSame": LanguageSame,
    "MaliciousURLs": MaliciousURLs,
    "NoRefusal": NoRefusal,
    "NoRefusalLight": NoRefusalLight,
    "ReadingTime": ReadingTime,
    "Regex": Regex,
    "Relevance": Relevance,
    "Sensitive": Sensitive,
    "Sentiment": Sentiment,
    "Toxicity": Toxicity,
    "URLReachability": URLReachability,
}


def get_scanner_by_name(scanner_name: str, scanner_config: Optional[Dict] = None) -> Scanner:
    """Instantiate an output scanner by name.

    Parameters:
        scanner_name: Name of the scanner class (e.g. ``"KoreanPipeline"``).
        scanner_config: Keyword arguments forwarded to the scanner constructor.

    Raises:
        ValueError: If *scanner_name* is not a registered scanner.
    """
    config = scanner_config or {}
    try:
        cls = _SCANNER_REGISTRY[scanner_name]
    except KeyError as exc:
        raise ValueError(f"Unknown scanner name: {scanner_name}!") from exc
    return cls(**config)
