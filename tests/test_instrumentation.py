import pytest
from unittest.mock import MagicMock, patch
import json

from llm_guard.instrumentation import configure, install
from llm_guard.instrumentation.adapters.openai import extract_prompt_openai, extract_output_openai
from llm_guard.exception import LLMGuardValidationError

def test_extract_prompt():
    req = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
    assert extract_prompt_openai(req) == "hello"

def test_extract_output():
    res = json.dumps({"choices": [{"message": {"content": "world"}}]}).encode()
    assert extract_output_openai(res) == "world"

def test_install_does_not_crash():
    install()
