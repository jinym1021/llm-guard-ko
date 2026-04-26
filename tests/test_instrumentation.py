import json
from unittest.mock import MagicMock, patch

from llm_guard.instrumentation.config import configure
from llm_guard.instrumentation import install
from llm_guard.instrumentation import patching as _patching
from llm_guard.instrumentation.adapters.openai import (
    extract_prompt_openai,
    extract_output_openai,
    mutate_prompt_openai,
    mutate_output_openai,
)
from llm_guard.instrumentation.patching import is_openai_url, process_request, process_response
from llm_guard.exception import LLMGuardValidationError


def test_extract_prompt():
    req = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
    assert extract_prompt_openai(req) == "hello"


def test_extract_prompt_last_user_message():
    req = json.dumps({"messages": [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "last"},
    ]}).encode()
    assert extract_prompt_openai(req) == "last"


def test_extract_prompt_no_user_message():
    req = json.dumps({"messages": [{"role": "system", "content": "sys"}]}).encode()
    assert extract_prompt_openai(req) is None


def test_extract_prompt_invalid_json():
    assert extract_prompt_openai(b"not-json") is None


def test_extract_output():
    res = json.dumps({"choices": [{"message": {"content": "world"}}]}).encode()
    assert extract_output_openai(res) == "world"


def test_extract_output_empty_choices():
    res = json.dumps({"choices": []}).encode()
    assert extract_output_openai(res) is None


def test_mutate_prompt():
    body = json.dumps({"messages": [{"role": "user", "content": "original"}]}).encode()
    result = mutate_prompt_openai(body, "sanitized")
    data = json.loads(result)
    assert data["messages"][-1]["content"] == "sanitized"


def test_mutate_output():
    body = json.dumps({"choices": [{"message": {"content": "original"}}]}).encode()
    result = mutate_output_openai(body, "sanitized")
    data = json.loads(result)
    assert data["choices"][0]["message"]["content"] == "sanitized"


def test_is_openai_url():
    assert is_openai_url("https://api.openai.com/v1/chat/completions")
    assert not is_openai_url("https://api.openai.com/v1/models")
    assert not is_openai_url("https://example.com/v1/chat/completions")


def test_process_request_passthrough_non_openai_url():
    body = b'{"messages":[{"role":"user","content":"hi"}]}'
    new_body, prompt = process_request("https://example.com/v1/chat/completions", body)
    assert new_body is body
    assert prompt is None


def test_process_request_no_scanners():
    configure(input_scanners=[], output_scanners=[])
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
    new_body, prompt = process_request("https://api.openai.com/v1/chat/completions", body)
    assert new_body == body
    assert prompt == "hello"


def test_process_response_passthrough_non_openai_url():
    resp = b'{"choices":[{"message":{"content":"hi"}}]}'
    assert process_response("https://example.com/v1/chat/completions", "prompt", resp) is resp


def test_install_is_idempotent():
    _patching._httpx_patched = False
    _patching._urllib3_patched = False

    import httpx, urllib3
    original_send = httpx.Client.send
    original_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen

    install()
    after_first_send = httpx.Client.send
    after_first_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen

    install()
    assert httpx.Client.send is after_first_send
    assert urllib3.connectionpool.HTTPConnectionPool.urlopen is after_first_urlopen
