import logging
from typing import Optional
from urllib.parse import urlparse

from llm_guard.evaluate import scan_prompt, scan_output
from llm_guard.exception import LLMGuardValidationError
from llm_guard.instrumentation.config import get_input_scanners, get_output_scanners
from llm_guard.instrumentation.adapters.openai import (
    extract_prompt_openai,
    extract_output_openai,
    mutate_prompt_openai,
    mutate_output_openai,
)

logger = logging.getLogger(__name__)

_httpx_patched = False
_urllib3_patched = False


def is_openai_url(url: str) -> bool:
    parsed = urlparse(url)
    return "openai.com" in parsed.netloc and "/v1/chat/completions" in parsed.path


def process_request(url: str, content: bytes) -> tuple[bytes, Optional[str]]:
    """Scan and optionally sanitize the outgoing prompt. Returns (body, prompt)."""
    if not is_openai_url(url):
        return content, None
    prompt = extract_prompt_openai(content)
    if not prompt:
        return content, None
    scanners = get_input_scanners()
    sanitized_prompt, results_valid, results_score = scan_prompt(scanners, prompt)
    if not all(results_valid.values()):
        raise LLMGuardValidationError(f"Prompt failed validation: {results_score}")
    if sanitized_prompt != prompt:
        return mutate_prompt_openai(content, sanitized_prompt), sanitized_prompt
    return content, prompt


def process_response(url: str, prompt: str, response_content: bytes) -> bytes:
    """Scan and optionally sanitize the incoming output."""
    if not is_openai_url(url):
        return response_content
    output = extract_output_openai(response_content)
    if not output:
        return response_content
    scanners = get_output_scanners()
    sanitized_output, results_valid, results_score = scan_output(scanners, prompt, output)
    if not all(results_valid.values()):
        raise LLMGuardValidationError(f"Output failed validation: {results_score}")
    if sanitized_output != output:
        return mutate_output_openai(response_content, sanitized_output)
    return response_content


def patch_httpx():
    global _httpx_patched
    if _httpx_patched:
        return
    try:
        import httpx
    except ImportError:
        return

    original_send = httpx.Client.send
    original_async_send = httpx.AsyncClient.send

    def send_wrapper(self, request: httpx.Request, **kwargs) -> httpx.Response:
        url = str(request.url)
        content = request.content

        new_content, prompt = process_request(url, content)
        if new_content != content:
            request._content = new_content
            request.headers["Content-Length"] = str(len(new_content))

        response = original_send(self, request, **kwargs)

        if prompt:
            new_resp_content = process_response(url, prompt, response.content)
            if new_resp_content != response.content:
                response._content = new_resp_content
                if "Content-Length" in response.headers:
                    response.headers["Content-Length"] = str(len(new_resp_content))

        return response

    async def async_send_wrapper(self, request: httpx.Request, **kwargs) -> httpx.Response:
        url = str(request.url)
        await request.aread()
        content = request.content

        new_content, prompt = process_request(url, content)
        if new_content != content:
            request._content = new_content
            request.headers["Content-Length"] = str(len(new_content))

        response = await original_async_send(self, request, **kwargs)

        if prompt:
            await response.aread()
            new_resp_content = process_response(url, prompt, response.content)
            if new_resp_content != response.content:
                response._content = new_resp_content
                if "Content-Length" in response.headers:
                    response.headers["Content-Length"] = str(len(new_resp_content))

        return response

    httpx.Client.send = send_wrapper
    httpx.AsyncClient.send = async_send_wrapper
    _httpx_patched = True


def patch_urllib3():
    global _urllib3_patched
    if _urllib3_patched:
        return
    try:
        import urllib3
    except ImportError:
        return

    original_urlopen = urllib3.connectionpool.HTTPConnectionPool.urlopen

    def urlopen_wrapper(self, method, url, body=None, headers=None, **kwargs):
        port = f":{self.port}" if self.port is not None else ""
        full_url = f"{self.scheme}://{self.host}{port}{url}"

        request_body = body.encode("utf-8") if isinstance(body, str) else body

        prompt: Optional[str] = None
        if request_body:
            new_content, prompt = process_request(full_url, request_body)
            if new_content != request_body:
                body = new_content
                if headers and "Content-Length" in headers:
                    headers["Content-Length"] = str(len(new_content))

        response = original_urlopen(self, method, url, body=body, headers=headers, **kwargs)

        if prompt:
            response_body = response.data
            if response_body:
                new_resp_content = process_response(full_url, prompt, response_body)
                if new_resp_content != response_body:
                    response._body = new_resp_content

        return response

    urllib3.connectionpool.HTTPConnectionPool.urlopen = urlopen_wrapper
    _urllib3_patched = True
