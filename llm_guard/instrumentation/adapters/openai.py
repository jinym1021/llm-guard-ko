import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_prompt_openai(request_body: bytes) -> Optional[str]:
    try:
        data = json.loads(request_body)
        for msg in reversed(data.get("messages", [])):
            if msg.get("role") == "user":
                return msg.get("content")
        return None
    except Exception:
        logger.debug("Failed to extract prompt from request body", exc_info=True)
        return None


def extract_output_openai(response_body: bytes) -> Optional[str]:
    try:
        data = json.loads(response_body)
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")
    except Exception:
        logger.debug("Failed to extract output from response body", exc_info=True)
        return None


def mutate_prompt_openai(request_body: bytes, new_prompt: str) -> bytes:
    try:
        data = json.loads(request_body)
        for msg in reversed(data.get("messages", [])):
            if msg.get("role") == "user":
                msg["content"] = new_prompt
                break
        return json.dumps(data).encode("utf-8")
    except Exception:
        logger.warning("Failed to mutate prompt; returning original body", exc_info=True)
        return request_body


def mutate_output_openai(response_body: bytes, new_output: str) -> bytes:
    try:
        data = json.loads(response_body)
        choices = data.get("choices", [])
        if choices and "message" in choices[0]:
            choices[0]["message"]["content"] = new_output
        return json.dumps(data).encode("utf-8")
    except Exception:
        logger.warning("Failed to mutate output; returning original body", exc_info=True)
        return response_body
