import json
from typing import Optional, Tuple

def extract_prompt_openai(request_body: bytes) -> Optional[str]:
    try:
        data = json.loads(request_body)
        messages = data.get("messages", [])
        if not messages:
            return None
        # Extract the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None
    except Exception:
        return None

def extract_output_openai(response_body: bytes) -> Optional[str]:
    try:
        data = json.loads(response_body)
        choices = data.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        return message.get("content")
    except Exception:
        return None

def mutate_prompt_openai(request_body: bytes, new_prompt: str) -> bytes:
    try:
        data = json.loads(request_body)
        messages = data.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                msg["content"] = new_prompt
                break
        return json.dumps(data).encode("utf-8")
    except Exception:
        return request_body

def mutate_output_openai(response_body: bytes, new_output: str) -> bytes:
    try:
        data = json.loads(response_body)
        choices = data.get("choices", [])
        if choices:
            if "message" in choices[0]:
                choices[0]["message"]["content"] = new_output
        return json.dumps(data).encode("utf-8")
    except Exception:
        return response_body
