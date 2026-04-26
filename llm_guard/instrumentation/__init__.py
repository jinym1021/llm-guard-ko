from .config import configure
from .patching import patch_httpx, patch_urllib3

def install():
    """
    Install auto-instrumentation hooks for popular HTTP clients to intercept LLM API calls.
    """
    patch_httpx()
    patch_urllib3()

__all__ = ["configure", "install"]
