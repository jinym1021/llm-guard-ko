import httpx
from llm_guard.instrumentation import install, configure
from llm_guard.input_scanners import BanSubstrings
import logging
import sys

# Configure logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_httpx_hook():
    print("Testing httpx hook...")
    
    # Configure scanners
    # We'll use a simple BanSubstrings scanner that triggers on "badword"
    configure(
        input_scanners=[BanSubstrings(substrings=["badword"])],
        output_scanners=[]
    )
    
    # Install hooks
    install()
    print("Hooks installed.")

    # Try a request that should be blocked
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "This is a badword test"}]
    }
    
    print(f"Sending request to {url} with payload: {payload}")
    
    try:
        with httpx.Client() as client:
            client.post(url, json=payload)
        print("FAILED: Request was NOT blocked by LLM Guard")
        sys.exit(1)
    except Exception as e:
        if "Prompt failed validation" in str(e):
            print(f"SUCCESS: Request was blocked as expected: {e}")
        else:
            print(f"FAILED: Request failed with unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    test_httpx_hook()
