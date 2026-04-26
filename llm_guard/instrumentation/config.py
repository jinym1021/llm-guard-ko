from typing import List, Optional

from llm_guard.input_scanners.base import Scanner as InputScanner
from llm_guard.output_scanners.base import Scanner as OutputScanner

_input_scanners: List[InputScanner] = []
_output_scanners: List[OutputScanner] = []

def configure(input_scanners: Optional[List[InputScanner]] = None, output_scanners: Optional[List[OutputScanner]] = None):
    """
    Configure global scanners to be used by the auto-instrumentation hooks.
    """
    global _input_scanners, _output_scanners
    if input_scanners is not None:
        _input_scanners = input_scanners
    if output_scanners is not None:
        _output_scanners = output_scanners

def get_input_scanners() -> List[InputScanner]:
    return _input_scanners

def get_output_scanners() -> List[OutputScanner]:
    return _output_scanners
