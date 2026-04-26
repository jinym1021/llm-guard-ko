"""
Stub missing required dependencies for environments where heavy packages
(torch, presidio, spacy, etc.) are not installed.

Strategy: pre-populate sys.modules with lightweight stubs for the scanner
sub-packages so llm_guard/__init__.py can load without the heavy deps.
The instrumentation module imports llm_guard.evaluate, which we stub here.
"""
import sys
import types
from unittest.mock import MagicMock


def _scan_prompt(scanners, prompt, fail_fast=False):
    sanitized = prompt
    valid, score = {}, {}
    for s in scanners:
        sanitized, ok, risk = s.scan(sanitized)
        valid[type(s).__name__] = ok
        score[type(s).__name__] = risk
        if fail_fast and not ok:
            break
    return sanitized, valid, score


def _scan_output(scanners, prompt, output, fail_fast=False):
    sanitized = output if output else ""
    valid, score = {}, {}
    for s in scanners:
        sanitized, ok, risk = s.scan(prompt, sanitized)
        valid[type(s).__name__] = ok
        score[type(s).__name__] = risk
        if fail_fast and not ok:
            break
    return sanitized, valid, score


def _install_stubs():
    # Build a real-looking llm_guard.evaluate module without importing scanners
    evaluate = types.ModuleType("llm_guard.evaluate")
    evaluate.scan_prompt = _scan_prompt
    evaluate.scan_output = _scan_output
    sys.modules["llm_guard.evaluate"] = evaluate

    # Stub the scanner packages so their __init__.py never runs
    for pkg in (
        "llm_guard.input_scanners",
        "llm_guard.output_scanners",
        "llm_guard.input_scanners.base",
        "llm_guard.output_scanners.base",
    ):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []
            sys.modules[pkg] = m

    # Provide Scanner base classes (protocol-style) so imports succeed
    class _Scanner:
        pass

    sys.modules["llm_guard.input_scanners"].Scanner = _Scanner
    sys.modules["llm_guard.output_scanners"].Scanner = _Scanner
    sys.modules["llm_guard.input_scanners.base"] = types.SimpleNamespace(Scanner=_Scanner)
    sys.modules["llm_guard.output_scanners.base"] = types.SimpleNamespace(Scanner=_Scanner)


_install_stubs()
