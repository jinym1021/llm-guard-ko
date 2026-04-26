"""Shared pytest fixtures.

Strategy: try to import the real ``llm_guard`` package first. Only if that
fails (e.g. heavy deps like ``torch``/``presidio`` aren't installed) do we
fall back to lightweight stubs so the legacy ``test_evaluate.py`` and the
instrumentation tests can still run. Scanner-specific tests need the real
modules to load, so we must not pre-empt them.
"""

import sys
import types


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


def _install_stubs() -> None:
    """Stub the scanner packages with empty namespaces.

    Only used when real scanner imports are unavailable.
    """
    evaluate = types.ModuleType("llm_guard.evaluate")
    evaluate.scan_prompt = _scan_prompt
    evaluate.scan_output = _scan_output
    sys.modules["llm_guard.evaluate"] = evaluate

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

    class _Scanner:
        pass

    sys.modules["llm_guard.input_scanners"].Scanner = _Scanner
    sys.modules["llm_guard.output_scanners"].Scanner = _Scanner
    sys.modules["llm_guard.input_scanners.base"] = types.SimpleNamespace(Scanner=_Scanner)
    sys.modules["llm_guard.output_scanners.base"] = types.SimpleNamespace(Scanner=_Scanner)


def _try_real_imports() -> bool:
    """Return True iff both scanner packages import cleanly with real submodules."""
    try:
        import llm_guard.input_scanners  # noqa: F401
        import llm_guard.output_scanners  # noqa: F401
        return True
    except Exception:
        return False


if not _try_real_imports():
    _install_stubs()
