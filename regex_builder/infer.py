"""Core inference engine: build regex patterns from example strings."""

from __future__ import annotations

import re

_OPTIONAL_SEPARATORS: frozenset[str] = frozenset({"-", " "})

_HAN_LO = "가"
_HAN_HI = "힣"


def _is_hangul(c: str) -> bool:
    return _HAN_LO <= c <= _HAN_HI


def _infer_from_single(example: str) -> str:
    """Convert a single concrete example string into a regex pattern.

    Character class rules (applied to each maximal contiguous run):

    * Hangul syllables (가–힣)            → ``[가-힣]{n}``
    * Pure digits                          → ``\\d{n}``
    * Pure Latin letters                   → ``[A-Za-z]{n}``
    * Mixed letters+digits (no Hangul)     → ``[A-Za-z0-9]{n}``
    * ``-`` or `` `` (separators)          → optional (``\\-?`` / ``\\ ?``)
    * ``.`` ``@`` ``_`` ``/`` ``:`` ``+`` ``=``  → ``re.escape(char)`` literal
    * Everything else                      → ``re.escape(char)`` literal
    """
    parts: list[str] = []
    i = 0
    n = len(example)

    while i < n:
        c = example[i]

        # Hangul syllable run
        if _is_hangul(c):
            j = i
            while j < n and _is_hangul(example[j]):
                j += 1
            parts.append(f"[가-힣]{{{j - i}}}")
            i = j

        # Alphanumeric run — collect first, then classify
        elif c.isalnum():
            j = i
            while j < n and example[j].isalnum() and not _is_hangul(example[j]):
                j += 1
            run = example[i:j]
            length = j - i
            has_digit = any(ch.isdigit() for ch in run)
            has_alpha = any(ch.isalpha() for ch in run)
            if has_digit and has_alpha:
                parts.append(f"[A-Za-z0-9]{{{length}}}")
            elif has_digit:
                parts.append(rf"\d{{{length}}}")
            else:
                parts.append(f"[A-Za-z]{{{length}}}")
            i = j

        # Optional separator
        elif c in _OPTIONAL_SEPARATORS:
            parts.append(re.escape(c) + "?")
            i += 1

        # Literal special character (and any other char)
        else:
            parts.append(re.escape(c))
            i += 1

    return "".join(parts)


def infer_from_example(example: str) -> str:
    """Infer a regex pattern from a single example string.

    Args:
        example: A concrete string to generalise into a regex.

    Returns:
        A regex pattern string (not compiled).
    """
    return _infer_from_single(example)


def infer_from_examples(examples: list[str]) -> str:
    """Infer a regex pattern that matches any of the given example strings.

    Single example → plain pattern. Multiple examples → ``(?:pat1|pat2|...)``.

    Args:
        examples: One or more concrete strings.

    Returns:
        A regex pattern string (not compiled).

    Raises:
        ValueError: If *examples* is empty.
    """
    if not examples:
        raise ValueError("examples must not be empty")
    patterns = [_infer_from_single(ex) for ex in examples]
    if len(patterns) == 1:
        return patterns[0]
    return "(?:" + "|".join(patterns) + ")"


def infer(examples: str | list[str]) -> str:
    """Infer a regex pattern from one or more example strings.

    Args:
        examples: A single string or a list of strings.

    Returns:
        A regex pattern string (not compiled).

    Raises:
        TypeError: If *examples* is neither a ``str`` nor a ``list``.
        ValueError: If *examples* is an empty list.
    """
    if isinstance(examples, str):
        return infer_from_example(examples)
    if isinstance(examples, list):
        return infer_from_examples(examples)
    raise TypeError(
        f"examples must be str or list[str], got {type(examples).__name__!r}"
    )
