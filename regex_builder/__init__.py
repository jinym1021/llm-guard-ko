"""regex_builder — infer Python regex patterns from example strings.

Public API
----------
infer(examples)               Primary entry point. Accepts str or list[str].
infer_from_example(example)   Single-example variant.
infer_from_examples(examples) Multi-example variant (OR-combined).
"""

from __future__ import annotations

from .infer import infer, infer_from_example, infer_from_examples

__all__ = ["infer", "infer_from_example", "infer_from_examples"]
__version__ = "0.1.0"
