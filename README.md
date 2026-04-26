llm-guard-ko
============

Korean-language extension pack for
[llm-guard](https://github.com/protectai/llm-guard).

Status: 0.1.0 — alpha. Layer 1 (regex PII + prompt-injection phrases)
is implemented and tested. Layers 2 (semantic) and 3 (LLM judge) are
stubbed with a clear interface; enable them via the `[semantic]` and
`[llm-judge]` extras once their models/keys are available.

Why
---

Upstream llm-guard's PII, toxicity, and prompt-injection scanners are
trained on English. Korean inputs slip past them because:

- Korean resident registration numbers (주민등록번호) don't look like
  US SSNs.
- Korean phone numbers use the `01X-XXXX-XXXX` shape.
- Prompt-injection triggers like "지금부터 넌" or "이전 지시 무시"
  never appear in English training data.

llm-guard-ko adds a thin layer of Korean-aware scanners that plug into
llm-guard's existing `Scanner` protocol, so you can mix them into any
existing llm-guard pipeline.

Install
-------

    pip install llm-guard-ko                # layer 1 only
    pip install "llm-guard-ko[semantic]"    # + layer 2
    pip install "llm-guard-ko[all]"         # + layer 3 (LLM judge)

Quickstart
----------

Standalone:

    from llm_guard_ko import KoreanPII

    scanner = KoreanPII()
    sanitized, is_valid, risk = scanner.scan("내 주민번호는 901010-1234567")
    # sanitized -> "내 주민번호는 [REDACTED]"
    # is_valid  -> False
    # risk      -> 1.0

Mixed with upstream llm-guard:

    from llm_guard import scan_prompt
    from llm_guard.input_scanners import Toxicity
    from llm_guard_ko import KoreanPII, KoreanInjection

    scanners = [KoreanPII(), KoreanInjection(), Toxicity()]
    sanitized, results, risk = scan_prompt(scanners, user_input)

Scanner contract
----------------

Every scanner implements llm-guard's `Scanner` protocol:

    scan(prompt: str) -> tuple[sanitized_text: str,
                               is_valid: bool,
                               risk_score: float]

`is_valid=True, risk=0.0` means the prompt is safe. Matches the
exact convention used by upstream llm-guard so scanners are
interchangeable.

License
-------

MIT — matches upstream llm-guard.
