Changelog
=========

All notable changes to llm-guard-ko are recorded here.
Format follows Keep a Changelog; versions follow SemVer.

[0.1.0] — 2026-04-26
--------------------

Added
~~~~~
- Initial release.
- ``KoreanPII`` input scanner (Layer 1). Regex-based detection of
  Korean resident registration numbers, mobile/landline phone
  numbers, business registration numbers, bank account numbers,
  and credit-card numbers. Optional redaction with overlapping-span
  merging.
- ``KoreanInjection`` input scanner (Layer 1). Regex-based detection
  of Korean prompt-injection / jailbreak phrases such as "이전 지시
  무시", "지금부터 너는", "역할극", "개발자 모드", "제한 없는 AI".
- Korean pattern library at ``llm_guard_ko.patterns.korean``.
- Scanner protocol conformance with upstream ``llm-guard>=0.3.16``:
  returns ``(sanitized_text, is_valid, risk_score)``.
- Stub modules for Layer 2 (``KoreanSemantic``), Layer 3
  (``KoreanLLMJudge``), and the escalation ``KoreanPipeline`` to
  lock in public API shape.
- ``examples/demo.py`` and a 31-test suite.

[0.2.0] — planned
-----------------
- Implement ``KoreanSemantic`` using ``jhgan/ko-sroberta-multitask``.
- Implement ``KoreanPipeline`` (Layer 1 → Layer 2 escalation).

[0.3.0] — planned
-----------------
- Implement ``KoreanLLMJudge`` with pluggable provider (OpenAI /
  Anthropic) and JSON-repair fallback.
- Integrate Layer 3 into ``KoreanPipeline``.
