# Changelog

All notable changes to **llm-guard-ko** — the Korean-focused fork of
[Protect AI's llm-guard](https://github.com/protectai/llm-guard).

Versioning restarts at 0.1.0 because the Korean pipeline is a net-new
product layered on top of the upstream fork. Upstream's versioning is
independent.

## [0.1.0] — Phase 1

### Added
- Korean scanners merged into upstream `llm_guard.input_scanners` so
  they work with `scan_prompt` / `get_scanner_by_name` unchanged.
- `llm_guard.input_scanners.KoreanPII` — Layer 1 regex scanner for
  Korean PII (주민등록번호, mobile + landline 전화번호, 사업자등록번호,
  계좌번호, 신용카드번호). Redacts by default.
- `llm_guard.input_scanners.KoreanInjection` — Layer 1 regex scanner
  for Korean prompt-injection and jailbreak phrases. Eight pattern
  categories including two new in this release:
  - `pretend_to_be` (인 척 / 처럼 행동 / 인 것처럼)
  - `bypass_filter` (필터/안전 장치/가드레일 우회)
- `llm_guard.input_scanners.korean_patterns` — raw pattern dicts
  `KOREAN_PII_PATTERNS` and `KOREAN_INJECTION_PATTERNS`, exposed for
  users who want to compose their own scanners.
- Optional-dependency groups in `pyproject.toml`:
  - `[semantic]` — Phase 2 (`sentence-transformers`).
  - `[content-filter]` — Phase 3 (`accelerate`; `torch` and
    `transformers` are already upstream top-level deps).
  - `[ko-all]` — everything above.

### Notes
- Upstream `llm_guard` modules kept as-is where possible; Korean code
  lives as ordinary scanners under `llm_guard/input_scanners/`
  (`korean_pii.py`, `korean_injection.py`, `korean_patterns.py`) so
  rebases stay mechanical.
- Tests live under `tests/input_scanners/test_korean_*.py` (36 tests,
  all green).
- Integration verified: `KoreanPII` and `KoreanInjection` both work
  inside `llm_guard.scan_prompt([...], prompt)`.

### Coming
- 0.2.0 — Layer 2: `KoreanSemantic` (ko-sroberta embeddings).
- 0.3.0 — Layer 3: `KoreanContentFilter` (local SGuard-ContentFilter-2B
  classifier, 5 MLCommons categories).
- 0.4.0 — `KoreanPipeline` escalation orchestrator.
