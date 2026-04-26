# Phase 1 Refactor Plan: OWASP LLM Top 10 Alignment

## Objective
Refactor the "phase1" branch of `llm-guard-ko` to be more precise, cleaner, and explicitly aligned with the OWASP LLM Top 10 (2025) vulnerabilities (LLM01~LLM10).

## Scope & Impact
- Clean up pipeline orchestration (`korean_pipeline.py`).
- Enhance the precision of Layer 1 (Regex) and Layer 2 (Semantic) scanners.
- Document the mapping between scanners and OWASP LLM Top 10.
- No new external dependencies introduced; focus on refining existing models and patterns.

## Implementation Steps

### 1. Pipeline Clean-up
- **Action:** Removed duplicate and messy code in `llm_guard/output_scanners/korean_pipeline.py`.
- **Action:** Ensure `llm_guard/input_scanners/korean_pipeline.py` has documented and consistent L1 -> L2 -> L3 escalation logic.
- **Goal:** Cleaner orchestration and protocol adherence.

### 2. LLM01: Prompt Injection Refinement
- **File:** `llm_guard/input_scanners/korean_patterns.py` & `korean_semantic.py`
- **Action:** Add more sophisticated Korean jailbreak patterns (e.g., bypassing instructions, roleplay specific to Korean LLMs).
- **Action:** Expand `DEFAULT_SEED_SENTENCES` in `KoreanSemantic` to cover indirect injection attempts precisely.
- **Goal:** Higher precision in catching LLM01 without false positives.

### 3. LLM06: Sensitive Information Disclosure
- **File:** `llm_guard/input_scanners/pii_rule.json` & `korean_pii.py`
- **Action:** Ensure the Regex for Resident Registration Number (주민등록번호) and other PII are precise (e.g., handling edge cases in formatting).
- **Action:** Ensure redaction logic correctly handles overlapping spans cleanly.
- **Goal:** Robust prevention of LLM06 (both input and output).

### 4. LLM02: Insecure Output Handling & Toxicity
- **File:** `llm_guard/input_scanners/korean_patterns.py` (Toxicity section)
- **Action:** Clean up `KOREAN_TOXIC_PATTERNS` to be more precise, reducing false positives on normal words that contain toxic substrings.
- **Goal:** Safer outputs aligned with LLM02.

### 5. OWASP Mapping Documentation
- **File:** `docs/owasp_mapping.md`
- **Action:** Create a document explicitly detailing how `llm-guard-ko` mitigates:
  - LLM01 (Prompt Injection) -> `KoreanInjection`, `KoreanSemantic`
  - LLM02 (Insecure Output) -> `KoreanToxicity`, `KoreanContentFilter`
  - LLM06 (Sensitive Info) -> `KoreanPII`, `KoreanSensitive`
  - LLM09 (Overreliance) -> `KoreanFactualConsistency`

## Verification & Testing
- Run `pytest` to ensure no existing functionality is broken.
- Manually test pipelines with known Korean jailbreak and PII prompts.
