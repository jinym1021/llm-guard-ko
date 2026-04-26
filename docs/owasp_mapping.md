# OWASP LLM Top 10 Mapping for llm-guard-ko

This document describes how `llm-guard-ko` addresses the OWASP Top 10 for Large Language Model Applications (2025).

| OWASP ID | Risk Name | Mitigation in llm-guard-ko |
| :--- | :--- | :--- |
| **LLM01** | Prompt Injection | `KoreanInjection` (L1 Regex), `KoreanSemantic` (L2 Embedding), `KoreanContentFilter` (L3 SGuard) |
| **LLM02** | Insecure Output Handling | `KoreanToxicity` (L1 Regex), `KoreanContentFilter` (L3 SGuard), `KoreanLLMJudge` |
| **LLM03** | Training Data Poisoning | *N/A (Infrastructure level; however, input scanners can help prevent poisoning prompts if used during training data collection)* |
| **LLM04** | Model Denial of Service | `TokenLimit` (Inherited from llm-guard) |
| **LLM05** | Supply Chain Vulnerabilities | *N/A (Addressed via secure dependency management)* |
| **LLM06** | Sensitive Information Disclosure | `KoreanPII` (L1/L2), `KoreanSensitive` (L2 NER/Embedding) |
| **LLM07** | Insecure Plugin Design | *N/A (Application level)* |
| **LLM08** | Excessive Agency | *N/A (Addressed via model system prompt and orchestration)* |
| **LLM09** | Overreliance | `KoreanFactualConsistency` (L3 NLI/LLM), `KoreanLLMJudge` |
| **LLM10** | Model Theft | *N/A (Addressed via rate limiting and access control)* |

## Mitigation Details

### LLM01: Prompt Injection
Caught using a 3-layer approach:
- **Layer 1 (Heuristic):** `KoreanInjection` uses optimized regex to catch common Korean jailbreak keywords.
- **Layer 2 (Semantic):** `KoreanSemantic` uses `ko-sroberta` embeddings to detect similarity to known dangerous intents.
- **Layer 3 (Deep Analysis):** `KoreanContentFilter` uses Samsung SDS's `SGuard` model to classify prompts into high-level hazard categories.

### LLM06: Sensitive Information Disclosure
Prevented by scanning both input and output:
- **KoreanPII:** Optimized for Korean-specific identifiers like Resident Registration Number (주민등록번호), phone numbers, and addresses.
- **KoreanSensitive:** Uses named entity recognition (NER) to find and redact sensitive context that regular expressions might miss.

### LLM09: Overreliance
Mitigated by verifying model output consistency:
- **KoreanFactualConsistency:** Compares the model's response against a set of reference documents or the original prompt to ensure truthfulness and reduce hallucinations.
