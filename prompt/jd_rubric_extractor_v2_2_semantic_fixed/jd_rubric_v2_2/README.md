# JD Rubric Extractor v2.2 — semantic fixes (schema unchanged)

**Purpose.** Replace the v2.1 JD-extraction prompt with the complete v2.2 FULL system prompt, or use the schema-mode variant with the unchanged JSON Schema. Extract and validate each **unique JD once**, then cache its reusable rubric for candidate matching. This is not a resume-scoring prompt.

## Changes from v2.1

| Issue in `results_full_text.csv` | Prompt revision | Optional validator behavior |
|---|---|---|
| JD-000016 says **currently studying for a degree**, but `education_requirement.min_level=bachelor` implied a completed degree | Store the current-enrollment condition as `requirements[].category="other"`; leave completed-degree min_level null unless JD separately requires it | Blocks missing enrollment condition or mislabelled completed-degree minimum |
| JD-000314 is a short, informal request for accounting/tax help, but returned `not_a_jd` | Mark an intelligible, role-specific hiring need `partial`; preserve only stated duties, no invented credentials | Detects a narrow class of misclassified brief hiring requests |
| JD-001951 and other records infer mandatory generic `otherskills` | Do not elevate catch-all soft skills to mandatory or a hard gate without concrete JD support | Issues a manual-review warning for suspicious mandatory `otherskills` and blocks publication until reviewed |

The full and schema-mode prompts are both **complete replacements**, not patches. The top-level and nested output schema keys/enums are unchanged. The JSON schema file is byte-for-byte identical to v2.1. The new rules add no fields and no extra model calls. More detailed JD rubrics can still generate more tokens, so measure actual latency.

## Files

- `JD_Rubric_Extractor_v2.2_FULL.txt` — **complete standalone system prompt** (best for simple replacement)
- `JD_Rubric_Extractor_v2.2_SCHEMA_MODE.txt` — complete shorter prompt when strict schema decoding is supported
- `jd_rubric.schema.json` — **unchanged** v2.1 strict schema
- `rubric_validator.py` — updated independent validation and human-review flags
- `tests/test_semantic_v22.py` — 9 tests grounded in prior actual extraction records
- `tests/fixtures/results_full_text.csv` — previous 12-run dataset, used only for regression testing
- `historical_v22_audit.json` — detailed results **against old outputs**, NOT newly rerun inference

## How to deploy

1. Replace only the **JD extraction** system prompt with `JD_Rubric_Extractor_v2.2_FULL.txt`. Keep the model, context, token budget and sampler identical to the previous run to make the test fair.
2. Run the same twelve source JDs again. Do not re-use outputs from the old prompt as evidence that v2.2 performed better.
3. If your serving backend supports strict JSON schema decoding, use `JD_Rubric_Extractor_v2.2_SCHEMA_MODE.txt` along with `jd_rubric.schema.json`. Do not use schema mode by itself.
4. Validate each rubric. **Cache only when `valid_for_cache=true`**. Warnings that require human review, including suspect generic-skill mandates, prevent cache publication. A valid JSON `not_a_jd` or `multiple_roles` output is not a usable candidate rubric.
5. Version cache keys by raw JD text, extractor prompt, model and schema. Re-extract old cache entries when promoting v2.2.

```bash
pip install jsonschema pandas
python -m unittest discover -s tests -v
python rubric_validator.py --jd job.txt --rubric extraction.json --report validation_report.json
```

## Critical contract details

- Enrollment is **not** the same as having earned a degree. The existing `education_requirement.min_level` represents an attained credential. For a student eligibility condition, use `requirements[].category="other"`, an exact `source_quote`, and `education_requirement.min_level=null` unless a completed degree is separately required. Update any downstream adapter that treats every education-related `other` item as a completed degree.
- `otherskills` is not a hard-gate heading. Preserve explicit core requirements when genuinely stated; do not automatically rewrite all generic skills as optional if the JD independently makes them required.
- Short informal hiring requests may have no skill list or numeric qualifications. **Do not fabricate them.** Use `input_status="partial"`, keep source-supported duties, and send uncertain cases for human review.
- All exact `source_quote` validation and duty-linked skill checks from v2.1 remain.
- The validator catches specific known failure patterns, not semantic correctness in every possible JD. Review representative JD rubrics against human-annotated inventories.

## Evidence and benchmark status

The 9 added tests pass. The historical audit flags old outputs requiring changes; no new LLM inference was run. Therefore actual extraction accuracy, tokens and latency for v2.2 remain **unmeasured**.

**Acceptance criteria for promotion:** 12/12 usable JDs correctly recognized (including partial informal request); intern enrollment not misrepresented as completed degree; generic catch-all skills not silently elevated; no regression in explicit must/preferred skill coverage, CI/CD duty coverage, exact quotes or JSON schema validity; compare token count and wall clock with previous v2.1, then validate a larger adjudicated set.
