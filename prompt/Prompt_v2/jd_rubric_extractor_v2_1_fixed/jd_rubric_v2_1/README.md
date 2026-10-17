# JD Rubric Extractor v2.1 — validated package

## What changed

This is a **JD extraction** system, not a resume evaluator. Extract a rubric **once per unique JD**, validate it, then reuse it for every JD–resume pair with that job. It retains all 29 top-level fields and their nested keys in the previous optimized JD prompt. **There are no model calls in the validator.**

| Failure identified in 12-JD / 24-run audit | Repair |
|---|---|
| Wrong response structure (`role_profile` etc.) | Explicit exact top-level contract + strict machine-readable JSON Schema |
| Quotes altered or fabricated | Exact-substring validation for D#, S#, R#; no semantic auto-repair |
| Named skills in duties (e.g. `implement CI/CD`) omitted | Prompt instruction + high-precision duty/tool inventory checks |
| Mandatory/hard-gate priorities inflated | Independent `priority`, `importance`, `priority_origin`, and `hard_gate_explicit` validation |
| Truncation or markdown fences | Classify finish_reason `length` as invalid; safely remove only a *whole* JSON markdown fence for legacy audit |

## Files

- `JD_Rubric_Extractor_v2.1_FULL.txt` — **standalone system prompt** if your serving backend is not enforcing a JSON response schema. Includes the complete existing output skeleton and vocabulary.
- `JD_Rubric_Extractor_v2.1_SCHEMA_MODE.txt` — **shorter prompt** for backends that support an externally supplied JSON Schema. Use with the schema file, not by itself.
- `jd_rubric.schema.json` — strict field/type/enum contract; `additionalProperties=false` for all objects.
- `rubric_validator.py` — downstream independent checks for quotes, IDs, groups, critical links, hard gates, and high-confidence named skill coverage.
- `tests/test_validator.py` — 12 regression tests.
- `original_24_run_audit.json` — analysis of the historical `jd_extraction_12.csv` outputs.
- `build_schema.py` / `generate_prompt.py` — reproducible schema and prompt generators; optional to run.

## Usage

```bash
pip install jsonschema
python -m unittest discover -s tests -v
python rubric_validator.py --jd path/to/jd.txt --rubric path/to/rubric.json --report validation_report.json
python rubric_validator.py --eval-csv /path/to/jd_extraction_12.csv --report legacy_audit.json
```

`--jd` + `--rubric` returns exit code **0** on all blocking checks passed and **1** otherwise. Never publish cache entries with `valid_for_cache=false`. The CSV audit is informational and prints its pass count.

### Hosted or local inference / constrained generation

If your runtime supports **JSON Schema** response formatting (not just `json_object` mode), provide `jd_rubric.schema.json` as the response schema and use `JD_Rubric_Extractor_v2.1_SCHEMA_MODE.txt` as system instructions. Model backends differ on supported schema keywords and whether `strict=true` is implemented. Test that all fields and enum restrictions are actually enforced. When the runtime does not support strict schemas, use `...FULL.txt` and still validate every response with `rubric_validator.py`.

A generic OpenAI-compatible request pattern (adapt field shape to your serving API):

```python
import json
from pathlib import Path
schema=json.loads(Path('jd_rubric.schema.json').read_text(encoding='utf-8'))
response = client.chat.completions.create(
    model=model,
    messages=[
        {'role':'system','content':Path('JD_Rubric_Extractor_v2.1_SCHEMA_MODE.txt').read_text(encoding='utf-8')},
        {'role':'user','content':'REFERENCE_DATE: 2026-10-17\n<JOB_DESCRIPTION>\n' + jd + '\n</JOB_DESCRIPTION>'}
    ],
    temperature=0,
    max_tokens=6144,
    response_format={'type':'json_schema','json_schema':{'name':'jd_rubric_v21','strict':True,'schema':schema}},
)
```

`response_format` support varies. Use `max_tokens=6144` as the current baseline, **not a blanket new global default**; test 8192 only for JDs that actually hit the output limit and fit in context. Save raw output and finish reason before validation. A length-finished response is invalid even if its partial JSON looks plausible. Avoid identical retries.

### Safely handle failures

1. **Generation `length`**: fail, save a protected diagnostic and retry once with a larger stage-specific budget or split unusually long JDs into bounded extraction batches followed by *validated* merge.
2. **Invalid JSON/fences**: if your backend returns a whole Markdown fence, strip ONLY the outer fence and revalidate. Do not parse arbitrary trailing fragments or make up missing fields.
3. **Schema failure**: fail. Regenerate against strict schema; do not blindly rename `input_validity` to `input_status` since nested structures differ.
4. **Wrong quote**: reject or request only corrected quotes using the source JD; don't turn a paraphrase into a source quote.
5. **Missing listed or duty-named skill**: reject or request targeted correction. Preserve underlying skill priority; a duty-named skill is **not automatically an explicit hard gate**.
6. **Priority ambiguity**: keep `hard_gate_explicit=false` and mark for rubric review. Do not infer rejections from a core duty alone.
7. **Human-review criteria**: do not use protected traits; document job-related reasons and audit for disproportionate exclusion.

## What the historical audit establishes

| Model | Run count | Parseable after fence cleanup | Strict schema + validator pass |
|---|---:|---:|---:|
| Qwen 3.8 27B | 12 | 11 | **0** |
| Gemma 4 12B QAT | 12 | 12 | **0** |

Those outputs were generated **before** this repair, under a different schema. The 0/24 pass count does not measure future accuracy of the revised prompt. **No new inference was executed.**

## Testing and acceptance

For a new extraction sample, require: (a) full schema valid, (b) every actual `source_quote` exact, (c) all listed skills present with proper priorities, (d) all explicitly duty-named technologies covered, (e) coherent OR groups and hard-gate provenance, (f) human-reviewed comparison for distinct duty completeness and semantic qualification precision. The validator deliberately does **not** assume its heuristic skills inventory is a complete gold-standard representation of complex prose JDs.

Test both Qwen and Gemma on the same 12 JD cases with the **same** schema-mode setup where feasible. Compare valid extractions, manual JD requirement recall/precision, hard-gate false positives, quoted evidence and total generation tokens. Cache only validated rubrics under hashes of original JD + model + prompt/schema version; include model/runtime configuration in metadata if reproducibility is required.

## Downstream compatibility caveat

The exact v2.1 extractor fields match the supplied v2 optimized JD extractor, but your **candidate scoring prompt** may still expect raw JD text and its own R1... requirement list. Do not assume these two JSON structures interoperate directly. Add an explicit rubric-to-candidate-check adapter (mapping D#/S#/R# identifiers) if your score path consumes this rubric. Validate on test cases before switching the scorer.
