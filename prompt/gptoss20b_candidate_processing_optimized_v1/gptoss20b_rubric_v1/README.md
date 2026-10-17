# GPT-OSS 20B — JD rubric and candidate matching prompts (experimental v1)

## Why this exists

In 12 historical JD extraction trials with GPT-OSS 20B, reasoning disabled, only 1/12 JD rubric outputs were accepted. Failures included malformed JSON after `null` and `true`, Unicode modifications of exact JD quotations, omitted skills, and excessive hard-gate classifications. The validated Qwen 3.8 v2.3 run is the control. **This package has NOT yet been run through GPT-OSS**; its accuracy/speed is unmeasured.

## Files

| File | Purpose | Use |
|---|---|---|
| `GPTOSS20B_JD_EXTRACTOR_SCHEMA_MODE.txt` | Preferred short prompt, no invalid pseudo-JSON template | Must pass `jd_rubric.schema.json` in response_format |
| `GPTOSS20B_JD_EXTRACTOR_STANDALONE.txt` | Full standalone replacement prompt, valid JSON shape illustration | When current inference interface cannot enforce JSON Schema |
| `jd_rubric.schema.json` | **Unchanged**, same 29 top-level fields as Qwen JD v2.3 | Attach to local backend; do not change downstream parser |
| `GPTOSS20B_CANDIDATE_MATCHER_EXPERIMENTAL.txt` | **Separate** JD+cached rubric+resume matcher, v2 compact output structure | For per-candidate stage, not interchangeable with JD extractor |
| `candidate_assessment.schema.json` | Compact scorer output schema | Attach for candidate matching |
| `rubric_validator.py` | The previously used v2.3 JD rubric semantic validator | Before JD cache |
| `verify_gptoss.py` | Rejects malformed or duplicate-key JSON; JD and candidate checks | After every model response |
| `run_local_once.py` | Optional example local API integration, no external API dependency | For a smoke test |
| `test_contracts.py` | 9 local schema/validator unit tests | CI or command line |
| `valid_json_shape_example.json` | Valid, EMPTY JD rubric key/type skeleton for reference | Not a production extraction |

## Fastest way to test

**If your existing app sends only a system prompt and no JSON Schema:** replace the JD extractor system prompt with `GPTOSS20B_JD_EXTRACTOR_STANDALONE.txt`. It is designed to preserve the same output keys as Qwen v2.3. The prompt alone cannot guarantee syntactic correctness or fidelity; do not turn off validators.

**Preferred:** set `GPTOSS20B_JD_EXTRACTOR_SCHEMA_MODE.txt` as system prompt and separately pass `jd_rubric.schema.json` via `response_format`. This reduces the chance that the model imitates an invalid schema text. Strict decoding ensures syntax only if your serving runtime implements it correctly; it cannot ensure all requirements are captured or quotations copied exactly.

### LM Studio / OpenAI-compatible example

For `/v1/chat/completions`, `response_format` is structured as:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "jd_rubric",
    "strict": true,
    "schema": {"...": "The entire contents of jd_rubric.schema.json, loaded as an object"}
  }
}
```

The `schema` value in this document is a **description**, not a usable schema. Use the actual `jd_rubric.schema.json` bytes as an object, as the provided runner does. Check whether your local backend supports the required JSON Schema constructs before relying on this mode. Affected llama.cpp/LM Studio versions have reported GPT-OSS / Harmony decoding issues.

```bash
pip install jsonschema
python -m unittest -v test_contracts.py

# Discover your locally loaded model ID from the runtime first:
python run_local_once.py jd --jd job.txt --model YOUR_LOADED_MODEL_ID --output jd_output.json --mode schema --reasoning-effort low
python verify_gptoss.py jd --jd job.txt --output jd_output.json

# Candidate processing, ONLY AFTER the JD rubric is validated and cached:
python run_local_once.py candidate --jd job.txt --rubric jd_output.json --resume resume.txt --model YOUR_LOADED_MODEL_ID --output assessment.json --mode schema --reasoning-effort low
python verify_gptoss.py candidate --jd job.txt --resume resume.txt --output assessment.json
```

The runner uses local `http://127.0.0.1:1234/v1/chat/completions` by default. `reasoning_effort` support differs by runtime, so inspect the server response/logs; the server may ignore or reject the field. Do not assume that requesting low reasoning automatically enables it. If ignored, set reasoning in the runtime's supported controls and run `--omit-reasoning-effort`. The runner uses a fixed demo reference date of 2026-10-17; change it for other dates.

**Do not manually construct or strip Harmony special tokens.** Let the serving runtime handle them. Ensure the correct GPT-OSS chat template is selected.

## Settings to benchmark, not guaranteed optimal

- Primary challenger: reasoning = **low**, temperature = 0, seed fixed; repeat at **medium** if semantic coverage is low.
- Output budget: start 8,192 total completion tokens for full JD rubric and reserve enough space for reasoning. Lower after observed distributions justify it; no token cap fixes semantically missing requirements.
- Context: enough for JD + system prompt + schema overhead and actual output. Do not assume context and output token budget are identical.
- Run 12 historical JDs with the same source texts and model quantization; compare model/runtimes as separate factors.
- For strict schema mode use same literal JSON Schema file for both runs; don't compare to prompt-only Qwen and claim an isolated model effect.
- Stop on JSON syntax failure or failed exact-quote validation. Retry *only with a changed setting or decoding state*, not identical deterministic settings. Store first-pass success and eventual success separately.

## Success criteria

| Metric | Desired result | Notes |
|---|---|---|
| Strict JSON parse, no duplicate keys | 12/12 | Measured on full raw final output, not repaired version |
| Complete rubric schema | 12/12 | 29 top-level keys + nested schema |
| Valid JD quotations | 100% | Strict Unicode codepoint equality, no silent normalization |
| Explicit mustskills/goodskills coverage | 105/105 for 8 structured JDs | Manual/source baseline from earlier runs |
| Named technologies in duties, especially CI/CD | All | Verify against original source |
| Open alternative groups and explicit logistics | Preserved | Manually review JD-000002 |
| Enrollment vs degree completed | Preserved | JD-000016 |
| Informal short job request | `partial` | JD-000314 |
| No invented internship experience min | `min_years=null` unless explicitly specified | JD-000016 |
| Lower median/total inference time | Compare with Qwen | Report extra retry/reasoning time |

## Limitations

This package verifies **output contracts**, not model-generation accuracy. No new GPT-OSS inference was performed here. All 9 unit tests exercise local JSON schemas and validator logic, not extraction quality. Existing semantic validator uses heuristic keyword checks and is not a replacement for human-labeled JD requirement inventory. The candidate matcher is experimental, and current compact v2 output keys are preserved but its behavior against a pre-extracted rubric must be benchmarked before moving production scoring. Human review and protected-trait exclusion apply throughout.
