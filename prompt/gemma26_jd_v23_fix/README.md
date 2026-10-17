# Gemma 4 26B-A4B QAT Q4_0 — JD v2.3 fix experiment (v0.1)

**Purpose:** Test a targeted fix for Gemma's schema drift without altering your working Qwen3.8-27B JD rubric extractor, v2.3 rubric contract, or semantic acceptance criteria.

**What is inside:** A runnable Python adapter, a minimal constrained-decoding smoke test, 20-JD benchmark runner, explicit Gemma field rules, exact-quote checks, content-addressed fingerprints, and regression tests.

**Important limitation:** I could inspect your supplied benchmark logs, but could not fetch the GitHub repository's current source. This ZIP is a **non-destructive integration patch**, not a claim that your repository has already been modified or that the improvement is proven. It needs your current v2.3 schema and prompt paths and an integration hook into the project's actual validator. See `INTEGRATION.md`.

## 1. Installation (on the PC, inside your project or a separate test folder)

Unzip this package. From its root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

It makes **no model-download or model-load calls**. The relevant model must already be running on your `9070xt` lane, and you need an OpenAI-compatible HTTP gateway that forwards `response_format`.

## 2. Preflight: prove schema constraints work on your llama.cpp build

Run the probe, specifying your exact loaded Gemma model name and already-running service endpoint:

```bash
python -m gemma_jd_fix probe \
  --api-base http://127.0.0.1:8080/v1 \
  --model gemma-4-26B-A4B-it-QAT-Q4_0 \
  --schema-mode llamacpp \
  --output-dir outputs/gemma26_schema_probe
```

The example port and name are placeholders: use your actual service's address and model identifier. The probe sends two short requests; the second deliberately requests an invalid enum. **If the response remains schema-valid, it is evidence of constraint enforcement**, though a production-schema test is still required. If your gateway uses the OpenAI-style envelope instead, repeat with `--schema-mode openai`. Do not automatically switch formats on HTTP errors without recording them.

A failing probe is a **stop condition**: inspect the gateway forwarding, server logs, and potential llama.cpp grammar bugs before sending 20 long JDs. Test the service's short request after any gateway change; do not replace the working Qwen deployment.

## 3. Controlled 20-JD baseline with exactly the same Qwen v2.3 prompt

Export your *existing* v2.3 prompt to `qwen_v23_prompt.txt`, and your existing schema to `rubric_v23.schema.json`. Use the same JD input CSV from your benchmark containing columns `jd_id` and `jd`.

First inspect the payload without making a network request:

```bash
python -m gemma_jd_fix benchmark \
  --input /path/to/20jd_input.csv \
  --schema /path/to/rubric_v23.schema.json \
  --system-prompt /path/to/qwen_v23_prompt.txt \
  --model gemma-4-26B-A4B-it-QAT-Q4_0 \
  --schema-mode llamacpp \
  --prompt-mode unchanged \
  --dry-run
```

Then run, only after the smoke test passes:

```bash
python -m gemma_jd_fix benchmark \
  --api-base http://127.0.0.1:8080/v1 \
  --input /path/to/20jd_input.csv \
  --schema /path/to/rubric_v23.schema.json \
  --system-prompt /path/to/qwen_v23_prompt.txt \
  --model gemma-4-26B-A4B-it-QAT-Q4_0 \
  --schema-mode llamacpp \
  --prompt-mode unchanged \
  --validator-hook /path/to/your_real_v23_adapter.py \
  --max-tokens 12288 \
  --limit 20 \
  --output-dir outputs/gemma26_v23_constrained_baseline
```

**The `--validator-hook` must call your existing, unchanged v2.3 semantic validator.** This package includes a fail-closed template, not a mock acceptance shortcut. If you omit `--validator-hook`, the runner correctly records `semantic_checked=False` and `accepted=False`, regardless of JSON/schema success. This makes partial integration safe to test.

The runner creates `attempts.jsonl` (including raw output and validator failures), `results.csv`, and `config.json` in a new output directory. It does **not** overwrite your preexisting `outputs/production_20jd/` or modify any source model configuration.

## 4. Only if constraints aren't sufficient: targeted Gemma wording

Repeat with `--prompt-mode targeted` and a **new** output directory. The addendum is in `gemma_jd_fix/core.py` as `GEMMA_RULES`. It specifically distinguishes:

- `key_responsibilities[].importance` = `core` or `secondary`
- `skills[].importance` = `critical` / `important` / `supporting`
- `skills[].priority` = schema-permitted values, **not** `inferred` or `responsibility`
- education/experience objects must use permitted fields (`jd_evidence`, not an invented `source_quote`)
- all required fields, hard-gate source evidence, listed technical skill coverage, and ID consistency

No automatic quote repair, taxonomy remapping, silent field deletion, or priority rewriting is included.

## 5. Acceptance and safety

- 20/20 parseable JSON and zero schema errors are necessary but not sufficient.
- Still require the project's **real semantic validator**, plus human-labeled evaluation of mandatory skills and hard gates.
- Review `JD-015413` manually before altering precedence: source `mustskills` and `goodskills` overlap. Do not fix by blindly changing mandatory to preferred.
- Check missing CI/CD skills on `JD-000753`, `JD-000961`, `JD-003899`.
- 12,288 output tokens plus ~8K prompt is close to your planned 20.5K context. Verify actual tokenized prompt, model overhead, and allocation with the service before trying the full 20 JDs.
- Do not use a schema-only accepted count as a reason to change production.
- The script makes one retry (same temperature and seed) by default; it logs every attempt. Use your existing retry policy in production.

## Contents

```text
gemma_jd_fix/          Python implementation
examples/              Fail-closed real-validator hook template; minimal schema
INTEGRATION.md         How to add response_format forwarding to dual_gpu_setup
requirements.txt       jsonschema dependency
tests/                 Offline unit tests
README.md              This file
```
