# Production runs

Three tasks, each runnable on a local model or a cloud model:

| Task | Items | Prompt |
|---|---|---|
| `resume_extraction` | unique resumes (`Input/resume_dataset.csv`) | `prompt/old prompt/1_resume_extraction_*` |
| `jd_extraction` | unique JDs (`Input/jd_dataset.csv`) | `prompt/jd_rubric_extractor_v2_3_targeted/jd_rubric_v2_3/`: the v2.3 FULL prompt, its schema and validator (see below) |
| `resume_jd_pair` | resume/JD pairs (`Input/Candidate_JD_eval_159k.parquet`) | `prompt/Prompt_v2/prompt.txt`, scored by `prompt/Prompt_v2/scoring.py` |

## Start

```
candidate-fit wizard
```

The wizard asks for the task, then local or cloud, then the model, then how many items to run (Enter for all).
It shows the plan and asks for confirmation before anything is sent. Cloud runs always use the provider's
batch API. Before a cloud submission the wizard shows the estimated cost and the data that will leave the
machine, and asks you to confirm that you may send it.

Without the wizard:

```
candidate-fit run --task jd_extraction --backend local --local-model gemma-4-12b --limit 12
candidate-fit run --task resume_jd_pair --backend cloud --cloud-provider anthropic --cloud-model claude-haiku-5-5 --max-cost-usd 25 --yes
```

## Keys and settings

- Cloud keys come only from environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. They are never written to disk.
  A missing key is reported before anything is confirmed.
- `OPENAI_LUNA_MODEL` sets the default GPT Luna model id; the wizard asks for the exact id each time.
- `--reasoning-effort` (OpenAI only): hidden reasoning counts against the output cap, so a reasoning model needs a
  larger `--max-output-tokens` or a lower effort. A reply cut off for that reason says so.
- `--max-cost-usd` refuses a cloud run whose worst-case estimate (every output at the cap) is higher.
- Local runs need the dual GPU service (`DGPU_ROOT`, default `F:/Projects/Dual_gpu_setup`) and a client registration
  (`.candidate-fit-client.json`, or `CANDIDATE_FIT_CLIENT_ID`). `--reasoning-budget` sets the local thinking budget
  (default 0, thinking off). Another application's model on the lane is not replaced unless `--replace-other-model`.

## The JD prompt and its schema

JD extraction uses a rubric package: a complete FULL system prompt (it already contains the output schema and the shared
vocabulary), `jd_rubric.schema.json` and `rubric_validator.py`. Every reply is checked by that package's validator. The
schema-mode prompt is not used, because neither the dual GPU service nor the cloud batch clients forward a response schema.

| Version | Package | Output folder |
|---|---|---|
| `v2.3` (default) | `prompt/jd_rubric_extractor_v2_3_targeted/jd_rubric_v2_3/` | `outputs/production/jd_extraction_v2_3/` |
| `v2.3-gemma26` (gemma-4-26B-A4B only, chosen automatically for it) | `prompt/jd_rubric_extractor_v2_3_targeted/jd_rubric_v2_3/` plus the rules in `prompt/gemma26_jd_v23_fix/` | `outputs/production/jd_extraction_v2_3_gemma26/` |
| `v2.3-gptoss` (gpt-oss-20b only, chosen automatically for it) | `prompt/gptoss20b_candidate_processing_optimized_v1/gptoss20b_rubric_v1/` | `outputs/production/jd_extraction_v2_3_gptoss/` |
| `v2.2` | `prompt/jd_rubric_extractor_v2_2_semantic_fixed/jd_rubric_v2_2/` | `outputs/production/jd_extraction_v2_2/` |
| `v2.1` | `prompt/Prompt_v2/jd_rubric_extractor_v2_1_fixed/jd_rubric_v2_1/` | `outputs/production/jd_extraction/` |

Pick one with `--jd-prompt-version` (`candidate-fit run` only; the wizard uses the default). **Each version has its own
folder, so a new prompt never writes into the results of an older one.** The version is also part of the fingerprint
(prompt, schema, validator, output cap and reference date), so results from one version are never reused by another.
The v2.1 and v2.2 folders and fingerprints are unchanged, so their results can still be resumed or exported with
`--jd-prompt-version v2.1` or `v2.2`. The JSON schema is the same in all three versions.

### What v2.2 adds (the schema is unchanged)

| Rule in the prompt | What the validator checks | What the runner does |
|---|---|---|
| A JD that says the person is currently studying for a degree is an enrollment condition, not a completed degree: it goes in `requirements` as category `other`, and `education_requirement.min_level` stays null | `ENROLLMENT_CONDITION_MISSING`, `ENROLLMENT_AS_COMPLETED_DEGREE` | rejects the reply and, on a local run, sends the reason with the retry |
| A short but intelligible hiring request ("need help with accounting and taxes") is `partial`, not `not_a_jd` | `MISCLASSIFIED_INFORMAL_JD` | same |
| A catch-all `otherskills` entry is not mandatory unless the JD says so for that item | `OTHER_SKILLS_PRIORITY_REVIEW`, `HARD_GATE_QUOTE_WEAK` are warnings that need a review | rejects the reply with those warnings as the reason. Before v2.2 a warning alone never blocked a reply |

Everything v2.1 checked still applies: exact quotes, id sequences, any-of groups, critical-skill links and hard-gate
provenance.

### What v2.3 adds (the schema is unchanged)

| Rule in the prompt | What the validator checks | What the runner does |
|---|---|---|
| Named examples of a category ("enterprise SaaS (Salesforce, ServiceNow, etc.)") stay as separate skills in one open any-of group with `min_required=1`, not one generic "SaaS Development" skill | `MISSING_OPEN_EXAMPLE`, `MISSING_OPEN_ALTERNATIVE_GROUP`, `INVALID_OPEN_ALTERNATIVE_GROUP` | rejects the reply with the reason |
| Every distinct competency listed under `otherskills` ("problem solving, agile, team leadership") is kept, as preferred and never as a hard gate | `MISSING_EXPLICIT_OTHER_SKILL`, `GENERIC_SKILL_HARD_GATE` | same |
| A stated sponsorship or relocation policy is kept as its own condition in `requirements`, with its exact wording, and is not a skill | `MISSING_EXPLICIT_LOGISTICS`; `LOGISTICS_POLICY_REVIEW` is a warning that needs a review | same. This is a review warning, so it blocks the reply even with no errors |

The validator checks a closed list of patterns, not meaning, so it catches these known cases and not every possible
omission.

### gemma-4-26B-A4B has the Gemma field rules (`v2.3-gemma26`)

With the shared v2.3 prompt, Gemma-26B (QAT Q4_0, thinking off) gave 2 usable rubrics of 20. The failures were format drift:
35 attempts used a word outside the allowed list for a responsibility's `importance` (`supporting` instead of
`core`/`secondary`) and 37 added a `source_quote` field that the education and experience objects do not have. The
`gemma26_jd_v23_fix` package has a block of rules (`GEMMA_RULES`) written for exactly that. `v2.3-gemma26` is the v2.3 prompt
**exactly as Qwen uses it, followed by that block**, and it is used **only for gemma-4-26B-A4B**: the plan picks it
automatically for that model, and any other model or the cloud is refused if it asks for it. The schema, validator, output cap
(12,288), decoding (temperature 0, seed 42), retry and thinking-off setting are the same as Qwen's. The rules are read from
the package file when the prompt is built, and they are part of the fingerprint, so editing them starts new results. The
earlier Gemma-26B results stay in their own folder. `--jd-prompt-version v2.3` still gives it the plain prompt.

**Not applied.** The package's main experiment is *schema-constrained decoding* through an HTTP gateway that forwards
`response_format`, with its own runner (`python -m gemma_jd_fix ...`). Our evaluation path cannot forward a schema (see the gpt-oss
section), and the package's runner calls an HTTP address directly, which we do not do for local models. Only the package's
prompt rules are used. Its README says to try the constrained baseline first and the targeted wording only if that still
fails, so a result here does not show what constrained decoding would give. The package's semantic-validator hook is not
needed, because our validator is the unchanged v2.3 one. Gemma-4-12B is not covered: the rules are for the 26B model.

### gpt-oss-20b has its own JD prompt (`v2.3-gptoss`)

On the 12 comparison JDs, gpt-oss-20b with the shared v2.3 prompt gave 1 usable rubric: invalid JSON (a stray quote after
`null` or `true`), quotes with a plain hyphen swapped for U+2011, and skipped listed skills. The gpt-oss package was written
for those failures. `v2.3-gptoss` uses it, and **only for gpt-oss-20b**: the plan picks it automatically for that model,
and a plan for any other model or for the cloud that asks for it is refused. Qwen, Gemma and the cloud run exactly as
before. Specifically:

| Setting | `v2.3-gptoss` | Every other model |
|---|---|---|
| System prompt | `GPTOSS20B_JD_EXTRACTOR_STANDALONE.txt`, sent **without** its closing `# Response Formats` block (see below). The package file itself is left as shipped | the v2.3 FULL prompt, sent as the file reads |
| Schema and validator | the package's copies, byte-identical to v2.3's, so the same checks | v2.3's |
| Output cap | 8,192 (the package's starting point) | 12,288 |
| A reply that repeats a JSON key (`"min_years","min_years":null`) | refused with the reason | read as before (the last value wins) |
| Decoding on the retry | temperature 0.3 and seed 43, because the same greedy request repeated the invalid JSON on 7 of the 8 items that had it | unchanged: temperature 0, seed 42 |
| Results go to | `jd_extraction_v2_3_gptoss/` | their own folder |

The retry values are our choice. The package only says to retry with a changed setting or decoding state, not which.

**Why the `# Response Formats` block is not sent.** With the prompt exactly as shipped, all 24 attempts on the 12 JDs failed
before any answer came back: the model server returned HTTP 500, "The model produced output that does not match the
expected peg-native format", so its own parser for gpt-oss rejected the reply. The shared v2.3 prompt never triggered it.
On one JD (JD-000314), the same request with only those three lines removed
(`# Response Formats` / `## jd_rubric` / `// Final response: ...`) returned valid JSON. This was tested on that one JD,
and the server version matters: the package itself warns that Harmony/GPT-OSS decoding on some llama.cpp versions has
issues. The removal is in `jd_system_prompt` in `tasks.py` and is part of the fingerprint, so results made with the block
are never reused.

**Not applied, and why:**

- *Schema mode* (`GPTOSS20B_JD_EXTRACTOR_SCHEMA_MODE.txt` with a response schema). The service's evaluation chat path
  forwards only the messages, temperature, top_p, max_tokens and seed. A schema is forwarded only by its
  OpenAI-compatible gateway, which needs the model reloaded in coding-agent mode and a different client. That is a separate
  piece of work. The package says to use the standalone prompt when the interface cannot send a schema.
- *Reasoning effort `low`.* The same path cannot set a per-request reasoning effort. gpt-oss stays on the thinking-off
  deployment it is loaded with.
- *The candidate matcher* (`GPTOSS20B_CANDIDATE_MATCHER_EXPERIMENTAL.txt`). It scores a resume against a validated JD
  rubric, and resume-JD pair scoring here reads the raw JD, so there is no rubric to give it. It is marked experimental in
  the package. Pair scoring is unchanged for every model.

`verify_gptoss.py` and `run_local_once.py` are the package's own command-line tools and are not used by the runner. Its
duplicate-key check is now part of reply validation.

### Employment logistics are not job fit

Sponsorship, relocation, work-authorisation, shift and location conditions say nothing about a candidate's skills, and a
"sponsorship is not available" line says nothing about any one person's right to work. From v2.3 on:

- the `logistics_conditions` column lists them for each rubric (empty when the JD states none; blank for results made
  with v2.1 or v2.2, which did not separate them);
- `candidate_fit.tasks.split_requirements(rubric)` returns `(fit_requirements, logistical_conditions)`. Anything that
  scores a candidate against a rubric must use the first list for job fit and send the second to a separate, human-reviewed
  step. Nothing in this repository scores against a rubric yet (resume-JD pair scoring reads the raw JD), so there is
  nothing to rewire today. The first thing that does must call this function.

`hard_gate_explicit=true` on a logistics condition means only that the JD states it as non-negotiable. It is not a
reason to reject a candidate or to lower a technical score.

### Which results are a usable rubric

A reply that passes the checks is final. The `valid_for_cache` column says whether it is also a rubric that can be used
to match candidates:

- `True`: `input_status` is `ok` or `partial`. Use these.
- `False`: the model correctly said the input is `not_a_jd`, `empty_or_unreadable` or `multiple_roles`. The reply is kept
  because asking again cannot help, but it is not a rubric. Do not cache it or score a candidate against it.
  `validator_warnings` starts with `NOT_A_RUBRIC`.

### Running the 12 comparison JDs again

[`config/jd_extraction_12_ids.txt`](../config/jd_extraction_12_ids.txt) lists the 12 JDs that the earlier results and each
package's regression fixtures are based on:

```
candidate-fit run --task jd_extraction --backend local --local-model qwen3.8-27b --ids-file config/jd_extraction_12_ids.txt
```

Keep the model, context, output cap and sampler the same as the earlier run so the comparison is fair. The packages'
audit files (`historical_v22_audit.json`, `v22_outputs_under_v23_validator.json`) replay old outputs through a newer
validator. They are not new runs, so accuracy, tokens and latency of a newer version are not measured until it is run.

To compare two runs, use the v2.3 package's `compare_runs.py`. Our `results.csv` already has the columns it reads. Run it
from that folder, and judge both files with the same validator:

```
cd prompt/jd_rubric_extractor_v2_3_targeted/jd_rubric_v2_3
python compare_runs.py --baseline ../../../outputs/production/jd_extraction_v2_2/local-qwen3.8-27b/results.csv --challenger ../../../outputs/production/jd_extraction_v2_3/local-qwen3.8-27b/results.csv --report comparison.json
```

It reports validity and operational numbers, not accuracy against a human answer.

## Long local runs: progress, pause and resume

For a run that takes hours or days, give it its own results folder and a progress interval:

```
candidate-fit run --task jd_extraction --backend local --local-model qwen3.8-27b --output-root outputs\production_full_qwen --progress-every 3 --yes
```

- **Progress.** After every 3 rows it prints one line, for example
  `progress 1,206/18,530 (6.51%) | usable 1,150, failed 56 | last 30 rows 91 s/row | 17,324 left, ETA 18d 3h (about Mon 03 Nov 14:20)`.
  The speed and ETA are averaged over the last 30 rows, so they settle after a while. A row is settled when it is usable or out of attempts.
- **Pause.** Press Ctrl+C once. The request in flight finishes (up to about two minutes), the model is released, `results.csv`
  is written, and the command exits. Press Ctrl+C a second time to stop at once; that loses only the request in flight.
- **Resume.** Run the same command again, with the same `--output-root`. Finished rows are skipped, and a row that had failed
  once is retried with the reason it failed. The first line says how many are already settled.
- **What is saved.** `results.jsonl` in the run folder is the record. Every attempt is appended the moment it finishes, so a
  crash, a power cut or a closed window costs at most the request in flight, and a half-written last line is ignored.
  `results.csv` is rebuilt when the command ends. After a crash, rebuild it with the same command plus `--export-only`.
- **A broken service stops the run.** If the model service answers with an error 5 times in a row, the run stops and says so,
  instead of marking every remaining row failed. The few rows hit are left out of attempts; rerun them later with `--max-attempts 3`.
  If the service cannot be reached at all, the run stops after 3 rows in a row, as before.
- **Time.** At about 108 s per JD (measured on Qwen3.8-27B with the v2.3 prompt, weighted by the real JD lengths), all 18,530 JDs
  take about 557 hours, or 23 days of continuous running. A 10-hour night is about 330 JDs.

## Outputs

Each (task, backend, model) has its own folder, `outputs/production/<task>/<backend>-<model>/` (for JD extraction the
task part is the version's folder from the table above):

| File | What it is |
|---|---|
| `results.jsonl` | one line per attempt per item; the source of truth |
| `batch_state.json` | cloud batch ids and any submission that was started but not confirmed |
| `results.csv` | one row per item with the complete source text beside it (`jd`, `resume`, or both for pairs) and the output last; local runs include tokens/sec and prefill rate; pairs also include verdict, score, reason and breakdown |
| `.lock` | held while a run is active; a second run in the same folder is refused |

### Reading `finish_reason`

`stop` is the good value: the model reached the end of its answer on its own. `length` means the answer was cut off at
the output cap (or the context was full); such a reply is always recorded as an error. A `stop` reply can still be
rejected by the validator, so check `status` and `error` for whether the content was accepted.

To rebuild `results.csv` from saved results without loading a model or calling an API, add `--export-only` to the
same `candidate-fit run` command. If the file is open in Excel, close it first.

## Resuming and retries

- Results are keyed by item id and tagged with a fingerprint of the task, prompts, validator, output cap and
  reference date. Rerunning continues where it stopped, a different sample reuses what it can, and a changed prompt
  never reuses older results (they stay in the file as history).
- A cloud batch submitted before a crash is polled, not resubmitted. If the crash came after the provider created a
  batch but before its id was saved, the next start looks for it at the provider (by metadata for OpenAI, by creation
  time and size for Anthropic). If it cannot tell which batch is the right one, it stops and says so.
- A reply that fails validation is retried in a second attempt, up to `--max-attempts` (default 2). The retry names
  the reason. A reply that raises an unexpected error in the validator is treated the same way.
- Network and rate-limit errors are retried with backoff (GET always, POST only on 429/5xx, never after a dropped
  connection). The local runner retries a busy or restarting service without using up an item's attempts and stops
  after three unreachable items in a row.
- Batches are split to stay under each provider's limits: 100,000 requests / 256 MB for Anthropic, 50,000 requests
  / 200 MB for OpenAI. Requests are built one batch at a time.
- Replies must be one JSON object. Extraction replies must contain the schema's required fields; pair replies are
  checked by the kit's calculator (every quote must appear in the source text).

## Limits and caveats

- The cloud batch clients are verified only with fake transports in `tests/test_cloud_batch.py`. Run a 2–3 item
  live batch before a large run; it costs cents.
- Cloud prices are secondary-source estimates (see `docs/cloud_pricing.md`).
- The system prompt is sent again in every request. For the full pair dataset that is about 0.7 GB of the 1.9 GB of
  request text; provider prompt caching could reduce that cost and is not used.
- Cloud result files are read into memory whole (a 100,000-reply batch is a few hundred MB).
- The local runner is sequential, one item at a time per lane; the summary warns when a run is large.
- Resume extraction uses the v3 prompt; its output is long, so output cost is the main cost driver.
- Results are matched to an item by its id and the fingerprint, not by a hash of its text. If `Input/jd_dataset.csv` or
  `Input/resume_dataset.csv` is rebuilt so that an id points to different text, earlier results for that id would be
  reused for the new text. Point `CANDIDATE_FIT_OUTPUT` at a new folder after rebuilding a dataset.
