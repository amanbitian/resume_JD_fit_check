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
