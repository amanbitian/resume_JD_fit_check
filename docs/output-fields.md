# Prediction output fields

The scorer writes append-only JSONL: one complete JSON object per candidate/JD pair. Nested
objects are retained so every fact used by the evaluator is auditable.

## Identity and reproducibility

| Field | Meaning |
|---|---|
| `pair_id` | Stable source-file and row identifier used for resume/restart |
| `content_hash` | Hash of the exact resume/JD pair |
| `prompt_hash` | Hash of the exact system rubric |
| `scoring_fingerprint` | Hash of model, prompt, truncation limits, and output-token budget; resume skips only matching successful rows |
| `source_file`, `row_index` | Location of the original input |
| `input_resume_chars`, `input_jd_chars` | Original input lengths |
| `gold_label` | Dataset label, retained for offline evaluation and never sent to the model |
| dataset metadata fields | `source`, `resume_domain`, and `jd_domain`, retained only for analysis and never sent to the model |

## Model decision and evidence

| Field | Contents |
|---|---|
| `fit_score` | Application-derived integer 0-10 using fixed dimension weights and mandatory-gap caps |
| `fit_label` | Application-derived `No Fit`, `Potential Fit`, or `Good Fit` |
| `confidence` | Evidence-completeness confidence from 0-1 |
| `jd_analysis` | Job title, role family, seniority, functional/industry domains and subdomains, minimum years, education/certification requirements, every JD skill, and critical skills |
| `candidate_analysis` | Primary/current role, demonstrated seniority, functional/industry domains and subdomains, experience totals, education, certifications, and every work experience |
| `dimension_scores` | 0-10 or non-applicable scores for skills, years, work experience, role alignment, domain alignment, education/certifications, and other requirements |
| `score_method` | Fixed weights, uncapped score, and any mandatory-gap caps applied by the application |
| `role_alignment`, `domain_alignment` | Controlled alignment level plus concise supporting evidence |
| `skill_assessment` | One match record for every extracted JD skill: priority, status, evidence, estimated years, and score |
| `requirement_checks` | One check for every explicit JD requirement, including non-skill requirements |
| `plus_points` | Important or mandatory requirements supported by resume evidence; derived from requirement checks |
| `gaps` | Partial or not-evidenced requirements, including priority and importance; derived from requirement checks |
| `hard_requirements_met` | Derived from whether the mandatory-gap list is empty |
| `unmet_mandatory_requirements` | Every mandatory requirement not fully evidenced |
| `needs_human_review`, `human_review_reasons` | Explicit abstention/review routing for incomplete, contradictory, ambiguous, or low-confidence evidence |
| `rationale` | Final concise hiring-manager explanation |

`jd_analysis.all_jd_skills` contains `{skill, skill_type, priority, jd_evidence}`.
`jd_analysis.top_critical_skills` contains `{skill, why_critical, jd_evidence}`.
`candidate_analysis.work_experiences` contains `{role_title, employer, start_date, end_date,
duration_years, functional_domains, industry_domains, skills_evidenced, relevance_to_jd,
relevance_reason}`.

## Runtime and dual-GPU attribution

| Field | Meaning |
|---|---|
| `status`, `error` | Success/error state; error rows are retried on resume |
| `model`, `endpoint` | Model and service used |
| `optimized_pipeline` | `true` when cached extraction plus indexed pair scoring produced the row |
| `pair_attempts` | Structured pair-generation attempts; bounded at two |
| `analysis_cache` | JD/resume hashes and whether each analysis was reused |
| `usage` | Prompt, completion, and total tokens when supplied by the server |
| `finish_reason` | Must not be `length`/`max_tokens`; partial answers are rejected as errors |
| `reasoning_tokens` | Hidden/thinking-token count reported by the backend, when applicable |
| `context_window` | Loaded model context reported by the service, when available |
| `dgpu_run_id` | Shared `dual_gpu_setup` run identifier |
| `timing`, `latency_seconds` | Prefill/decode/end-to-end performance |
| `pipeline_latency_seconds` | End-to-end row time including cache lookup and cold extraction calls |
| `resources` | GPU and memory telemetry returned by `dual_gpu_setup` |
| `client_delivery` | Confirms metrics were written to this repository's registered output |
| `scored_at` | UTC timestamp |

Raw resume and JD strings are not duplicated into prediction JSONL. They stay in the parquet and
are addressable through `source_file` + `row_index`; this avoids doubling storage and copying
sensitive text into metrics artifacts.
