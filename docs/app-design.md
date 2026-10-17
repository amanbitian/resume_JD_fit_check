# Candidate Evaluation Application Design

## Processing flow

```text
Parquet/CSV ingestion
        |
        v
Normalization + SHA-256 identity layer
        |
        +-----------------------+
        |                       |
        v                       v
Unique JD analyzer         Unique resume analyzer
(versioned cache)          (versioned cache)
        |                       |
        +-----------+-----------+
                    v
          Pair-evaluation queue
          (bounded parallelism)
                    |
                    v
       Schema validation + retry
                    |
                    v
 Deterministic weighted score + mandatory caps
                    |
                    v
       Append-only result/audit store
                    |
          +---------+---------+
          |                   |
          v                   v
   Review dashboard      Evaluation reports
   + adjudication        + model comparison
```

The extraction cache key is `(content hash, model, prompt/schema/inference contract)`. The
pair-result key is `(resume hash, JD hash, scoring fingerprint)`. Changing a prompt, schema, model,
output limit, or generation setting creates a different fingerprint and cannot silently reuse an
incompatible analysis or result.

## Services

1. **Ingestion and identity** validates columns, normalizes only line endings/whitespace chosen by
   policy, calculates SHA-256 hashes, and records source lineage. Raw resumes remain in the secure
   data boundary.
2. **JD rubric extraction** runs once for each unique JD and produces the fixed requirements used
   for every candidate attached to that JD.
3. **Resume evidence extraction** runs once for each unique resume and produces a JD-independent
   factual profile.
4. **Pair evaluation workers** compare cached structures. The queue is bounded, retries malformed
   structured responses, and supports checkpoint/resume.
5. **Policy scorer** calculates the final score in application code from versioned weights and
   mandatory-gap caps. The model cannot choose its own final score or label.
6. **Review and adjudication** sends low-confidence, ambiguous, and sampled ordinary cases to a
   human reviewer. Reviewer corrections are stored separately from immutable model output.
7. **Evaluation and monitoring** reports coverage, errors, class metrics, calibration, drift,
   disagreement, throughput, token use, and cost by scoring fingerprint.

## Core tables

| Table | Key | Important fields |
|---|---|---|
| `model_run` | `run_id` | model, provider, prompt/schema hashes, generation config, time, status |
| `jd_analysis` | `jd_hash + contract_hash` | domains, role, skills, requirements, usage |
| `resume_analysis` | `resume_hash + contract_hash` | roles, dates, skills, domains, usage |
| `pair_evaluation` | `pair_hash + scoring_fingerprint` | dimensions, checks, score, label, rationale, usage |
| `human_review` | `review_id` | pair key, reviewer label, corrections, notes, adjudication status |

## Review dashboard

The main screen should show JD and resume evidence side by side with a requirement matrix in the
middle. Reviewers can filter by model, JD/domain, score band, confidence, mandatory gaps, error
status, and model disagreement. Each assessment row displays the source evidence, model status,
and a reviewer correction control. A comparison page reports the same fixed human-labeled sample
for each model rather than comparing models on different records.

For quality estimation, review a stratified random sample across predicted labels, score bands,
domains, confidence bands, and mandatory-gap states. Weight stratum metrics back to production
prevalence and report confidence intervals. Keep a separate targeted error slice for diagnosis;
do not mix that biased slice into the production accuracy estimate.
