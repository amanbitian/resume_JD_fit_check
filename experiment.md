# Thinking-budget experiment: resume–JD fit scoring

**Status:** Partial benchmark; do not treat this as a final accuracy study.

## Question

Does allocating more model thinking tokens improve the accuracy of structured 0–10 resume–job-description fit scoring?

The pipeline scores each pair using evidence grounded only in the resume and JD.  It checks job/candidate domains, indispensable JD skills, other skills, years of experience, education, and work experience.  The output must conform to the required JSON schema, including a check for every indexed JD requirement.

## Models and planned settings

| Model | GPU | Thinking settings |
|---|---|---|
| Qwen 3.8 27B Q4_K_M | R9700 | Off (0), Low (512), Medium (2,048), High (4,096) tokens |
| GPT-OSS-20B GGUF | 9070 XT | Off (0), Low (512), Medium (2,048), High (4,096) tokens |

The study sample is nine pairs from the supplied training shard: three smallest by combined resume/JD length, three median-length, and three longest.  The planned full matrix is 72 pair scores (9 pairs × 2 models × 4 settings).

## Findings to date

### Qwen, thinking off

- In an earlier five-pair comparison against the dataset labels, Qwen agreed on **4/5 pairs (80%)**.  This is directional only; five examples are far too few to establish accuracy.
- The warm cached pipeline mean was **55.6 seconds per pair** in that mixed-size sample.
- On the longest validated pair, the optimized cached pipeline took **34.3 seconds**, versus **126.6 seconds** for the original single-pass prompt: a **72.9% reduction** (3.69× faster).

### Qwen, medium thinking

- A medium-thinking run (2,048-token budget) on a median-size pair took **112.5 seconds** for the pair-scoring call, materially slower than reasoning off.
- It changed the result from **8 / Good Fit** with thinking off to **4 / Potential Fit**.
- This single example shows that added thinking can materially change a judgement, but it does **not** demonstrate an accuracy improvement.  The result is less aligned with the off-setting result and requires a labelled comparison over the complete sample.

### Qwen, low thinking

- The low-thinking configuration (512-token budget) completed all nine selected benchmark pairs with schema-valid output.
- Label-accuracy aggregation is still pending; therefore no accuracy claim should be made yet.

### GPT-OSS

- The Off and Low runs exposed reliability failures on long, highly structured outputs: omitted requirement indices, concatenated/malformed JSON, and one service-completion hang on a long Low-thinking case.
- Those failures were rejected by validation rather than silently persisted as candidate scores.  This protects output quality but means the partial benchmark cannot fairly compare GPT-OSS accuracy yet.
- GPT-OSS output throughput was also slower on the observed long cases, with service telemetry indicating possible memory spill.

## Interpretation

Current evidence does **not** support a simple relationship where more thinking yields more accurate fit scores.  Resume screening here is mostly a constrained evidence-extraction and rubric-following task.  Extra thinking can increase latency and may compete with the structured evidence/checklist output budget, reducing format reliability.

The current working production policy is:

1. Use **thinking off** as the bulk-scoring default.
2. Use **Low thinking (512 tokens)** only as a review tier for borderline outcomes (for example, score 4–6), subject to final labelled validation.
3. Do not use Medium or High thinking for bulk scoring unless the completed benchmark shows a clear, reproducible labelled-accuracy gain that justifies the extra latency.

## Limitations and next experiment

- The current evidence is incomplete: Qwen Low is complete, Qwen Off has one validation-rejected long result, and GPT-OSS is incomplete.
- Dataset labels are useful as a benchmark target but should be supplemented with human hiring-manager review, particularly for borderline scores.
- Before resuming the 72-run matrix, harden GPT-OSS long-response handling: split or compact very large evidence payloads, enforce one JSON object only, and fix the dual-GPU service completion issue.
- After hardening, calculate per-setting label agreement, confusion matrix, score error, valid-output rate, latency percentiles, and throughput on the same fixed nine-pair sample.  Only then select a production thinking level.

## Reproducibility artifacts

- Benchmark runner: `scripts/benchmark_thinking_levels.py`
- Selected sample and partial results: `outputs/thinking-level-benchmark/`
- Pipeline and prompts: `src/candidate_fit/optimized.py`
- Runtime optimization history: `docs/optimization-log.md`

