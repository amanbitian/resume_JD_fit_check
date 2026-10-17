# Runtime optimization log

Measured on 2026-10-11 through the registered `dual_gpu_setup` `/api/chat` endpoint. Qwen is
`Qwen3.8-27B-Q4_K_M` on `r9700`; GPT-OSS is `gpt-oss-20b-GGUF` on `9070xt`. Both production
deployments use a 16,384-token context and reasoning disabled.

## Changes evaluated

| Step | Change | Result | Decision |
|---|---|---|---|
| 1 | Full single-pass baseline | Longest pair: 126.609 s, 7,681 input and 3,512 output tokens, score 1 | Baseline |
| 2 | Compact JSON key names in the same single pass | Longest pair: 142.531 s and 4,002 output tokens | Rejected; the model produced more evidence and was slower |
| 3 | Analyze each distinct JD and resume once, then cache it | Dataset has 280 unique JDs and 642 unique resumes across 40,009 pairs in this shard | Accepted; reduces repeated extraction calls from 80,018 to 922 |
| 4 | Pair scorer receives indexed cached evidence and returns indexed matches | Warm longest pair: 34.344 s, 2,472 input and 932 output tokens, same score 1 | Accepted; 72.9% lower latency, about 3.69x faster |
| 5 | Append-only analysis cache and prediction JSONL | Ctrl+C-safe; successful analyses and pairs are reused on restart | Accepted |
| 6 | Strict index/enum/range validation plus one bounded structured-output retry | Recovered a malformed pair output; invalid domains/statuses cannot be silently saved | Accepted |
| 7 | Qwen medium thinking budget (2,048) | Median pair valid call 112.515 s vs 56.579 s off; two attempts; cold pipeline 467.850 s; score changed 8 to 4 against a `Good Fit` label | Rejected |
| 8 | 4,096 vs 6,144 output ceiling | Post-fix small pair completed, but 24.188 s vs an earlier 21.813 s at 6,144 is normal run variance, not a speedup | Keep 6,144 production default for output safety |
| 9 | Qwen server slots: 1 to 2 at the same 16K context | Two warm identical pairs: 35.838 s wall-clock concurrent vs 48.376 s serial baseline; each request became ~26% slower, but aggregate throughput rose 1.35x; no VRAM spill | Promising, but needs cache-safe parallel scoring support before production use |

## Requested five-pair smoke set

One longest pair, one median-sized pair, and three below-average pairs were run warm through the
optimized Qwen pipeline.

| Row | Size group | Resume chars | JD chars | Gold | Score/label | Pair latency | Output tokens |
|---:|---|---:|---:|---|---|---:|---:|
| 3,218 | longest | 23,309 | 7,669 | No Fit | 1 / No Fit | 34.344 s | 932 |
| 24,695 | median | 5,369 | 2,403 | Good Fit | 8 / Good Fit | 56.579 s | 1,694 |
| 33,659 | small | 2,987 | 806 | No Fit | 1 / No Fit | 21.813 s | 670 |
| 25,111 | small | 4,560 | 1,344 | Potential Fit | 2 / No Fit | 91.657 s | 2,825 |
| 27,639 | small | 5,054 | 2,032 | No Fit | 1 / No Fit | 73.515 s | 2,253 |

Mean model latency was 55.582 seconds. Label agreement was 4/5 (80%), macro-F1 0.619, and mean
ordinal error 0.2. This sample is deliberately useful for smoke testing but is too small to claim
production accuracy or subgroup fairness. The row 25,111 disagreement must be included in a
larger calibration set before operational use.

Output length, especially the number of skills and requirements that must be audited, is a better
latency predictor than raw resume/JD character length. Row 25,111 had 28 skills and therefore took
longer than the nominally longest input.

## Final post-fix verification

The stricter canonical-domain build was tested on row 33,659 with a 4,096-token cap:

- cold cache: 106.836 s total (JD extraction 48.297 s, resume extraction 22.469 s, pair 23.328 s);
- warm cache: 30.244 s total, 24.188 s model latency;
- score stayed 1 / No Fit, matching the dataset label;
- natural `stop`, one attempt, 674 output tokens, both cache hits on the warm repeat;
- all emitted functional and industry domains passed the controlled taxonomy validator.

The production recommendation remains reasoning off, 16,384 context, 6,144 maximum output tokens,
and `score-optimized` with one worker per model lane.

## Concurrency experiment

The isolated two-slot Qwen profile is saved as
[`config/qwen-concurrency-benchmark.toml`](../config/qwen-concurrency-benchmark.toml). It was used
only for a controlled test and the service was restored to the normal two-GPU, one-slot-per-lane
profile afterward.

The experiment proves that two active sequences can fit in the R9700 at 16K context without
reported VRAM spill. It improves aggregate throughput, not individual request latency. The current
`score-optimized` command deliberately remains single-worker because its shared append-only cache
needs a bounded, cache-safe concurrent scheduler before this setting can be used for a full run.
