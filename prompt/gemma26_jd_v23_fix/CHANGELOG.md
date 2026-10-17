# v0.1 — 2026-10-17

- Added isolated constrained decoding for Gemma-4-26B-A4B QAT with llama.cpp/openai JSON response format options.
- Added two-stage schema enforcement smoke test, including invalid-value challenge.
- Added unchanged vs targeted Gemma prompt variants; Qwen prompt is left intact.
- Added full-schema validation and exact `source_quote` checks without output repair.
- Added fail-closed hook for the real existing v2.3 semantic validator.
- Added content-addressed fingerprints to experiment records without touching production cache.
- Added isolated 20-JD benchmarking and regression tests.

**Not included:** Repo-specific in-place edits, inferred fixes to mustskills/goodskills semantics, live model/gateway configuration changes, or claims of improved measured model accuracy. The original project source could not be fetched in this session.
