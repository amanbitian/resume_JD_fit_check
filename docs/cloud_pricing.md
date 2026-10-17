# Cloud batch prices used for estimates

The prices in `src/candidate_fit/cloud_batch.py` (`PRICES_PER_MILLION`) are used only to show an estimate
before a run. Batch rates are 50% of the standard rates for both providers.

| Model id | Input USD / M tokens | Output USD / M tokens | Source | Checked |
|---|---|---|---|---|
| claude-haiku-5-5 | 0.10 | 0.50 | Secondary summaries of Claude Haiku 5.5 pricing (eesel.ai, braindetox.kr), 2026 | not against anthropic.com |
| gpt-5.6-luna | 0.20 | 1.20 | AnotherWrapper GPT-5.6 Luna page, Aug 2026. Other sources list $1 / $6 for the same name | not against openai.com |

Before a billed run, check both pages:
- https://www.anthropic.com/pricing
- https://developers.openai.com/api/docs/pricing

Update `PRICES_PER_MILLION` and this table together.
