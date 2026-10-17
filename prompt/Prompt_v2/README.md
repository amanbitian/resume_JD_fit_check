# JD–Resume Compact Scoring Kit (v4.1)

**Exactly three files:** `prompt.txt`, `scoring.py`, `README.md`.

This is a lightweight replacement **experiment** for the older verbose extraction → matching → scoring process. It uses **one LLM call per JD–resume pair**, then Python validates the JSON and calculates a **provisional 1–10 relevance score**. It does **not** automatically hire or reject anyone. The result supports human review.

## 1. What changed and why

| Gap in older prompts | Change in v4.1 |
|---|---|
| Repeated skill judgments disagree (`sa` vs `rc`) | One canonical check per JD requirement |
| Model invents final score/does inconsistent arithmetic | No LLM final score; Python computes dimensions and caps |
| Verbose explanations waste output tokens | Short literal JD/resume evidence quotes + <=35-word reason |
| Cloud cost from 2,000–8,000 output tokens | Aim for compact response; **actual tokens depend on JD size and reasoning settings** |
| Hallucinated evidence | Python checks quoted substrings against source JD and resume |
| Missing evidence treated as failed parsing | Parse/validation failure returns an error, **never** score 1 |
| Uncertain work years mistakenly treated as zero | Neutral provisional years dimension (5/10), cap and review |
| Implied/inferred must-haves cause arbitrary caps | Only **explicitly mandatory** checks create qualification caps |
| Overqualification penalty | Prohibited; judge fit against requirements |
| Skills in an `A or B` JD clause counted as two failures | One requirement checks the complete alternative clause |

**Tradeoff:** One-call assessment saves tokens, but no deterministic program can prove the model captured *every* JD requirement from free text. For high-stakes use, compare the `checks` against a frozen JD rubric or human-reviewed requirement inventory. The source-quote check validates provenance, **not** whether the interpretation is correct.

## 2. Use the prompt

Place `prompt.txt` in the **system** message. Place the actual JD and resume in the **user** message, preferably as a JSON object:

```json
{"job_description":"<JD text>","resume":"<resume text>"}
```

Keep requirements exactly as written in the JD. Send the **same** system prompt to each model during A/B evaluation. Temperature `0` is a useful reproducibility starting point, but not a guarantee of deterministic output. Ask the API for `json_object` mode where available.

Example model JSON (demonstration only):

```json
{
  "status": "scored",
  "checks": [
    {"id":"R1","jd":"Python","kind":"skill","priority":"mandatory","status":"met","quote":"Python developer"},
    {"id":"R2","jd":"AWS or Azure","kind":"skill","priority":"mandatory","status":"not_evidenced","quote":null},
    {"id":"R3","jd":"Git","kind":"skill","priority":"preferred","status":"not_evidenced","quote":null},
    {"id":"R4","jd":"5 years of relevant experience","kind":"experience","priority":"mandatory","status":"partial","quote":"Software Engineer, 2022-2025"}
  ],
  "work": 6,
  "role": 6,
  "years": {"required":5,"relevant":3,"certain":true},
  "reason": "Python experience is demonstrated, but the resume lacks evidence of the required cloud platform and full experience threshold.",
  "review": true
}
```

For the example to pass validation, the JD must literally contain `Python`, `AWS or Azure`, `Git`, and `5 years of relevant experience`, and the resume must literally contain both cited quotes. Evidence snippets **must not be paraphrased**. Here, `years.certain=true` is illustrative and should only be used for reliable, verified non-overlapping relevant employment periods.

## 3. Running the code

Requirements: **Python 3.10+; no pip packages**. Run in the directory containing these three files.

### Option A — Score existing model JSON (offline; no API request)

Create `jd.txt`, `resume.txt`, `assessment.json` as UTF-8 text files (they are user inputs, **not** extra kit files).

```bash
python scoring.py --jd jd.txt --resume resume.txt --assessment assessment.json --output result.json
```

The command prints JSON and optionally saves `result.json`.

### Option B — Call a local LM Studio model

Start LM Studio's OpenAI-compatible server and load a model. Use its **exact** API model ID:

```bash
python scoring.py --jd jd.txt --resume resume.txt \
  --model 'YOUR_LM_STUDIO_MODEL_ID' \
  --api-base http://localhost:1234/v1 \
  --max-tokens 2048 --json-mode --output result.json
```

If your server does not support JSON mode, remove `--json-mode`. You may need a larger output budget for JDs with many requirements. The script **rejects** any output where `finish_reason=length`; it never scores truncated text.

### Option C — Call an OpenAI-compatible cloud API

```bash
export OPENAI_API_KEY='YOUR_API_KEY'
python scoring.py --jd jd.txt --resume resume.txt \
  --model 'YOUR_AVAILABLE_MODEL_ID' \
  --api-base 'https://api.openai.com/v1' \
  --max-tokens 2048 --json-mode --output result.json
```

On Windows PowerShell, use `$env:OPENAI_API_KEY="..."`. This sends the complete JD and resume to the external provider; review personal-data handling and provider contractual terms first. The script is **sequential** and designed for evaluation, **not** a 159K-pair batch uploader. For bulk jobs, integrate `calculate()` and `validate()` into your existing orchestration and use your provider's supported Batch mechanism.

## 4. Scoring calculation

The model returns **requirement checks**, work/role ratings, and relevant-years evidence. Python derives the final score.

| Dimension | Weight | Source |
|---|---:|---|
| Skills | 40% | Mean of all `kind=skill` requirement statuses |
| Work responsibilities | 30% | Mean of `kind=responsibility` statuses, if any; otherwise model `work` rating |
| Role alignment | 15% | Model `role` rating |
| Years | 10% | Explicit JD minimum vs reliably evidenced relevant years |
| Education | 5% | Mean of `kind=education` statuses if any |

For skill, responsibility, and education requirements, status values are: **met=10, partial=5, not_evidenced=0, uncertain=5**. These represent a **provisional ranking rubric**, not probabilities. Explicit `mandatory` requirements have weight 1.0, `preferred` 0.4, and `unclear` 0.4 within a dimension. `uncertain` forces reviewer attention, even though its interim scoring value is neutral.

If a dimension is genuinely **not applicable** (for example no degree requirement), its weight is **redistributed proportionally** among applicable dimensions. A required but uncertain years minimum is not removed: the interim years dimension is 5 and the final score is capped at 8 with review.

For a reliable `years.relevant / years.required` ratio:

| Ratio | Years dimension |
|---|---:|
| >=1.0 | 10 |
| >=0.8 and <1.0 | 7 |
| >=0.6 and <0.8 | 5 |
| >=0.4 and <0.6 | 3 |
| >0 and <0.4 | 1 |
| 0 | 0 |

`raw = (sum of dimension score × applicable weight) / (sum of applicable weights)`.

`base = round_half_down(raw)`, constrained to **1–10**. Thus 6.50 rounds to 6, and 6.51 rounds to 7. This rounding convention is retained from the previous scoring package for comparability.

### Mandatory-requirement caps (policy defaults, not statistically calibrated)

- One explicitly mandatory skill, responsibility, or education requirement `not_evidenced` → cap **6**.
- At least two explicitly mandatory substantive requirements `not_evidenced` → cap **4**.
- An explicitly mandatory requirement `partial` or `uncertain` → cap **8**.
- Numeric relevant experience, confidently below 50% of minimum → cap **4**; below 80% → cap **6**.
- Numeric relevant experience uncertain → cap **8** and require review.
- Work authorization, location, language, and other conditions are **reviewed separately**, not auto-rejected or score-capped by this calculator.
- An `experience` check with a numeric years minimum does **not** trigger an additional requirement cap; the years policy handles it once.

`final_score = min(base, all applicable caps)`. Fit bands: 1–2 `no_fit`; 3–4 `weak_fit`; 5–6 `potential_fit`; 7–8 `good_fit`; 9–10 `strong_fit`.

**Important:** A high score with `review=true` is **not** a verified eligibility result. A missing mention means *not evidenced*, not *the candidate cannot do the work*. All final selection/rejection decisions remain with human reviewers.

### Worked example

Assume a JD has two mandatory skills; one is `met` and the other `not_evidenced`. There are no degree, numeric-year, or explicit responsibility requirements. The model assigns `work=8`, `role=7`.

- Skills = (10 + 0)/2 = **5**.
- Applicable weights = 0.40+0.30+0.15 = **0.85**.
- Raw = (0.40×5+0.30×8+0.15×7)/0.85 = **6.41**.
- Base score = **6**. Missing mandatory skill caps at 6 → **6/10 (`potential_fit`)**, flagged for human review.

## 5. Integration into your existing evaluator

Your previous code already has model calls, parser validation and result storage. Add this as a **new** `v4-compact` mode; do not overwrite the existing `v2.1`, `v3`, or earlier `v4` modes.

1. In your model client, use `prompt.txt` as the system prompt; pass JD and resume as a JSON user message.
2. Parse and store the **raw model JSON** and API `usage` with model name, model quantization, prompt version, sample ID, and timestamp. Do not log sensitive resume text indiscriminately.
3. Call `from scoring import calculate` and then `result = calculate(assessment, jd_text, resume_text)`.
4. Save `result['score']` as the calculated score and `result['fit']` as the fit band; retain `dimensions`, `caps`, `review`, and `reason` for audit.
5. A validation exception, unreadable document, or truncated response is **unscorable**; queue it for repair or review. Never silently assign score=1.
6. For 159K pairs, cache deduplicated JD/resume parsing or canonical source text in your own system, and group calls by repeated JD to improve prefix-cache hit rates where the provider supports them.
7. Keep a frozen human-labelled benchmark containing good, bad, and borderline matches. Compare qualified-candidate recall, precision, false-negative rate, evidence correctness, coverage of explicit JD must-haves, score agreement, schema pass rate, latency, and **total billed** tokens against the existing prompt. Include reasoning tokens if billed.

**Do not compare v4.1 scores directly to previous v2.1/v3 scores without calibration:** their dimensions and policies are not identical.

## 6. Known limitations and safeguards

- **Coverage cannot be fully validated from raw JD text.** A missing extracted requirement is invisible to the Python check. For consequential use, establish a frozen, human-reviewed JD requirement list before evaluating the candidate pool; that increases robustness at modest one-time cost per JD.
- **Requirement priority is inferred by the same model in one call.** The script does not independently verify the meaning of “mandatory”; a mistaken classification may activate score caps. Require JD rubric review, especially for hard gates.
- **Dates are not recomputed from an employment timeline.** `years.certain=true` is the model's assertion, not proof; the score is always marked `provisional=true`. For production-grade experience scoring, use the overlap-aware date calculator from the original pipeline.
- **Quote validation is limited.** A verbatim quote proves the words exist but not that they satisfy the qualification or are not negated elsewhere.
- **Work and role remain subjective model ratings.** Their values require calibration against human reviewers.
- **No automatic rejections.** Treat the scores as ranking signals subject to review, fair-use and applicable employment law.
- **No API retries or batch processing included.** These are intentionally out of scope for the three-file package; production orchestration should implement bounded retries, idempotent IDs, quotas, safety controls and structured-output support.
- **Prompt caching is provider-dependent.** Stable system instructions and JD-first input ordering can help; no cache hit is guaranteed. Actual cloud costs require current price and usage measurement.

## 7. Recommended next test

Run 100–300 labelled pairs including positives and borderline cases with Qwen and your chosen cloud model. Compare `v4-compact` to your earlier evaluator under identical documents; verify every missed mandatory requirement manually. Do not scale to 159K until performance and error rate are acceptable. This kit has code-level tests but **no claim of measured model accuracy improvement**.
