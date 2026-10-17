# JD Rubric Extractor v2.3 — targeted semantic refinement

This is a **JD extraction prompt**, not the resume-matching / candidate-scoring prompt. It operates **once per unique JD** and publishes a validated rubric for reuse. Developed from the v2.2 prompt with three narrow modifications; **no model has yet been re-run with v2.3**, so accuracy and token-speed gains are not established.

## Files

- `JD_Rubric_Extractor_v2.3_FULL.txt`: full drop-in system prompt, including complete schema.
- `JD_Rubric_Extractor_v2.3_SCHEMA_MODE.txt`: use only when supplying `jd_rubric.schema.json` as a strict structured-output response format.
- `jd_rubric.schema.json`: **unchanged from v2.2** (byte-identical).
- `rubric_validator.py`: v2.2 validator plus narrow new coverage/policy checks.
- `compare_runs.py`: compare new and baseline CSV runs, same 12 JDs.
- `tests/test_targeted_v23.py`: nine regression tests including good and deliberately incorrect rubrics.
- `tests/fixtures/results_v22.csv`: the supplied v2.2 outputs used as the regression baseline; **not** v2.3 generation.
- `v22_outputs_under_v23_validator.json`: diagnostic audit against existing v2.2 outputs.

## Precisely what changed

| Issue in v2.2 outputs | v2.3 prompt behavior | Validator behavior |
|---|---|---|
| JD-000002: named Salesforce / ServiceNow SaaS examples collapsed to generic `SaaS Development` | Treat category examples with "etc." as two named example skills, one **open** `any_of_group` with `min_required=1`; do not enforce both tools | High-precision pattern checks expected members and matching open group |
| JD-008861, JD-016936: explicitly named `problem solving` missing from `otherskills` | Keep each distinct listed competency, even broad ones, as preferred/supporting unless explicitly required | Check a narrow known list of recognizable `otherskills`; does not require vague attitudes |
| JD-000002: explicit visa sponsorship / relocation conditions conflated with job-fit hard gates | Preserve verbatim conditions in `requirements`, respect non-negotiable wording, but **do not treat them as technical/role-fit skills** | Detect known explicit policies, flag uncertain policy encoding; helper separates logistical conditions from scoring requirements |

**Important:** `hard_gate_explicit` in the existing schema describes whether the JD expresses a non-negotiable condition. It does **not** authorize automatic rejection or technical-score penalties. A non-sponsorship policy does not establish anything about a particular candidate's immigration status. Handle employment logistics separately, with policy/legal review where necessary.

## Compatibility

The schema is unchanged. Replacing the FULL prompt is sufficient **to run a new test** if your existing code already consumes the flat v2.2 rubric structure. The stricter validator is **optional for an isolated A/B test** and strongly recommended before caching. Do not overwrite previously validated v2.2 cache records: use a prompt/versioned hash and compare outputs.

## Run tests / validate one extraction

```bash
python -m pip install jsonschema
python -m unittest discover -s tests -v
python rubric_validator.py --jd jd.txt --rubric rubric.json --report validation.json
```

## Compare same twelve JDs

Generate `results_v23.csv` with the unchanged Qwen model, quantization, temperature, context window, output-token limit, and inference backend. Then run:

```bash
python compare_runs.py --baseline tests/fixtures/results_v22.csv --challenger results_v23.csv --report comparison.json
```

Evaluate *both* files under the v2.3 validator so numbers are comparable. The old twelve produce **9/12 cache-valid** with the stricter checks (the prior v2.2 validator accepted 12/12). This is a **stricter audit of old output, not reduced accuracy on a new run**.

The validator intentionally only catches a **subset** of semantic omissions. It is not a proof of complete coverage, legal compliance, or recruiter accuracy. Review full JD evidence manually for an additional set of 50–100 JDs before promoting to production.

## Cache and scoring separation

Version cache keys by JD text, model, extractor prompt, schema, and relevant runtime settings. Only publish rubrics that pass validation and any required human review. For downstream use:

```python
from rubric_validator import partition_requirements_for_scoring
fit_requirements, logistical_conditions = partition_requirements_for_scoring(rubric)
# Pass fit_requirements into technical/role-fit scoring.
# Route logistical_conditions to a separate, human-reviewed workflow.
```

You must **actually integrate** this helper with the scoring pipeline; merely adding it to a validator does not prevent the original scorer from using the entire unfiltered rubric. For candidates, retain uncertainty and route hiring decisions to human review.

## Suggested acceptance gates

- 12/12 parse, strict schema, genuine source quotations, cache validation (or justified human-reviewed exceptions).
- JD-000002: Salesforce and ServiceNow as examples of one open enterprise SaaS alternative; sponsorship and relocation conditions preserved without inflating technical fit.
- JD-008861/016936: explicitly listed problem solving retained at appropriate low priority.
- JD-000016: degree enrollment not turned into completed-degree minimum.
- JD-000314: informal account/tax hiring need classified `partial`, without invented qualifications.
- Retain CI/CD duty-linked skills, all explicit mustskills/goodskills, and material responsibilities.
- Output tokens/latency measured; no assumption of improvement without model runs.
