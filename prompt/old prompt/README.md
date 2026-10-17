# JD–Resume Relevance Pipeline — optimized v3 (reference implementation)

## What changed

This package improves the original `prompt_v2.zip`. It provides three revised system prompts, all three user templates, a deterministic score calculator, and an evidence-quote auditing helper. It is **not a bit-for-bit drop-in**: JD schema adds `hard_gate_explicit`, `priority_origin`, `source_quote`, and alternative-group metadata; resume schema adds exact evidence quote fields; the recommended step-3 prompt now emits matches and evidence **without** calculating score fields.

All thresholds and weights are *provisional policy defaults*, **not empirically optimized or validated**. The 15 included automated tests confirm specific code invariants, not real-world accuracy.

## Recommended pipeline

```text
RAW RESUME  ────► (1) Resume extractor ──► structured profile ───┐
                 cache by resume version + prompt version          │
                                                               (3) Match-only model
RAW JD      ────► (2) JD extractor ──────► frozen rubric ──────────┤  └─► evidence/status
                 cache by JD version + prompt version              │         │
                                                                   └─────────▼
                                                          Strict validation + quotes
                                                                       │
                                                                       ▼
                                                    score_calculator.recompute(...)
                                                                       │
                                                                       ▼
                                                    ranked score OR unscorable
                                                   + review reasons + audit trail
```

| Step | System file | Action |
|---|---|---|
| Resume extraction | `1_resume_extraction_system_prompt.txt` | Extract factual evidence; do not infer fit or missing years. |
| JD extraction | `2_jd_extraction_system_prompt.txt` | Freeze structured rubric and distinguish hard gates from inference. |
| Matching (RECOMMENDED) | `3_relevancy_scoring_system_prompt.txt` | Match JD items against evidence; **do not** perform arithmetic. |
| Matching (legacy-style alternative) | `3_relevancy_scoring_system_prompt_legacy_full.txt` | Full output including model arithmetic for migration/debugging only; application must override it. |
| Deterministic calculation | `score_calculator.py` | Validate item coverage, recalculate dates and score, apply explicit caps. |
| Provenance audit | `evidence_audit.py` | Map verbatim evidence excerpts back to offsets in the original text. |

## Minimum integration code

```python
from score_calculator import recompute, merge_assessment, compare
from evidence_audit import annotate_quotes

# resume_json = model step 1 response (parsed JSON)
# jd_json = model step 2 response (parsed JSON)
# match_json = model step 3 response (parsed JSON, evidence only)

resume_json, resume_issues = annotate_quotes(resume_json, raw_resume_text, 'resume')
jd_json, jd_issues = annotate_quotes(jd_json, raw_jd_text, 'jd')

# Check the issues and send ambiguous/unverified sources to review.
calculation = recompute(jd_json, resume_json, match_json, reference_date='2026-10-17')
final_result = merge_assessment(match_json, calculation)

if final_result['assessment_status'] == 'unscorable':
    # Never add to low-relevance reject cohort.
    route_to_manual_review(final_result)
elif final_result['needs_human_review'] or resume_issues or jd_issues:
    route_to_manual_review(final_result)
else:
    add_to_priority_queue(final_result)  # humans make interview decisions
```

`route_to_manual_review` and `add_to_priority_queue` are **application pseudocode**, not defined functions. Replace them with your workflow. The reference date must be the one used in both extractions; cache keys must include it if ongoing work can change the experience calculation.

For model structured output, construct genuine JSON Schema / Pydantic schemas from the output specifications and enable the model provider's JSON schema feature. The angle brackets in system prompt examples are explanatory placeholders, *not* valid JSON schema. Validate every model response before calling the calculator.

## Behavior and migration changes

1. **Unscorable**: invalid resume, invalid JD, or several roles returns `final_score: null`, `fit_band: "unscorable"`, `assessment_status: "unscorable"`. Do not convert null into 1/10 in SQL, analytics or ranking.
2. **Hard caps**: use `hard_gate_explicit == true` for caps; a generic Requirements heading no longer unilaterally triggers a strict 3/10 cap. A non-hard mandatory gap still reduces the base score and triggers review.
3. **Input dates**: `recompute(..., reference_date="YYYY-MM-DD")` is the canonical calculator call. Year-only or missing dates and unknown part-time FTE produce an unknown eligibility ratio, *not zero years*. Score without an experience dimension and route for review.
4. **No seniority-overqualification penalty**: more senior experience no longer reduces role-alignment points automatically, though scope discrepancies may require review.
5. **Projects**: project-only skill evidence is distinguishable from professional production experience; explicit hard, critical mid+ requirements are capped at partial.
6. **Quote provenance**: extraction adds evidence_quote/source_quote; `evidence_audit.py` can attach offsets or report missing/ambiguous spans. One missing quote is not proof the whole resume is fabricated.
7. **Open alternatives**: `any_of_groups[].open_category=true` allows JD examples like "Power BI or Tableau or similar BI tool" to cover a suitably evidenced unlisted tool. This remains a semantic judgment requiring review.
8. **Input completeness**: for valid assessments all S#, G#, D#, R#, E# must have exactly one match. Missing/duplicate matches raise ValueError instead of silently becoming missing evidence.
9. **Preserved weights**: Skills 35%, Work 25%, Role 15%, Domain 10%, Years 10%, Education 5% remain unchanged baseline policy values; this is NOT a data-fitted optimization.
10. **Not every gap is fixed in code**: see GAP_AUDIT.md for outstanding risks, especially incomplete skill-duration evidence, semantic equivalence, repeated evidence weighting, legal/process controls and population-level benchmarking.

## Run tests

```bash
cd prompt_v3_optimized
python -m unittest discover -s tests -v
```

15 scenario tests cover valid fit, unscorable input, 0.80 rounding boundary, uncertain dates, hard-vs-inferred requirements, project-only capability, internship counting, overlap, missing matches, postprocessing, and open-ended reference dates. These are unit tests of implementation behavior, not benchmark evaluations.

## Preproduction evaluation

1. Version all three prompts, source parsers, JD taxonomy and calculator separately. Freeze each JD rubric before comparing candidates. Keep the reference date for each scored pair.
2. Sample a *role-stratified* human-labeled gold set, including cross-domain candidates, freshers, career changers, internal promotions, skills stated only in lists, year-only dates, tool alternatives, invalid documents, and borderline fits.
3. Use at least two blinded independent reviewers for a subset; reconcile disagreements using a written criterion. Evaluate reviewer agreement with weighted kappa. Never treat an adjudicated label as error-free.
4. For each JD, evaluate recall for truly qualified candidates, false-negative rate, precision@K, ranking correlation, and flip rate around 6/7. Also track abstention, review volume, quote verification, parse validity, and group-equivalence overrides.
5. Do not use a global 7+ acceptance rule until per-role calibration is measured. Implement an appeal/correction route and human decision authority.
6. Keep raw documents access-controlled, minimize identifiers, and audit evidence/model/policy versions. Check jurisdiction-specific hiring compliance before deployment.

## Known limitations

- Model-determined statuses, domain equivalences and relevance labels are not deterministic. Human-validated skill ontology is still needed for stronger consistency.
- Resume spans do not establish actual day-by-day tool usage. `approx_months_used` is uncertain; do not treat it as audited proficiency tenure.
- The scoring weights and numeric caps have not been fitted to a labeled data set.
- Different roles may need different feature weights, minimum thresholds, and permitted equivalents.
- Requirement categories `other`, some language/logistics details, and some preferred-only conditions need explicit organizational policy outside this reference numeric scorer.
- JSON schemas and a production-grade source normalizer/validator need to be generated before running at scale.
- The scoring policy computes fit priority, not a probability of success or a hiring eligibility determination.
