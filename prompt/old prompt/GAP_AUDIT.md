# Detailed Gap Audit: original v2 → optimized v3

**Scope:** reviewed the three v2 system prompts, three user prompt templates, `README.md` and `score_calculator.py`. Original model outputs or human-labelled benchmark examples were NOT supplied. The table distinguishes changes implemented in this package from matters that require further work or testing.

| # | Severity | Component | Specific v2 problem | Failure example or mechanism | v3 response | State |
|---|---|---|---|---|---|---|
| 1 | P0 | Invalid inputs | Invalid resume or JD is automatically `1/10 no_fit` | An unreadable PDF becomes a false negative | `unscorable`, score null, manual review | Implemented |
| 2 | P0 | Years | Trust model-reported `years.ratio` rather than recomputing from dates | Incorrect eligibility caps and inflated relevant years | Monthly union calculation with explicit reference date | Implemented (date limits below) |
| 3 | P0 | Rounding | Model ratio rounded 2 decimals before CAP_YEARS threshold | 47 months / 4.9 yr = 0.7993 displayed 0.80, but below cap threshold | Decisions use raw ratio; rounded only for display | Implemented |
| 4 | P0 | Internship | `relevant_years <= professional_years` contradicted entry-level internship inclusion | Intern with zero full-time experience could never earn 0.5 relevant internship years | Internships eligible for entry/intern scoring without conflating with professional months | Implemented |
| 5 | P0 | Inferred hard caps | Heading/inference treated like explicit hard must-have | A merely suggested skill applies a 3/10 cap | `hard_gate_explicit` and `priority_origin`; only explicit caps | Implemented, requires rubric QA |
| 6 | P0 | Source evidence | Scorer could accept matches absent from resume reference IDs | Invented `E19` or missing S# silently changes ranking | Strict coverage and skill ref validation; verbatim source auditing | Partly implemented |
| 7 | P0 | Project-only work | Standalone hobby project can satisfy mid+ core production skill as `met` | One personal PyTorch tutorial meets a hard deployment requirement | Critical explicit mid+ project-only skill downgraded to partial; no years | Implemented |
| 8 | P1 | Equivalence | `pandas` → Python near-exact without demonstration | Library keyword interpreted as broad production engineering | Requires evidenced capability, directional specific_of | Prompt fix, ontology outstanding |
| 9 | P1 | Date quality | Year-only midpoint and undated roles used as exact tenure | Unknown months lead eligibility false negatives | Inexact dates suppress years dimension/cap; review | Implemented; interval uncertainty model still outstanding |
| 10 | P1 | Education | A past end date automatically means a degree was completed | Expected 2024, no awarded degree, treated as graduate | Require positive credential/completion evidence | Prompt fix |
| 11 | P1 | Role type | Default every unqualified job to full-time | Unknown weekly hours counted as full-time-equivalent years | Mark unknown, review unclear FTE | Implemented / prompt fix |
| 12 | P1 | Overqualification | Seniority >2 levels receives automatic minus two points | Suitable senior applicant unnecessarily demoted | Remove automatic overqualification deduction | Implemented |
| 13 | P1 | Requirements categories | Some non-skill and preferred items are not in numeric score | An explicitly preferred shift schedule has no numeric component | Continue capturing every R#, flag gaps, define business policy | Outstanding weighting-policy decision |
| 14 | P1 | Any-of groups | `e.g.` list incorrectly treated as closed set | Looker expertise rejected for `BI tool such as Tableau or Power BI` | Open categories with alternative resume evidence | Implemented with mandatory review |
| 15 | P1 | Skill duration | Counts whole role span as tool experience | Python listed in five-year role treated as five verified years | Warn estimated duration, require verification for duration gates | Partial; dated skill episodes needed |
| 16 | P1 | Repeated evidence | Same duty appears in skill, responsibility, role and domain components | SQL dashboards could reward four dimensions | Keep baseline, recommend measurement of redundancy/correlation | Outstanding calibration |
| 17 | P1 | Reference date | Current date implicit in current-work intervals in calculator | Cached candidate works one additional month unknowingly | Explicit `reference_date` argument | Implemented |
| 18 | P1 | Unsupported score fields | Full scoring prompt asks model to do dates, averages, gates, and status matching | LLM arithmetic drift | New recommended stage3: evidence only, code is score truth | Implemented |
| 19 | P1 | Missing matches | v2 calculator silently defaults missing skill/responsibility to no match | Model truncated JSON lowers candidate scores without error | Fail validation, do not rank incomplete record | Implemented |
| 20 | P1 | Prestige/fairness | No score penalty for prestige specified, but incidental proxy bias remains | Employer names/schools may still leak correlates | Keep nonprestige rules and human review | Outstanding robust bias audit |
| 21 | P2 | Prompt size | Three long full prompts repeat taxonomy; scoring prompt duplicates arithmetic policy | Higher token costs and possibly lower instruction adherence | Evidence-only stage3 is shorter; shared vocab retained to prevent drift | Partial; shared vocabulary/prefix caching |
| 22 | P2 | Output schema | Pseudo-JSON with `<int>`/`<bool>` is not an actual JSON Schema | Model returns a string enum with pipe separators or invalid JSON | README requires native schema enforcement | Not implemented: generate Pydantic/JSON Schema |
| 23 | P2 | Provenance | E#/S# ids without source offsets or citation quotes | Reviewer cannot immediately verify matches | JD and resume quote fields, offset audit helper | Partly implemented (PDF parse quality still matters) |
| 24 | P2 | Scoring confidence | Handcrafted confidence = 1 minus fixed penalties | 0.8 may look like an 80% probability despite no calibration | Explicitly label heuristic; review relevant unknowns | Actual confidence calibration outstanding |
| 25 | P2 | Prompt injection | Delimiters alone do not prevent instructions embedded in raw documents | Malicious résumé text may claim system authority | Strict data/not-instructions rules and injection flags | Partial; host-side escaping and red-team tests needed |
| 26 | P2 | Role type / degree equivalence | Implicit year/education conversions are coarse | One PGDM, M.Tech and professional license may not be job-equivalent | Remove fixed experience equivalence; explicit JD alternatives | Partly implemented; licensed review needed |
| 27 | P2 | Policy management | 35/25/15/10/10/5 are hand-picked and reused across roles | More technical JDs dominate skills; people-management JDs may not | Mark weights as provisional, validate by role family | Outstanding |
| 28 | P2 | Production control | No exact golden set, adjudication process, drift dashboard | Can't distinguish prompt gains from score-distribution shifts | README evaluation plan and unit tests | Benchmark execution outstanding |
| 29 | P2 | Careers/projects | Promotion dates can be unknown; forcing recent title loses older duty context | Several internal roles merged and seniority inflated | Preserve title history and flag ambiguous dates | Prompt fix only |
| 30 | P2 | Audit and versioning | Missing policy schema version, frozen JD hash, reviewer overrides | Scores from different prompt versions silently mixed | Versioning/migration instructions | Application work outstanding |

## Suggested acceptance criteria before deployment

1. 100% parsed JSON schema validity after validation and retry/abstention; never silently replace a missing match with `not_evidenced`.
2. 100% immutable JD rubric per scored cohort (JD ID + version/hash) and same reference date policy.
3. Zero automatic `no_fit` classifications for invalid/unreadable/multiple-document inputs.
4. Human-verified source spans for all explicit hard gate failures, or mandatory review instead of candidate rejection.
5. Evaluate false-negative rate across JD family/skill equivalence/career path slices, with recruiter adjudication and confidence intervals.
6. Measure speed/cost per pair (input tokens, output tokens, retries, JSON validity), not only macro score correlation.
7. Record model ID, prompt version, calculator policy version, schema version, reference date, quote audit, scorer status and human overrides in every assessment.

This package is an implementation draft and risk-reduction reference, not a validated hiring decision engine.
