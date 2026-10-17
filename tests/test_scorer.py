import json

from candidate_fit.client import Completion
from candidate_fit.dataset import PairRecord
from candidate_fit.scorer import ScoreOptions, score_dataset


class FakeClient:
    model = "fake"
    base_url = "http://local"

    def complete(self, _prompt, **_kwargs):
        return Completion(
            content=json.dumps(
                {
                    "fit_score": 7,
                    "fit_label": "Good Fit",
                    "confidence": 0.8,
                    "dimension_scores": {
                        "skills": 8,
                        "years_experience": 7,
                        "education": 7,
                        "work_experience": 8,
                    },
                    "requirement_checks": [
                        {
                            "category": "skill",
                            "requirement": "Python",
                            "priority": "mandatory",
                            "status": "met",
                            "resume_evidence": "Python developer",
                        }
                    ],
                    "unmet_mandatory_requirements": [],
                    "rationale": "Relevant evidence.",
                }
            ),
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_seconds=0.1,
        )


def record(pair_id="data.parquet:0"):
    return PairRecord(pair_id, "data.parquet", 0, "Python developer", "Need Python", "Good Fit", {})


def test_score_and_resume(tmp_path):
    output = tmp_path / "scores.jsonl"
    first = score_dataset([record()], client=FakeClient(), output_path=output, options=ScoreOptions())
    second = score_dataset([record()], client=FakeClient(), output_path=output, options=ScoreOptions())
    assert first == {"scored": 1, "errors": 0, "skipped": 0, "aborted": 0, "interrupted": 0}
    assert second == {"scored": 0, "errors": 0, "skipped": 1, "aborted": 0, "interrupted": 0}
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["fit_score"] == 7
    assert row["prompt_hash"]
    assert row["scoring_fingerprint"]
    assert "resume" not in row and "job_description" not in row

