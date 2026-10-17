import json

from candidate_fit.metrics import evaluate_jsonl


def test_metrics_macro_and_ordinal_error(tmp_path):
    rows = [
        {"status": "ok", "gold_label": "No Fit", "fit_label": "No Fit"},
        {"status": "ok", "gold_label": "Potential Fit", "fit_label": "Good Fit"},
        {"status": "ok", "gold_label": "Good Fit", "fit_label": "Good Fit"},
        {"status": "error", "gold_label": "Good Fit"},
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = evaluate_jsonl(path)
    assert result["evaluated"] == 3
    assert result["ignored"] == 1
    assert result["accuracy"] == 0.666667
    assert result["mean_ordinal_error"] == 0.333333


def test_metrics_do_not_mix_scoring_fingerprints(tmp_path):
    rows = [
        {
            "status": "ok",
            "pair_id": "pair-1",
            "scoring_fingerprint": "run-a",
            "gold_label": "Good Fit",
            "fit_label": "Good Fit",
            "model": "model-a",
        },
        {
            "status": "ok",
            "pair_id": "pair-1",
            "scoring_fingerprint": "run-b",
            "gold_label": "Good Fit",
            "fit_label": "No Fit",
            "model": "model-b",
        },
    ]
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = evaluate_jsonl(path)

    assert result["mixed_scoring_fingerprints"] is True
    assert result["runs"]["run-a"]["accuracy"] == 1.0
    assert result["runs"]["run-b"]["accuracy"] == 0.0

