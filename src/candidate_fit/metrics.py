from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABELS = ("No Fit", "Potential Fit", "Good Fit")


def _calculate_metrics(rows: list[tuple[str, str]]) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
    score_error_sum = 0.0
    for gold, pred in rows:
        confusion[gold][pred] += 1
        score_error_sum += abs(LABELS.index(gold) - LABELS.index(pred))
    valid = len(rows)

    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in LABELS if gold != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
    correct = sum(confusion[label][label] for label in LABELS)
    macro_f1 = sum(float(per_class[label]["f1"]) for label in LABELS) / len(LABELS)
    return {
        "evaluated": valid,
        "accuracy": round(correct / valid, 6) if valid else 0.0,
        "macro_f1": round(macro_f1, 6),
        "mean_ordinal_error": round(score_error_sum / valid, 6) if valid else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def evaluate_jsonl(path: str | Path) -> dict[str, Any]:
    groups: dict[str, dict[str, tuple[str, str]]] = {}
    group_models: dict[str, set[str]] = {}
    invalid_json = 0
    status_errors = 0
    unevaluable = 0
    duplicate_success_rows = 0
    total_rows = 0

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if row.get("status") != "ok":
                status_errors += 1
                continue
            gold, pred = row.get("gold_label"), row.get("fit_label")
            if gold not in LABELS or pred not in LABELS:
                unevaluable += 1
                continue
            fingerprint = str(row.get("scoring_fingerprint") or "legacy_or_unspecified")
            pair_id = str(row.get("pair_id") or f"line:{line_number}")
            bucket = groups.setdefault(fingerprint, {})
            if pair_id in bucket:
                duplicate_success_rows += 1
            bucket[pair_id] = (gold, pred)
            model = row.get("model")
            if model:
                group_models.setdefault(fingerprint, set()).add(str(model))

    ignored = invalid_json + status_errors + unevaluable
    run_metrics = {
        fingerprint: {
            **_calculate_metrics(list(rows.values())),
            "scoring_fingerprint": fingerprint,
            "models": sorted(group_models.get(fingerprint, set())),
        }
        for fingerprint, rows in groups.items()
    }
    audit = {
        "total_rows": total_rows,
        "ignored": ignored,
        "invalid_json": invalid_json,
        "status_errors": status_errors,
        "unevaluable": unevaluable,
        "duplicate_success_rows": duplicate_success_rows,
    }
    if len(run_metrics) == 1:
        only = next(iter(run_metrics.values()))
        return {**only, **audit, "mixed_scoring_fingerprints": False}
    if not run_metrics:
        return {
            **_calculate_metrics([]),
            **audit,
            "mixed_scoring_fingerprints": False,
            "scoring_fingerprint": None,
            "models": [],
        }
    return {
        **audit,
        "mixed_scoring_fingerprints": True,
        "warning": "Metrics are separated because this file contains incompatible scoring fingerprints.",
        "runs": run_metrics,
    }

