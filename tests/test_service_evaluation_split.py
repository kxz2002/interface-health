import pandas as pd

from src.service.labeling.evaluation import evaluation_table


def _rows(case_id, source, label, count, offset=0):
    return [
        {
            "dataset_id": "d",
            "run_id": case_id,
            "case_id": case_id,
            "canonical_service_id": "svc",
            "window_start_ms": (i + offset) * 15000,
            "window_end_ms": (i + offset + 1) * 15000,
            "sample_id": f"{case_id}-{i + offset}",
            "label": label,
            "label_source": source,
            "label_scope": "service_window",
            "label_reasons": [],
            "evidence_ids": [],
        }
        for i in range(count)
    ]


def test_per_case_split_balances_fault_cases_and_limits_normal_case():
    rows = (
        _rows("fault-a", "injection_association", 1, 3)
        + _rows("fault-a", "pre_intervention_reference", 0, 5, 3)
        + _rows("fault-b", "injection_association", 1, 2)
        + _rows("fault-b", "pre_intervention_reference", 0, 2, 2)
        + _rows("normal", "normal_run_reference", 0, 20)
    )
    labels = pd.DataFrame(rows)
    splits = labels[
        [
            "dataset_id",
            "run_id",
            "case_id",
            "canonical_service_id",
            "window_start_ms",
            "window_end_ms",
            "sample_id",
        ]
    ].copy()
    splits["split"] = "test"
    protocol = {
        "evaluation_version": "per-case-v1",
        "evaluation_excluded_cases": [],
        "evaluation_selection": {
            "mode": "per_case_balanced_v1",
            "normal_case_id": "normal",
            "excluded_cases": [],
            "normal_case_quota": "mean_fault_case_matched_negative",
        },
    }
    result = evaluation_table(labels, splits, protocol)
    selected = result[result.evaluation_eligible].merge(
        labels[["sample_id", "label"]], on="sample_id", validate="one_to_one"
    )
    counts = selected.groupby(["case_id", "label"], dropna=False).size().to_dict()
    assert counts[("fault-a", 0)] == counts[("fault-a", 1)] == 3
    assert counts[("fault-b", 0)] == counts[("fault-b", 1)] == 2
    assert counts[("normal", 0)] == 2


def test_historical_protocol_keeps_all_normal_references():
    rows = _rows("normal", "normal_run_reference", 0, 2)
    labels = pd.DataFrame(rows)
    splits = labels[
        [
            "dataset_id",
            "run_id",
            "case_id",
            "canonical_service_id",
            "window_start_ms",
            "window_end_ms",
            "sample_id",
        ]
    ].copy()
    splits["split"] = "test"
    result = evaluation_table(
        labels,
        splits,
        {"evaluation_version": "old", "evaluation_excluded_cases": []},
    )
    assert result.evaluation_eligible.all()
