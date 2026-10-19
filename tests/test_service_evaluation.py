"""Service evaluation preserves missing evidence and conservative retrieval."""

import numpy as np
import pandas as pd
import pytest

from src.service.evaluation import aggregate_events, evaluate_s1, evaluate_s2
from src.service.protocol import ServiceProtocol


def protocol():
    return ServiceProtocol(
        dict(
            window_ms=15000,
            event_min_coverage=0.8,
            event_top_fraction=0.2,
            feature_groups={
                "CPU-only": ["cpu_cores"],
                "R3": ["cpu_cores", "memory_usage_ratio", "process_count"],
            },
            methods=["L2", "max", "OCSVM", "IF"],
        )
    )


def frames(n=10):
    rows = []
    for i in range(n):
        rows.append(
            dict(
                sample_id=str(i),
                case_id="CPU",
                canonical_service_id="target",
                window_start_ms=i * 15000,
                window_end_ms=(i + 1) * 15000,
                partition="fault",
                feature_group="R3",
                method="L2",
                anomaly_score=float(i),
                calibrated_score=i / 10,
                score_status="Scored",
                quality_reasons=[],
            )
        )
    scores = pd.DataFrame(rows)
    index = scores[
        [
            "sample_id",
            "case_id",
            "canonical_service_id",
            "window_start_ms",
            "window_end_ms",
            "partition",
        ]
    ].copy()
    index["proxy_exposure_label"] = [0] * (n // 2) + [1] * (n - n // 2)
    events = pd.DataFrame(
        [
            dict(
                event_id="CPU",
                case_id="CPU",
                configured_target_service="target",
                evaluation_start_ms=0,
                evaluation_end_ms=n * 15000,
            )
        ]
    )
    return scores, index, events


def test_top20_and_coverage():
    scores, _, events = frames()
    scores.loc[8:, ["anomaly_score", "calibrated_score"]] = np.nan
    scores.loc[8:, "score_status"] = "Unknown"
    result = (
        aggregate_events(scores, events, ["target"], protocol())
        .query("feature_group == 'R3' and method == 'L2'")
        .iloc[0]
    )
    assert result.n_expected == 10 and result.n_valid == 8
    assert result.event_score == pytest.approx(0.65)
    scores.loc[7, ["anomaly_score", "calibrated_score"]] = np.nan
    scores.loc[7, "score_status"] = "Unknown"
    assert (
        aggregate_events(scores, events, ["target"], protocol())
        .query("feature_group == 'R3' and method == 'L2'")
        .iloc[0]
        .status
        == "Unscorable"
    )


def test_rank_ties_and_missing():
    _, _, events = frames()
    values = pd.DataFrame(
        [
            dict(
                event_id="CPU",
                canonical_service_id=s,
                feature_group="R3",
                method="L2",
                event_score=v,
                status="Scored" if v is not None else "Unscorable",
            )
            for s, v in [("a", 0.9), ("target", 0.8), ("z", 0.8), ("u", None)]
        ]
    )
    table, metrics = evaluate_s2(values, events, protocol())
    row = table.iloc[0]
    assert row.rank_best == 2 and row.rank_worst == 4 and row.unknown_count == 1
    assert row.hit_at_3 == 0 and row.rr == 0.25
    values.loc[values.canonical_service_id == "target", "event_score"] = np.nan
    table, _ = evaluate_s2(values, events, protocol())
    assert table.iloc[0].rr == 0


def test_all_events_candidates_and_arms():
    scores, _, events = frames()
    events = pd.concat(
        [
            events,
            events.assign(event_id="K1", case_id="K1"),
            events.assign(event_id="K2", case_id="K2"),
        ]
    )
    result = aggregate_events(scores, events, ["target"] + [f"s{i}" for i in range(24)], protocol())
    assert len(result) == 3 * 25 * 8
    comparison, metrics = evaluate_s2(result, events, protocol())
    assert len(comparison) == 24
    assert all(x["denominator"] == 3 for x in metrics["arms"])
    assert comparison[comparison.event_id != "CPU"].rr.eq(0).all()


def evidence(**kwargs):
    base = dict(
        evidence_id="e",
        case_id="CPU",
        canonical_service_id="target",
        unit="WINDOW",
        start_ms=0,
        end_ms=150000,
        state_class="RESOURCE_CHANGE",
        evidence_grade="INDEPENDENT_STATE",
        review_status="confirmed",
        heldout_from_model=True,
        depends_on_input_sources=[],
        independent_label=1,
        fact_type="RESOURCE_STATE",
    )
    base.update(kwargs)
    return base


@pytest.mark.parametrize(
    "change",
    [
        {"unit": "INSTANCE", "state_class": "INSTANCE_TERMINATION", "fact_type": "Killing"},
        {"state_class": "SERVICE_UNAVAILABLE"},
        {"depends_on_input_sources": ["metric"]},
        {"heldout_from_model": False},
        {"review_status": "pending"},
        {"independent_label": None},
    ],
)
def test_s1_rejects_non_gold(change):
    scores, index, _ = frames()
    result = evaluate_s1(scores, pd.DataFrame([evidence(**change)]), index)
    assert result["independent_s1"]["n_known"] == 0
    assert result["independent_s1"]["auroc"] is None
    assert result["s1_proxy_controlled_exposure"]["rows"][0]["auroc"] == 1


def test_s1_single_class_null_and_unknown():
    scores, index, _ = frames()
    result = evaluate_s1(scores, pd.DataFrame([evidence()]), index)
    assert result["independent_s1"]["n_known"] == 10
    assert result["independent_s1"]["rows"][0]["auroc"] is None
    assert result["independent_s1"]["rows"][0]["auprc"] is None


def test_evaluate_rejects_missing_manifest_before_output(tmp_path):
    from scripts.service_experiment import evaluate

    data = tmp_path / "data"
    run = tmp_path / "run"
    data.mkdir()
    run.mkdir()
    (data / "_SUCCESS").write_text("ok")
    (run / "_SUCCESS").write_text("ok")
    with pytest.raises((ValueError, FileNotFoundError)):
        evaluate(
            __import__("pathlib").Path("configs/service/minimal_v1.yaml"),
            __import__("pathlib").Path("configs/data/new_merge.yaml"),
            data,
            run,
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_guard_interval_has_35_complete_windows():
    scores, _, events = frames(45)
    events["evaluation_start_ms"] = 30001
    events["evaluation_end_ms"] = 580000
    result = (
        aggregate_events(scores, events, ["target"], protocol())
        .query("feature_group == 'R3' and method == 'L2'")
        .iloc[0]
    )
    assert result.n_expected == 35
    assert result.n_valid == 35


def test_fit_labels_are_not_independent_eval():
    scores, index, _ = frames()
    scores["partition"] = "normal_fit"
    index["partition"] = "normal_fit"
    result = evaluate_s1(scores, pd.DataFrame([evidence()]), index)
    assert result["independent_s1"]["n_known"] == 0


def test_manifest_inventory_requires_normal_references(tmp_path):
    import json

    from scripts.service_experiment import RUN_REQUIRED_ARTIFACTS, _load_score_manifest

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"config": {}, "outputs": {"scores.parquet": {"sha256": "0" * 64, "size_bytes": 0}}}
        )
    )
    with pytest.raises(ValueError, match="missing required"):
        _load_score_manifest(tmp_path, RUN_REQUIRED_ARTIFACTS)
