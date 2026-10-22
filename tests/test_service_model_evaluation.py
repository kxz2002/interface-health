from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.service.labeling.evaluation import evaluation_table
from src.service.modeling.data import SampleLists
from src.service.modeling.evaluation import calibrate, evaluate_run
from tests.test_service_model_baselines import hashed


def evaluation_fixture():
    labels = pd.DataFrame(
        [
            dict(
                dataset_id="tt",
                run_id=case,
                case_id=case,
                canonical_service_id="svc",
                window_start_ms=index * 15000,
                window_end_ms=(index + 1) * 15000,
                sample_id=str(index),
                label=label,
                label_source=source,
            )
            for index, (case, label, source) in enumerate(
                [
                    ("normal", 0, "normal_run_reference"),
                    ("fault", 1, "network_participant_association"),
                    ("fault", 0, "pre_intervention_reference"),
                    ("fault", pd.NA, "unknown"),
                ]
            )
        ]
    )
    splits = labels.drop(columns=["label", "label_source"]).assign(split="test")
    versions = dict(label_version="l", split_version="s", evaluation_version="e")
    evaluation = evaluation_table(labels, splits, versions)
    tables = SimpleNamespace(
        labels=labels, splits=splits, evaluation=evaluation, versions=versions, protocol=versions
    )
    lists = SampleLists(
        common_fit=(),
        common_calibration=(),
        common_main_test=("0", "1"),
        extended_main_test=("0", "1"),
        extended_inference=("0", "1", "2", "3"),
        diagnostic_pre=("2",),
        diagnostic_positive=("1",),
        diagnostic_groups={
            "fault|svc": dict(pre=("2",), positive=("1",), reference_available=True)
        },
        excluded=(),
        reasons={},
        lists_hash="",
        mode="common",
    )
    lists = hashed(lists)
    scores = labels[["sample_id"]].assign(score=[0.0, 4.0, 2.0, 9.0])
    scores["lists_hash"] = lists.lists_hash
    return tables, lists, scores


def test_quantiles_and_missing_calibration():
    scores = pd.DataFrame(dict(sample_id=list("abcde"), score=range(5)))
    scores["label"] = 0
    scores["split"] = "calibration"
    thresholds = calibrate(scores, set("abcde"), [0.95, 0.975, 0.99, 0.995])
    assert thresholds["thresholds"]["0.99"] == pytest.approx(3.96)
    with pytest.raises(ValueError):
        calibrate(scores.iloc[:-1], set("abcde"), [0.99])
    with pytest.raises(ValueError):
        calibrate(scores.iloc[:0], set(), [0.99])


def test_roundtrip_attrs_and_main_evaluation_join(monkeypatch):
    tables, lists, scores = evaluation_fixture()
    from src.service.modeling import evaluation as module

    original = module.join_evaluation_scores
    called = []

    def recording(*args):
        called.append(args[2])
        return original(*args)

    monkeypatch.setattr(module, "join_evaluation_scores", recording)
    report = evaluate_run(tables, lists, scores, {"thresholds": {"0.99": 3.96}})
    assert called[0] == {"0", "1"}
    assert report["main"]["sample_ids"] == ["0", "1"]
    for bad in [
        pd.concat([scores, scores.iloc[:1]]),
        scores.assign(score=np.inf),
        scores.iloc[:-2],
    ]:
        with pytest.raises(ValueError):
            evaluate_run(tables, lists, bad, {"thresholds": {"0.99": 3.96}})
    bad = scores.assign(label_version="conflict")
    with pytest.raises(ValueError):
        evaluate_run(tables, lists, bad, {"thresholds": {"0.99": 3.96}})


def test_single_class_main_cases_and_zero_metrics():
    tables, lists, scores = evaluation_fixture()
    report = evaluate_run(tables, lists, scores, {"thresholds": {"0.99": 10.0}})
    main = report["main"]
    assert main["auroc"] == 1 and main["ap"] == 1
    assert main["case_macro_auroc"] is None and main["case_macro_ap"] is None
    assert main["by_case"]["fault"]["auroc"] is None
    metrics = main["classification"]["0.99"]
    assert metrics["precision"] is None and metrics["recall"] == 0 and metrics["f1"] == 0
    assert metrics["positive_count"] == 1 and metrics["negative_count"] == 1


def test_diagnostic_scope_does_not_change_main_or_threshold():
    tables, lists, scores = evaluation_fixture()
    thresholds = {"thresholds": {"0.99": 3.96}}
    report = evaluate_run(tables, lists, scores, thresholds)
    diagnostic = report["within_case_diagnostic"]["fault|svc"]
    assert diagnostic["status"] == "available"
    assert diagnostic["pre"]["classification"]["0.99"]["normal_alert_rate"] == 0
    assert diagnostic["positive"]["classification"]["0.99"]["recall"] == 1
    assert report["unknown"]["sample_ids"] == ["3"]
    assert report["main"]["sample_ids"] == ["0", "1"]
    assert thresholds == {"thresholds": {"0.99": 3.96}}
    with pytest.raises(ValueError):
        evaluate_run(tables, lists, scores[scores.sample_id.ne("2")], thresholds)
    lists.diagnostic_groups["fault|svc"]["pre"] = ()
    lists.diagnostic_groups["fault|svc"]["reference_available"] = False
    lists = hashed(replace(lists, diagnostic_pre=()))
    scores["lists_hash"] = lists.lists_hash
    report = evaluate_run(tables, lists, scores, thresholds)
    assert report["within_case_diagnostic"]["fault|svc"]["status"] == "unavailable"


def test_equal_threshold_is_not_alert_and_unknown_score_is_conserved():
    tables, lists, scores = evaluation_fixture()
    scores.loc[scores.sample_id.eq("1"), "score"] = 3.96
    result = evaluate_run(tables, lists, scores, {"thresholds": {"0.99": 3.96}})
    assert result["main"]["classification"]["0.99"]["tp"] == 0
    assert result["coverage"]["inference"]["scored"] == 4
    assert result["coverage"]["main"]["by_case"]["fault"]["expected"] == 1
    with pytest.raises(ValueError, match="Missing required"):
        evaluate_run(tables, lists, scores.iloc[:-1], {"thresholds": {"0.99": 3.96}})
    tables.protocol = None
    with pytest.raises(ValueError, match="protocol"):
        evaluate_run(tables, lists, scores, {"thresholds": {"0.99": 3.96}})


def test_unknown_identity_and_nonfinite_extra_scores_are_rejected():
    tables, lists, scores = evaluation_fixture()
    scores["case_id"] = ["normal", "fault", "fault", "wrong"]
    with pytest.raises(ValueError, match="identity"):
        evaluate_run(tables, lists, scores, {"thresholds": {"0.99": 3.96}})


def test_calibration_refuses_wrong_role_extra_scores_and_quantile_selection():
    frame = pd.DataFrame(
        dict(sample_id=["a", "b"], score=[1.0, 2.0], label=[0, 0], split=["calibration"] * 2)
    )
    for bad, expected, quantiles in [
        (frame.assign(split="test"), {"a", "b"}, [0.95, 0.975, 0.99, 0.995]),
        (frame.assign(label=1), {"a", "b"}, [0.95, 0.975, 0.99, 0.995]),
        (frame, {"a"}, [0.95, 0.975, 0.99, 0.995]),
        (frame, {"a", "b"}, [0.99]),
    ]:
        with pytest.raises(ValueError):
            calibrate(bad, expected, quantiles)


def test_neural_score_calibration_interoperability(tmp_path):
    from src.service.modeling.evaluation import annotate_calibration
    from src.service.modeling.training import score_model, train_model
    from tests.test_service_model_training import TinyModel, fixture

    prepared, lists, config = fixture(3)
    config["training"]["epochs"] = 1
    from src.service.modeling.config import content_hash

    config["config_sha256"] = content_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    model = TinyModel()
    assert train_model(model, prepared, lists, config, tmp_path / "run")["status"] == "completed"
    raw = score_model(model, prepared, {"calibration"})
    annotated = annotate_calibration(raw, prepared, {"calibration"})
    thresholds = calibrate(annotated, {"calibration"}, [0.95, 0.975, 0.99, 0.995])
    assert thresholds["count"] == 1
    assert "label" not in raw and "split" not in raw
    with pytest.raises(ValueError, match="known-normal"):
        annotate_calibration(score_model(model, prepared, {"test"}), prepared, {"test"})


def test_changed_membership_or_wrong_score_hash_is_rejected():
    tables, lists, scores = evaluation_fixture()
    thresholds = {"thresholds": {"0.99": 3.96}}
    for changed in [replace(lists, common_main_test=("0",)), replace(lists, diagnostic_pre=())]:
        with pytest.raises(ValueError, match="content/hash"):
            evaluate_run(tables, changed, scores, thresholds)
    with pytest.raises(ValueError, match="lists_hash"):
        evaluate_run(tables, lists, scores.assign(lists_hash="wrong"), thresholds)
    conflicting = scores.assign(model_version="service-baseline-v1:L2")
    conflicting.attrs["model_version"] = "service-model-v1"
    with pytest.raises(ValueError, match="model_version"):
        evaluate_run(tables, lists, conflicting, thresholds)
