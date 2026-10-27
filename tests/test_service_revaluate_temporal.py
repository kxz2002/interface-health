from types import SimpleNamespace

import pandas as pd
import pytest

from scripts.service_revaluate_temporal import evaluate_source, rebind
from src.service.modeling.data import InputTables, build_sample_lists
from tests.test_service_model_membership import membership_fixture


def test_rebind_keeps_training_membership_and_excludes_case_everywhere():
    tables, numeric, logs = membership_fixture()
    tables.labels["label_version"] = "old"
    tables.splits["split_version"] = "split"
    tables = InputTables(
        grid=tables.grid,
        resource=None,
        execution=None,
        log=None,
        labels=tables.labels,
        splits=tables.splits,
        evaluation=tables.evaluation,
        versions={"label_version": "old", "split_version": "split", "evaluation_version": "v2"},
        hashes={},
        feature_tables={},
        config={},
        protocol={},
    )
    original = build_sample_lists(tables, numeric, logs, "common")
    prepared = SimpleNamespace(
        hashes={"label_manifest_sha256": "source"},
        numeric=numeric,
        logs={"masks": logs.to_dict()},
        lists=original,
    )
    protocol = {"evaluation_version": "v3", "evaluation_excluded_cases": ["fault"]}
    labels = tables.labels.copy()
    labels["label_version"] = "new"
    bundle = {
        "manifest": {
            "derivation": {"source_manifest_sha256": "source"},
            "label_version": "new",
            "split_version": "split",
            "evaluation_version": "v3",
        },
        "labels": labels,
        "splits": tables.splits,
        "evaluation": tables.evaluation.assign(
            evaluation_eligible=tables.evaluation.evaluation_eligible
            & tables.evaluation.case_id.ne("fault")
        ),
        "protocol": protocol,
    }
    updated, lists = rebind(tables, prepared, bundle)
    assert updated.versions["evaluation_version"] == "v3"
    assert lists.fit == original.fit and lists.calibration == original.calibration
    assert lists.main_test == ("c",)
    assert lists.extended_inference == ("c",)
    assert not lists.diagnostic_groups


def test_source_checks_binding_and_thresholds(tmp_path, monkeypatch):
    import json

    from scripts import service_revaluate_temporal as module

    source = tmp_path / "source"
    (source / "run").mkdir(parents=True)
    prepared_path = tmp_path / "prepared"
    prepared_path.mkdir()
    (prepared_path / "manifest.json").write_text("{}")
    versions = {"label_version": "old", "split_version": "split", "evaluation_version": "v2"}
    prepared = SimpleNamespace(
        lists=SimpleNamespace(lists_hash="old", calibration=("cal",)), versions=versions
    )
    lists = SimpleNamespace(calibration=("cal",))
    provenance = {
        "parents": {
            "prepared": {"manifest_sha256": module.file_digest(prepared_path / "manifest.json")}
        },
        "lists_hash": "old",
        "versions": versions,
    }
    artifacts = {
        name: {"path": f"run/{name}.{'json' if name == 'thresholds' else 'parquet'}"}
        for name in ("scores", "calibration_scores", "thresholds")
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {"stage": "temporal_experiment", "provenance": provenance, "artifacts": artifacts}
        )
    )
    pd.DataFrame(
        {"sample_id": ["cal"], "score": [1.0], "label": [0], "split": ["calibration"]}
    ).to_parquet(source / "run/calibration_scores.parquet")
    pd.DataFrame({"sample_id": ["test"], "score": [2.0]}).to_parquet(source / "run/scores.parquet")
    thresholds = module.calibrate(
        pd.read_parquet(source / "run/calibration_scores.parquet"), {"cal"}, list(module.QUANTILES)
    )
    (source / "run/thresholds.json").write_text(json.dumps(thresholds))
    monkeypatch.setattr(module, "check_run", lambda path: None)
    monkeypatch.setattr(module, "evaluate_run", lambda *args, **kwargs: {"ok": True})
    assert evaluate_source(source, None, prepared, lists, prepared_path)[0] == {"ok": True}
    thresholds["thresholds"]["0.99"] = 5
    (source / "run/thresholds.json").write_text(json.dumps(thresholds))
    with pytest.raises(ValueError, match="Stored thresholds"):
        evaluate_source(source, None, prepared, lists, prepared_path)


def test_baseline_reuses_bound_scores_and_rejects_changed_model(tmp_path, monkeypatch):
    import json

    from scripts import service_revaluate_temporal as module

    prepared_path = tmp_path / "prepared"
    prepared_path.mkdir()
    (prepared_path / "manifest.json").write_text("{}")
    versions = {"label_version": "old", "split_version": "split", "evaluation_version": "v2"}
    prepared = SimpleNamespace(lists=SimpleNamespace(lists_hash="old"), versions=versions)
    lists = SimpleNamespace(calibration=("cal",))
    source, score, calibration = (tmp_path / name for name in ("eval", "score", "cal"))
    for root in (source, score, calibration):
        (root / "run").mkdir(parents=True)
    (source / "evaluation").mkdir()
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "manifest.json").write_text("{}")
    model = {"path": str(model_root), "manifest_sha256": "model_hash"}
    for root, scope in ((score, "test"), (calibration, "calibration")):
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": "score",
                    "provenance": {"parents": {"model": model}, "score_scope": scope},
                }
            )
        )
    pd.DataFrame({"sample_id": ["test"], "score": [2.0]}).to_parquet(score / "run/scores.parquet")
    cal = pd.DataFrame(
        {"sample_id": ["cal"], "score": [1.0], "label": [0], "split": ["calibration"]}
    )
    cal.to_parquet(calibration / "run/scores.parquet")
    thresholds = module.calibrate(cal, {"cal"}, list(module.QUANTILES))
    (source / "evaluation/thresholds.json").write_text(json.dumps(thresholds))
    parents = {
        "prepared": {"manifest_sha256": module.file_digest(prepared_path / "manifest.json")},
        "scores": {"path": str(score)},
        "calibration": {"path": str(calibration)},
        "model": model,
    }
    manifest = {
        "stage": "evaluate",
        "training_versions": versions,
        "provenance": {"parents": parents, "score_scope": "test"},
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "check_run", lambda path: None)
    monkeypatch.setattr(
        module,
        "_model_lists",
        lambda *args: SimpleNamespace(calibration=("cal",), lists_hash="method_hash"),
    )
    monkeypatch.setattr(module, "replace", lambda value, **kwargs: value)
    monkeypatch.setattr(module, "evaluate_run", lambda *args, **kwargs: {"ok": True})
    assert evaluate_source(source, None, prepared, lists, prepared_path)[0] == {"ok": True}
    (score / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "score",
                "provenance": {"parents": {"model": {"path": "other"}}, "score_scope": "test"},
            }
        )
    )
    with pytest.raises(ValueError, match="model binding"):
        evaluate_source(source, None, prepared, lists, prepared_path)
