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
        json.dumps({"provenance": provenance, "artifacts": artifacts})
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
