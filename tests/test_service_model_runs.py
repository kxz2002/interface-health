import json
from pathlib import Path

import pytest

from src.service.modeling.runs import RunRecorder, check_reuse


def configuration(tmp_path):
    return {"tracking_uri": "sqlite:///" + (tmp_path / "tracking.db").as_posix(), "seed": 42}


def test_no_overwrite_and_atomic_status(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    output = tmp_path / "runs" / "one"
    with pytest.raises(MemoryError):
        with RunRecorder(output, configuration(tmp_path), {"stage": "fit"}) as run:
            run.publish({"checkpoint": {"weights": [1]}})
            raise MemoryError("synthetic OOM")
    assert not (output / "manifest.json").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        RunRecorder(output, configuration(tmp_path), {})
    for forbidden in [tmp_path / "raw" / "new", tmp_path / "outputs" / "service"]:
        with pytest.raises(ValueError):
            RunRecorder(forbidden, configuration(tmp_path), {})


def test_mlflow_failure_is_visible(tmp_path, monkeypatch):
    from src.service.modeling import runs

    def fail(*args, **kwargs):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(runs, "ExperimentLogger", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="tracking unavailable"):
        with RunRecorder(output, configuration(tmp_path), {}):
            pass
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "failed"
    assert status["run_id"] == "failed"
    assert not (output / "manifest.json").exists()


def test_evaluation_links_to_training_run(tmp_path):
    import mlflow

    config = configuration(tmp_path)
    provenance = {"stage": "fit", "versions": {"label_version": "old"}}
    with RunRecorder(tmp_path / "fit", config, provenance) as run:
        run.log_epoch({"loss": 1.0, "auc": None}, 0)
        training = run.publish({"checkpoint": {"weights": [1]}, "center": [0.1]})
    evaluations = {}
    for evaluation_id in ["common", "extended"]:
        with RunRecorder(
            tmp_path / evaluation_id,
            config,
            {
                "stage": "evaluate",
                "model_manifest": training,
                "evaluation_id": evaluation_id,
                "versions": {"label_version": "new"},
                "evaluation_table_hash": "table",
                "quality_rule_hash": "quality",
                "main_test_hash": "main",
                "diagnostic_lists_hash": "diag",
            },
        ) as run:
            result = run.publish(
                {
                    "main_metrics": {"auc": 0.7},
                    "diagnostic_metrics": {"recall": 1.0},
                    "thresholds": {"0.99": 3.96},
                }
            )
        assert result["mlflow_run_id"] != training["mlflow_run_id"]
        assert result["training_run_id"] == training["run_id"]
        assert result["training_local_run_id"] == training["run_id"]
        assert result["training_mlflow_run_id"] == training["mlflow_run_id"]
        evaluations[evaluation_id] = result["mlflow_run_id"]
        assert result["training_versions"] == provenance["versions"]
        assert result["evaluation_versions"]["label_version"] == "new"
    tags = (
        mlflow.tracking.MlflowClient(tracking_uri=config["tracking_uri"])
        .get_run(training["mlflow_run_id"])
        .data.tags
    )
    assert "evaluation.common.manifest_sha256" in tags
    assert "evaluation.extended.manifest_sha256" in tags
    assert evaluations["common"] != evaluations["extended"]
    client = mlflow.tracking.MlflowClient(tracking_uri=config["tracking_uri"])
    for evaluation_id, evaluation_run_id in evaluations.items():
        assert tags[f"evaluation.{evaluation_id}.mlflow_run_id"] == evaluation_run_id
        assert (
            client.get_run(evaluation_run_id).data.tags["training_mlflow_run_id"]
            == training["mlflow_run_id"]
        )
    assert len(training["artifacts"]["checkpoint"]["sha256"]) == 64


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite:///C:/local/tracking.db",
        "sqlite:///C:\\local\\tracking.db",
        "sqlite:////tmp/local.db",
    ],
)
def test_sqlite_absolute_uri(tmp_path, uri):
    recorder = RunRecorder(tmp_path / "unused", {"tracking_uri": uri}, {})
    assert "\\" not in recorder.uri


def test_original_exception_propagates(tmp_path):
    failure = KeyboardInterrupt("synthetic interrupt")
    with pytest.raises(KeyboardInterrupt) as captured:
        with RunRecorder(tmp_path / "interrupted", configuration(tmp_path), {}):
            raise failure
    assert captured.value is failure


def test_version_invalidation():
    original = dict(
        fit_ids_hash="f",
        calibration_ids_hash="c",
        feature_hash="x",
        mask_hash="m",
        template_hash="t",
        evaluation_table_hash="e",
        scored_ids=["a"],
        required_score_ids=["a"],
    )
    changed = dict(original, evaluation_table_hash="new", required_score_ids=["a", "b"])
    reuse = check_reuse(original, changed)
    assert not reuse["refit"] and not reuse["recalibrate"]
    assert reuse["rescore_ids"] == ["b"]
    assert reuse["reevaluate"]
    for key in ["fit_ids_hash", "feature_hash", "mask_hash", "template_hash"]:
        assert check_reuse(original, dict(original, **{key: "changed"}))["refit"]
    cal = check_reuse(original, dict(original, calibration_ids_hash="changed"))
    assert cal["recalibrate"] and not cal["refit"]
