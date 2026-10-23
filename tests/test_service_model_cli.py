import json

import numpy as np
import pandas as pd
import pytest

from scripts import service_model as cli
from tests.test_service_model_evaluation import evaluation_fixture


def test_frame_hash_supports_array_object_cells():
    frame = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "evidence_ids": [np.asarray(["x", "y"]), ["z"]],
        }
    )
    digest = cli._frame_hash(frame)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_frame_hash_retains_nested_array_contents():
    values = np.arange(2000)
    changed = values.copy()
    changed[1000] = -1
    original = pd.DataFrame({"sample_id": ["a"], "evidence": [{"ids": values}]})
    modified = pd.DataFrame({"sample_id": ["a"], "evidence": [{"ids": changed}]})
    assert cli._frame_hash(original) != cli._frame_hash(modified)


def test_frame_hash_matches_parquet_round_trip(tmp_path):
    frame = pd.DataFrame({"sample_id": ["b", "a"], "evidence": [["x"], []]})
    frame.to_parquet(tmp_path / "frame.parquet", index=False)
    restored = pd.read_parquet(tmp_path / "frame.parquet")
    assert cli._frame_hash(frame) == cli._frame_hash(restored.iloc[::-1])
    assert isinstance(frame.evidence.iloc[0], list)


def test_cli_has_no_implicit_training(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("help must not load inputs or construct models")

    monkeypatch.setattr(cli, "load_inputs", forbidden)
    monkeypatch.setattr(cli, "build_model", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "probe-fit" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main([])
    assert not list(tmp_path.iterdir())


def test_test_scoring_includes_diagnostics_once():
    _, lists, _ = evaluation_fixture()
    assert cli.score_ids(lists, "fit") == set(lists.fit)
    assert cli.score_ids(lists, "test") == {"0", "1", "2"}
    assert cli.score_ids(lists, "main_test") == {"0", "1"}
    assert cli.score_ids(lists, "diagnostic") == {"1", "2"}
    assert cli.score_ids(lists, "extended_test") == {"0", "1", "2", "3"}


def test_default_evaluation_does_not_require_unrequested_extended_scores():
    from src.service.modeling.evaluation import evaluate_run

    tables, lists, scores = evaluation_fixture()
    report = evaluate_run(
        tables,
        lists,
        scores[scores.sample_id.ne("3")],
        {"thresholds": {"0.99": 3.0}},
        score_scope="test",
    )
    assert report["main"]["sample_ids"] == ["0", "1"]
    assert report["within_case_diagnostic"]["fault|svc"]["pre"]["sample_ids"] == ["2"]
    assert report["extended"]["status"] == "not_requested"
    assert report["coverage"]["inference"]["missing_ids"] == ["3"]


def test_method_matrix_is_complete_and_explicit():
    matrix = json.loads(cli.MATRIX_CONFIG.read_text())
    assert set(matrix["methods"]) == set(cli.METHODS)
    assert len(matrix["methods"]) == 9
    assert matrix["automatic_execution"] is False


def test_production_runtime_requires_fixed_local_revision(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []

    class LocalOnly:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoModel=LocalOnly, AutoTokenizer=LocalOnly)
    )
    config = cli.load_config(cli.DEFAULT_CONFIG)
    runtime = cli.LocalRuntime()
    for revision in [None, "main", "test-revision"]:
        config["log"]["revision"] = revision
        with pytest.raises(ValueError, match="fixed"):
            runtime.backbone(config)
    assert calls == []
    config["log"]["revision"] = "a" * 40
    config["log"]["tokenizer_revision"] = "b" * 40
    runtime.backbone(config)
    runtime.tokenizer(config)
    assert all(kwargs["local_files_only"] is True for _, kwargs in calls)
    assert [kwargs["revision"] for _, kwargs in calls] == ["a" * 40, "b" * 40]


@pytest.mark.parametrize(
    "override",
    [
        {"baselines": {"iforest": {"n_estimators": 9}}},
        {"baselines": {"ocsvm": {"nu": 0.2}}},
        {"calibration": {"method": "nearest"}},
        {"calibration": {"comparison": "greater_equal"}},
        {"calibration": {"quantile": 0.95}},
        {"preprocessing": {"std_ddof": 1}},
        {"preprocessing": {"fallback_scale": 2.0}},
    ],
)
def test_unsupported_fixed_config_is_rejected_before_fit(tmp_path, override):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError, match="fixed"):
        cli.load_config(path)
