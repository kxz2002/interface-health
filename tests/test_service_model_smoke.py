import json
from pathlib import Path
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from scripts import service_model as cli
from src.service.labeling.evaluation import evaluation_table
from src.service.modality_preparation import full_grid
from src.service.modeling.data import InputTables
from tests.test_service_model_logs import FakeTokenizer


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(10000, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyRuntime:
    identity = {"runtime": "synthetic_test_only", "production_bert": False}

    def tokenizer(self, config):
        return FakeTokenizer()

    def backbone(self, config):
        return TinyBackbone()


def synthetic_tables(config):
    cases = ["fit-a", "fit-a", "cal", "normal-test", "fault", "fault", "fault"]
    grid = pd.concat(
        [
            full_grid("tt", case, ["svc"], index * 15000, (index + 1) * 15000)
            for index, case in enumerate(cases)
        ],
        ignore_index=True,
    ).assign(case_id=cases)
    labels = grid.assign(
        label=pd.array([0, 0, 0, 0, 0, 1, pd.NA], dtype="Int8"),
        label_source=["normal_run_reference"] * 4
        + ["pre_intervention_reference", "network_participant_association", "unknown"],
    )
    splits = grid.assign(split=["fit", "fit", "calibration"] + ["test"] * 4)
    versions = dict(
        label_version="synthetic-l", split_version="synthetic-s", evaluation_version="synthetic-e"
    )
    evaluation = evaluation_table(labels, splits, versions)
    resource = grid.assign(
        cpu_cores=np.arange(7) + 1.0,
        memory_usage_ratio=0.4,
        process_count=3.0,
        resource_valid=True,
        quality_status="RESOURCE_VALID",
    )
    execution = grid.assign(
        observation_count=2.0,
        duration_mean_ms=3.0,
        duration_p95_ms=4.0,
        duration_max_ms=5.0,
        error_count=0.0,
        error_known_count=2.0,
        error_coverage=1.0,
        quality_status="OBSERVED",
    )
    log = grid.assign(
        messages_json='[{"message":"INFO request done","count":1}]',
        event_count=1.0,
        quality_status="OBSERVED",
    )
    return InputTables(
        grid,
        resource,
        execution,
        log,
        labels,
        splits,
        evaluation,
        versions,
        {"input_sha256": "synthetic-only"},
        {},
        config,
        versions,
    )


@pytest.fixture
def offline(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"training": {"epochs": 1, "device": "cpu"}}))
    monkeypatch.setattr(cli, "load_inputs", lambda preparation, labels, cfg: synthetic_tables(cfg))
    common = ["--config", str(config), "--tracking-uri", f"sqlite:///{tmp_path / 'mlflow.db'}"]
    return config, common


@pytest.mark.parametrize("value_method", ["L2", "transformer_cls"])
def test_normal_fit_tail_fusion_end_to_end(tmp_path, offline, value_method):
    from scripts.service_score_fusion import main as fuse

    _, common = offline
    runtime = TinyRuntime()

    def stage(command, name, extra):
        output = tmp_path / name
        assert (
            cli.main([command, *common, "--output", str(output), *extra], runtime_factory=runtime)
            == 0
        )
        return output

    prepared = stage(
        "prepare", "prepared", ["--preparation", "synthetic", "--label-bundle", "synthetic"]
    )
    arguments = [
        "--prepared",
        str(prepared),
        "--output",
        str(tmp_path / "fused"),
        "--tracking-uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
    ]
    for channel, method in [("value", value_method), ("availability", "availability")]:
        fit = stage("fit", channel, ["--prepared", str(prepared), "--method", method])
        for scope in ["fit", "calibration", "test"]:
            scored = stage(
                "score",
                channel + "-" + scope,
                ["--prepared", str(prepared), "--model-run", str(fit), "--score-scope", scope],
            )
            arguments.extend([f"--{channel}-{scope}", str(scored)])
    assert fuse(arguments) == 0
    assert cli.check_run(tmp_path / "fused")["status"] == "success"
    report = json.loads((tmp_path / "fused/evaluation/report.json").read_text())
    assert report["main"]["auroc"] is not None
    assert report["coverage"]["missing_ids"] == []
    state = json.loads((tmp_path / "fused/run/fusion_state.json").read_text())
    _, data, _ = cli.read_prepared(prepared)
    assert state["fit_ids"] == sorted(data.lists.fit)
    with pytest.raises(FileExistsError):
        fuse(arguments)
    bad_arguments = list(arguments)
    bad_arguments[bad_arguments.index("--output") + 1] = str(tmp_path / "bad-fusion")
    bad_arguments.extend(["--value-fit", str(tmp_path / "value-calibration")])
    with pytest.raises(ValueError, match="scope or input binding"):
        fuse(bad_arguments)
    assert not (tmp_path / "bad-fusion").exists()


@pytest.mark.parametrize("method", ["gru", "lag_IF", "service_lag_IF"])
@pytest.mark.parametrize("with_prior", [False, True, "bert", "global"])
def test_temporal_model_end_to_end(tmp_path, offline, method, with_prior, monkeypatch):
    from scripts.service_temporal_model import main as temporal

    preparation_config, common = offline
    if with_prior == "bert":
        settings = json.loads(preparation_config.read_text())
        settings["log"] = dict(
            revision="387825ce42dbb39b87911cdf8e383ee3b25184f8",
            tokenizer_revision="387825ce42dbb39b87911cdf8e383ee3b25184f8",
        )
        preparation_config.write_text(json.dumps(settings))
    prepared = tmp_path / "prepared"
    assert (
        cli.main(
            [
                "prepare",
                *common,
                "--output",
                str(prepared),
                "--preparation",
                "synthetic",
                "--label-bundle",
                "synthetic",
            ],
            runtime_factory=TinyRuntime(),
        )
        == 0
    )
    config_name = (
        "tt_temporal_prior_bert_gpu_b_v1.json"
        if with_prior == "bert"
        else (
            "tt_temporal_prior_gpu_b_v1.json"
            if with_prior
            else "tt_temporal_predictor_gpu_b_v1.json"
        )
    )
    config = json.loads((Path("configs/service") / config_name).read_text())
    if with_prior == "global":
        config["availability_prior"]["source"] = "normal_fit_global"
    config["training"].update(device="cpu", epochs=2)
    if with_prior == "bert":
        from scripts import service_temporal_model

        monkeypatch.setattr(
            service_temporal_model,
            "build_frozen_text_embeddings",
            lambda prepared, ids, **kwargs: np.ones((len(ids), 256), dtype=np.float32),
        )
    config_path = tmp_path / "temporal.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "temporal"
    arguments = [
        "--config",
        str(config_path),
        "--prepared",
        str(prepared),
        "--output",
        str(output),
        "--method",
        method,
        "--seed",
        "1" if with_prior else "2",
        "--tracking-uri",
        f"sqlite:///{tmp_path / 'mlflow.db'}",
    ]
    assert temporal(arguments) == 0
    assert cli.check_run(output)["status"] == "success"
    report = json.loads((output / "evaluation/report.json").read_text())
    assert report["main"]["auroc"] is not None
    assert report["coverage"]["missing_ids"] == []
    checkpoint = torch.load(output / "run/model.pt", weights_only=False)
    expected_seed = 1 if with_prior else 2
    assert checkpoint["config"]["training"]["seed"] == expected_seed
    assert checkpoint["config"]["isolation_forest"]["random_state"] == expected_seed
    if method != "gru":
        estimators = (
            checkpoint["estimator"].values()
            if method == "service_lag_IF"
            else [checkpoint["estimator"]]
        )
        assert all(estimator.random_state == expected_seed for estimator in estimators)
    _, data, _ = cli.read_prepared(prepared)
    assert checkpoint["fit_ids"] == sorted(data.lists.fit)
    if method == "gru":
        assert "model_state" in checkpoint
    elif method == "service_lag_IF":
        assert set(checkpoint["estimator"]) == {"svc"}
    if with_prior:
        assert checkpoint["availability_prior"]["fit_ids"] == sorted(data.lists.fit)
        assert (output / "run/availability_prior.json").exists()
    if with_prior == "bert":
        metadata = json.loads((output / "run/text_embedding_metadata.json").read_text())
        cache = np.load(output / "run/text_embeddings.npy", allow_pickle=False)
        assert cache.shape == (len(data.sample_ids), 256)
        assert metadata["sample_ids"] == sorted(data.sample_ids)
        assert metadata["config"] == config["text_embedding"]
        assert metadata["embedding_dim"] == 256
        if method == "gru":
            monkeypatch.setattr(
                service_temporal_model,
                "build_frozen_text_embeddings",
                lambda *args, **kwargs: pytest.fail("cache reuse re-encoded BERT"),
            )
            cached_arguments = list(arguments)
            cached_arguments[cached_arguments.index("--output") + 1] = str(tmp_path / "cached")
            cached_arguments.extend(["--text-cache-run", str(output)])
            assert temporal(cached_arguments) == 0
            assert cli.check_run(tmp_path / "cached")["status"] == "success"
            np.testing.assert_array_equal(
                np.load(tmp_path / "cached/run/text_embeddings.npy"), cache
            )
    with pytest.raises(FileExistsError):
        temporal(arguments)


def test_offline_end_to_end(tmp_path, offline):
    config, common = offline
    runtime = TinyRuntime()

    def stage(command, name, extra=()):
        output = tmp_path / name
        assert (
            cli.main([command, *common, "--output", str(output), *extra], runtime_factory=runtime)
            == 0
        )
        assert cli.check_run(output)["status"] == "success"
        return output

    checked = stage(
        "check-inputs", "checked", ["--preparation", "synthetic", "--label-bundle", "synthetic"]
    )
    assert not list(checked.rglob("*.pt"))
    prepared = stage(
        "prepare", "prepared", ["--preparation", "synthetic", "--label-bundle", "synthetic"]
    )
    fit = stage("fit", "fit", ["--prepared", str(prepared)])
    score_args = ["--prepared", str(prepared), "--model-run", str(fit)]
    scores = stage("score", "scores", score_args)
    replay = stage("score", "replay", score_args)
    calibration = stage("score", "calibration", [*score_args, "--score-scope", "calibration"])
    pd.testing.assert_frame_equal(cli.read_scores(scores), cli.read_scores(replay))
    tables, data, _ = cli.read_prepared(prepared)
    assert set(cli.read_scores(scores).sample_id) == cli.score_ids(data.lists, "test")
    assert data.logs["statistics"]["state_hash"] == data.provenance["log_statistics_state_hash"]
    assert tables.protocol == tables.versions
    evaluated = stage(
        "evaluate",
        "evaluated",
        [*score_args, "--scores", str(scores), "--calibration-scores", str(calibration)],
    )
    report = json.loads((evaluated / "evaluation/report.json").read_text())
    assert report["main"]["auroc"] is not None
    assert report["within_case_diagnostic"]["fault|svc"]["status"] == "available"
    manifest = json.loads((fit / "manifest.json").read_text())
    assert manifest["provenance"]["runtime"]["production_bert"] is False
    client = mlflow.tracking.MlflowClient(tracking_uri=manifest["tracking_uri"])
    assert client.get_run(manifest["mlflow_run_id"]).info.status == "FINISHED"
    prepared_manifest = json.loads((prepared / "manifest.json").read_text())
    assert prepared_manifest["artifacts"]["prepared"]["local_only"] is True
    prepared_artifacts = client.list_artifacts(prepared_manifest["mlflow_run_id"], "training")
    assert "training/prepared.json" not in {artifact.path for artifact in prepared_artifacts}
    assert "training/config.json" in {artifact.path for artifact in prepared_artifacts}
    assert "training/manifest.pending.json" in {artifact.path for artifact in prepared_artifacts}
    assert cli.main(["check-run", "--run", str(evaluated)]) == 0
    with pytest.raises(FileExistsError):
        cli.main(["score", *common, "--output", str(scores), *score_args], runtime_factory=runtime)
    (scores / "run/scores.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash|artifact"):
        cli.check_run(scores)
    with pytest.raises(ValueError, match="hash|artifact"):
        cli.check_run(evaluated)


@pytest.mark.parametrize(
    "method",
    [
        "transformer_mean",
        "concat_svdd",
        "L2",
        "Max",
        "numeric_IF",
        "numeric_OCSVM",
        "trimodal_IF",
        "trimodal_OCSVM",
        "availability",
    ],
)
def test_all_matrix_methods_support_offline_fit_score_evaluation(
    tmp_path, offline, method, monkeypatch
):
    _, common = offline
    runtime = TinyRuntime()
    if method in {"L2", "Max", "numeric_IF", "numeric_OCSVM", "availability"}:

        def no_backbone(config):
            raise AssertionError("Numeric methods must not load BERT")

        runtime.backbone = no_backbone
    if method in {"L2", "Max", "availability"}:

        def forbidden(*args, **kwargs):
            raise AssertionError("Rule fit must not fit unrequested sklearn methods")

        monkeypatch.setattr("src.service.modeling.baselines.fit_iforest", forbidden)
        monkeypatch.setattr("src.service.modeling.baselines.fit_ocsvm", forbidden)
    prepared = tmp_path / "prepared"
    fit = tmp_path / "fit"
    assert (
        cli.main(
            [
                "prepare",
                *common,
                "--output",
                str(prepared),
                "--preparation",
                "synthetic",
                "--label-bundle",
                "synthetic",
            ],
            runtime_factory=runtime,
        )
        == 0
    )
    assert (
        cli.main(
            ["fit", *common, "--output", str(fit), "--prepared", str(prepared), "--method", method],
            runtime_factory=runtime,
        )
        == 0
    )
    arguments = [*common, "--prepared", str(prepared), "--model-run", str(fit)]
    for name, scope in [("scores", "test"), ("calibration", "calibration")]:
        assert (
            cli.main(
                ["score", *arguments, "--output", str(tmp_path / name), "--score-scope", scope],
                runtime_factory=runtime,
            )
            == 0
        )
    assert (
        cli.main(
            [
                "evaluate",
                *arguments,
                "--output",
                str(tmp_path / "evaluation"),
                "--scores",
                str(tmp_path / "scores"),
                "--calibration-scores",
                str(tmp_path / "calibration"),
            ],
            runtime_factory=runtime,
        )
        == 0
    )


def test_check_inputs_never_constructs_runtime_dependencies(tmp_path, offline):
    _, common = offline

    class ForbiddenRuntime:
        identity = {"runtime": "must_not_construct_dependencies"}

        def tokenizer(self, config):
            raise AssertionError("check-inputs loaded tokenizer")

        def backbone(self, config):
            raise AssertionError("check-inputs loaded backbone")

    output = tmp_path / "checked"
    assert (
        cli.main(
            [
                "check-inputs",
                *common,
                "--output",
                str(output),
                "--preparation",
                "synthetic",
                "--label-bundle",
                "synthetic",
            ],
            runtime_factory=ForbiddenRuntime(),
        )
        == 0
    )
    assert not list(output.rglob("*.pt"))


def test_probe_does_not_publish_a_trained_model(tmp_path, offline):
    _, common = offline
    runtime = TinyRuntime()
    prepared, probe = tmp_path / "prepared", tmp_path / "probe"
    cli.main(
        [
            "prepare",
            *common,
            "--output",
            str(prepared),
            "--preparation",
            "synthetic",
            "--label-bundle",
            "synthetic",
        ],
        runtime_factory=runtime,
    )
    cli.main(
        ["probe-fit", *common, "--output", str(probe), "--prepared", str(prepared)],
        runtime_factory=runtime,
    )
    assert not list(probe.rglob("*.pt"))
    report = json.loads((probe / "report/probe_report.json").read_text())
    assert report["restoration_status"] == "restored"
    with pytest.raises(ValueError, match="successful fit"):
        cli.main(
            [
                "score",
                *common,
                "--output",
                str(tmp_path / "scores"),
                "--prepared",
                str(prepared),
                "--model-run",
                str(probe),
            ],
            runtime_factory=runtime,
        )


@pytest.mark.parametrize(
    "mutation",
    ["evaluation", "new_required", "fit_label", "cal_label", "input", "mask", "split", "model"],
)
def test_fixed_model_reuse_checks_full_evidence(tmp_path, offline, monkeypatch, mutation):
    from dataclasses import replace

    from src.service.modeling.config import content_hash
    from src.service.preparation import file_digest

    _, common = offline
    runtime = TinyRuntime()
    prepared, fit = tmp_path / "prepared", tmp_path / "fit"
    cli.main(
        [
            "prepare",
            *common,
            "--output",
            str(prepared),
            "--preparation",
            "synthetic",
            "--label-bundle",
            "synthetic",
        ],
        runtime_factory=runtime,
    )
    cli.main(
        [
            "fit",
            *common,
            "--output",
            str(fit),
            "--prepared",
            str(prepared),
            "--method",
            "transformer_cls",
        ],
        runtime_factory=runtime,
    )
    arguments = [*common, "--prepared", str(prepared), "--model-run", str(fit)]
    for name, scope in [("scores", "test"), ("calibration", "calibration")]:
        cli.main(
            ["score", *arguments, "--output", str(tmp_path / name), "--score-scope", scope],
            runtime_factory=runtime,
        )
    checkpoint = fit / "run/model/checkpoint_last.pt"
    before = {
        str(path): file_digest(path)
        for path in [
            checkpoint,
            tmp_path / "scores/run/scores.parquet",
            tmp_path / "calibration/run/scores.parquet",
        ]
    }
    tables, data, manifest = cli.read_prepared(prepared)
    labels, splits = tables.labels.copy(), tables.splits.copy()
    versions = {
        **tables.versions,
        "label_version": "synthetic-new-l",
        "evaluation_version": "synthetic-new-e",
    }
    if mutation == "fit_label":
        labels.loc[splits.split.eq("fit"), "label"] = 1
    elif mutation == "cal_label":
        labels.loc[splits.split.eq("calibration"), "label"] = 1
    elif mutation == "split":
        splits.loc[splits.split.eq("fit"), "split"] = "test"
    elif mutation == "new_required":
        labels.loc[labels.label.isna(), "label"] = 1
        labels.loc[labels.label_source.eq("unknown"), "label_source"] = (
            "network_participant_association"
        )
    else:
        labels.loc[labels.label_source.eq("network_participant_association"), "label_source"] = (
            "controlled_new_association"
        )
    evaluation = evaluation_table(labels, splits, versions)
    new_tables = replace(
        tables,
        labels=labels,
        splits=splits,
        versions=versions,
        evaluation=evaluation,
        protocol=versions,
    )
    numeric = {name: frame.copy() for name, frame in data.numeric.items()}
    if mutation == "input":
        numeric["values"].loc[0, "cpu_cores"] += 1
    if mutation == "mask":
        numeric["masks"].loc[0, "cpu_cores"] = False
    lists = cli.build_sample_lists(new_tables, numeric, pd.Series(data.logs["masks"]), "common")
    new_data = replace(
        data, numeric=numeric, labels=labels, splits=splits, versions=versions, lists=lists
    )
    fresh = tmp_path / "new-prepared"
    provenance = {**manifest["provenance"], "versions": versions, "lists_hash": lists.lists_hash}
    prepared_payload = cli._unpack(cli._read(prepared / "data/prepared.json"), prepared)
    arguments = SimpleNamespace(tracking_uri=common[-1])
    with cli._record(fresh, tables.config, arguments, provenance) as recorder:
        artifacts = cli._save_prepared(
            fresh, new_tables, new_data, prepared_payload["preprocessing"]
        )
        cli._publish(recorder, artifacts, local_only={"prepared"})
    assert cli.check_run(fresh)["status"] == "success"
    new_common = common
    if mutation == "model":
        altered = tmp_path / "altered.json"
        altered.write_text(json.dumps({"training": {"epochs": 2, "device": "cpu"}}))
        new_common = ["--config", str(altered), *common[2:]]
    command = [
        "evaluate",
        *new_common,
        "--prepared",
        str(fresh),
        "--model-run",
        str(fit),
        "--scores",
        str(tmp_path / "scores"),
        "--calibration-scores",
        str(tmp_path / "calibration"),
        "--output",
        str(tmp_path / "reevaluated"),
    ]
    if mutation in {"fit_label", "cal_label", "input", "mask", "split", "model"}:
        with pytest.raises(ValueError, match="refit|configuration"):
            cli.main(command, runtime_factory=runtime)
    elif mutation == "new_required":
        with pytest.raises(ValueError, match="rescore_ids"):
            cli.main(command, runtime_factory=runtime)
        cli.main(
            [
                "score",
                *common,
                "--prepared",
                str(fresh),
                "--model-run",
                str(fit),
                "--reuse-scores",
                str(tmp_path / "scores"),
                "--output",
                str(tmp_path / "supplemented"),
            ],
            runtime_factory=runtime,
        )
        evidence = json.loads((tmp_path / "supplemented/run/provenance.json").read_text())[
            "compatibility"
        ]
        assert len(evidence["rescore_ids"]) == 1
        command[command.index("--scores") + 1] = str(tmp_path / "supplemented")
        cli.main(command, runtime_factory=runtime)
    else:
        cli.main(command, runtime_factory=runtime)
        report = json.loads((tmp_path / "reevaluated/evaluation/report.json").read_text())
        assert report["versions"]["evaluation_version"] == "synthetic-new-e"
        assert report["score_source_binding"]["versions"]["evaluation_version"] == "synthetic-e"
        assert "source_evaluation_version" in report["scores"][0]
    assert all(file_digest(Path(path)) == digest for path, digest in before.items())
