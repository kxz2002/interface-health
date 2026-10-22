import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.service.modeling.config import DEFAULT_CONFIG, content_hash, load_config
from src.service.modeling.training import probe_fit, score_model, train_model


class TinyModel(nn.Module):
    def __init__(self, constant=False):
        super().__init__()
        self.text_encoder = nn.Module()
        self.text_encoder.backbone = nn.Linear(1, 1, bias=False)
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.register_buffer("center", torch.zeros(1))
        self.register_buffer("center_initialized", torch.tensor(False))
        self.constant = constant
        self.seen = []

    def encode(self, batch):
        self.seen.extend(batch["sample_ids"])
        value = batch["resource"][:, :1]
        representation = self.weight * (torch.ones_like(value) if self.constant else value)
        return dict(representation=representation, aux_losses={})

    def forward(self, batch):
        output = self.encode(batch)
        output["score"] = (output["representation"] - self.center).square().sum(-1)
        return output


def fixture(count=17):
    ids = tuple(f"fit-{index:03}" for index in range(count))
    all_ids = ids + ("test", "calibration")
    columns = [
        "cpu_cores",
        "memory_usage_ratio",
        "process_count",
        "observation_count",
        "duration_mean_ms",
        "duration_p95_ms",
        "duration_max_ms",
        "error_rate",
    ]
    values = pd.DataFrame({name: np.arange(len(all_ids)) + 1.0 for name in columns})
    values.insert(0, "sample_id", all_ids)
    masks = values.copy()
    masks[columns] = True
    lists = SimpleNamespace(fit=ids, lists_hash="frozen")
    logs = {
        name: {sample_id: [] for sample_id in all_ids}
        for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]
    }
    stats = pd.DataFrame(
        dict(sample_id=all_ids, **{f"s{i}": [0.0] * len(all_ids) for i in range(4)})
    )
    stats_masks = pd.DataFrame(
        dict(sample_id=all_ids, **{f"s{i}": [True] * len(all_ids) for i in range(4)})
    )
    logs["statistics"] = dict(
        values=stats, masks=stats_masks, feature_order=list(stats)[1:], state_hash="stats"
    )
    logs["cache"] = dict(cache_key="cache", state_hash="logs")
    identity = pd.DataFrame(
        dict(
            sample_id=all_ids,
            case_id=["normal"] * len(all_ids),
            canonical_service_id=["service"] * len(all_ids),
            window_start_ms=np.arange(len(all_ids)) * 15000,
        )
    )
    prepared = SimpleNamespace(
        sample_ids=all_ids,
        numeric=dict(values=values, masks=masks),
        logs=logs,
        lists=lists,
        identity=identity,
        labels=pd.DataFrame(dict(sample_id=all_ids, label=[0] * len(all_ids))),
        splits=pd.DataFrame(
            dict(sample_id=all_ids, split=["fit"] * count + ["test", "calibration"])
        ),
        versions=dict(
            label_version="synthetic", split_version="synthetic", evaluation_version="synthetic"
        ),
        hashes=dict(input_sha256="synthetic"),
        provenance=dict(
            numeric_state_hash="numeric", log_state_hash="logs", log_statistics_state_hash="stats"
        ),
    )
    config = load_config(DEFAULT_CONFIG)
    config["training"]["epochs"] = 2
    config["training"]["device"] = "cpu"
    config["config_sha256"] = content_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    return prepared, lists, config


def test_accumulation_tail_matches_window_mean(tmp_path, monkeypatch):
    prepared, lists, config = fixture()
    gradients = []

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            gradients.append(float(self.param_groups[1]["params"][0].grad))
            return super().step(closure)

    monkeypatch.setattr("src.service.modeling.training.torch.optim.AdamW", RecordingSGD)
    model = TinyModel()
    config["training"]["epochs"] = 1
    config["config_sha256"] = content_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    result = train_model(model, prepared, lists, config, tmp_path / "run")
    assert result["status"] == "completed"
    torch.manual_seed(config["training"]["prototype_seed"])
    ordered = torch.randperm(17).tolist()
    numbers = np.array(ordered) + 1.0
    center = 4.5
    expected = np.mean(2 * (0.5 * numbers[:16] - center) * numbers[:16])
    assert gradients[0] == pytest.approx(expected)
    updated = 0.5 - config["training"]["other_lr"] * expected
    assert gradients[1] == pytest.approx(2 * (updated * numbers[-1] - center) * numbers[-1])
    assert result["optimizer_steps"] == 2


def test_training_progress_phases_and_failure(tmp_path, monkeypatch, caplog):
    from src.service.modeling import training

    prepared, lists, config = fixture()
    events = []
    original = training._write_progress

    def capture(output, progress):
        original(output, progress)
        recorded = json.loads((output / "progress.json").read_text())
        assert recorded == progress
        events.append(recorded)

    monkeypatch.setattr(training, "_write_progress", capture)
    with caplog.at_level("INFO", logger=training.__name__):
        result = train_model(TinyModel(), prepared, lists, config, tmp_path / "progress")
    assert result["status"] == "completed"
    assert [event["stage"] for event in events] == [
        "center_initialization",
        "initial_diagnostic",
        "training",
        "epoch_diagnostic",
        "training",
        "epoch_diagnostic",
        "completed",
    ]
    assert events[0]["windows_per_second"] is None
    assert events[-1]["windows_completed_in_epoch"] == len(lists.fit)
    assert events[-1]["optimizer_steps"] == 4
    assert events[-1]["elapsed_seconds"] >= 0
    assert "Epoch 1/2 started" in caplog.text
    assert "Epoch 2/2 completed" in caplog.text
    assert not (tmp_path / "progress/progress.json.tmp").exists()
    model = TinyModel()
    model.weight.data.fill_(float("nan"))
    failed = train_model(model, prepared, lists, config, tmp_path / "failure")
    assert failed["status"] == "failed"
    assert events[-1]["stage"] == "failed"
    assert "error" in events[-1]


def test_training_progress_updates_inside_epoch(tmp_path, monkeypatch):
    from src.service.modeling import training

    prepared, lists, config = fixture(161)
    snapshots = []
    monkeypatch.setattr(
        training, "_write_progress", lambda output, progress: snapshots.append(dict(progress))
    )
    result = train_model(TinyModel(), prepared, lists, config, tmp_path / "periodic")
    assert result["status"] == "completed"
    updates = [
        progress
        for progress in snapshots
        if progress["stage"] == "training" and progress["optimizer_steps"] == 10
    ]
    assert len(updates) == 1
    assert updates[0]["windows_completed_in_epoch"] == 160
    assert updates[0]["windows_per_second"] > 0
    assert updates[0]["eta_training_seconds"] > 0


def test_partition_inaccessible_and_last_checkpoint(tmp_path):
    prepared, lists, config = fixture()
    model = TinyModel()
    result = train_model(model, prepared, lists, config, tmp_path / "run")
    assert set(model.seen) == set(lists.fit)
    checkpoint = torch.load(result["checkpoint"], weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["lists_hash"] == lists.lists_hash
    assert checkpoint["selection"] == "last"
    torch.testing.assert_close(checkpoint["model_state"]["weight"], model.weight)
    before = model.weight.clone()
    scores = score_model(model, prepared, {"test", "calibration"})
    assert list(scores.sample_id) == ["calibration", "test"]
    assert "score" in scores and "model_version" in scores
    restored = TinyModel()
    restored.load_state_dict(checkpoint["model_state"])
    restored.training_identity = {
        name: checkpoint[name]
        for name in [
            "model_version",
            "versions",
            "input_hashes",
            "preprocessor",
            "lists_hash",
            "config_sha256",
        ]
    }
    restored.training_identity["checkpoint_sha256"] = result["checkpoint_sha256"]
    pd.testing.assert_frame_equal(score_model(restored, prepared, {"test", "calibration"}), scores)
    torch.testing.assert_close(before, model.weight)
    with pytest.raises(ValueError, match="membership"):
        train_model(
            TinyModel(),
            prepared,
            SimpleNamespace(fit=("test",), lists_hash="other"),
            config,
            tmp_path / "bad",
        )


def test_diagnostics_eval_and_failed_seed_retained(tmp_path):
    prepared, lists, config = fixture(270)
    model = TinyModel(constant=True)
    modes = []
    model.register_forward_pre_hook(lambda module, args: modes.append(module.training))
    result = train_model(model, prepared, lists, config, tmp_path / "constant")
    assert len(result["diagnostic_ids"]) == 256
    assert all(row["degenerate"] for row in result["diagnostics"])
    assert result["seed"] == 42
    assert False in modes
    invalid = TinyModel()
    invalid.weight.data.fill_(float("nan"))
    failed = train_model(invalid, prepared, lists, config, tmp_path / "failed")
    assert failed["status"] == "failed"
    assert failed["seed"] == 42
    assert (tmp_path / "failed" / "training_result.json").is_file()


def test_probe_has_no_formal_model_and_restores_state():
    prepared, lists, config = fixture()
    model = TinyModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = probe_fit(model, prepared, lists, config)
    assert report["status"] == "completed"
    assert report["windows"] == 16 and report["finite_gradients"]
    assert report["backward_calls"] == 16
    assert report["accumulation_windows"] == 16
    assert report["log_window_count"] == 0
    assert report["backbone_gradient_status"] == "unavailable"
    assert report["backbone_gradient_reason"] == "no_log_windows"
    assert report["backbone_finite_gradients"] is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_probe_rejects_disconnected_backbone_with_log_input():
    prepared, lists, config = fixture(3)
    prepared.logs["input_ids"][lists.fit[0]] = [[1, 2, 3]]
    report = probe_fit(TinyModel(), prepared, lists, config)
    assert report["status"] == "failed"
    assert report["backward_calls"] == 3
    assert report["log_window_count"] == 1
    assert report["backbone_gradient_parameter_count"] == 0
    assert report["backbone_gradient_status"] == "failed"
    assert "backbone" in report["error"]


def test_fit_label_guard_and_real_network(tmp_path):
    from src.service.modeling.model import build_model

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4)
            self.embedding = nn.Embedding(8, 4)

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    prepared, lists, config = fixture(3)
    config["training"]["epochs"] = 1
    config["config_sha256"] = content_hash(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )
    for index, sample_id in enumerate(lists.fit):
        count = index + 1
        prepared.logs["input_ids"][sample_id] = [[1, 2, 3]] * count
        prepared.logs["attention_mask"][sample_id] = [[1, 1, 1]] * count
        prepared.logs["special_mask"][sample_id] = [[1, 0, 1]] * count
        prepared.logs["group_ids"][sample_id] = ["template:a"] * count
    model = build_model({"model": {"fusion": "concat_svdd"}}, Backbone())
    probe = probe_fit(model, prepared, lists, config)
    assert probe["status"] == "completed"
    assert probe["backward_calls"] == probe["log_window_count"] == 3
    assert probe["backbone_gradient_status"] == "passed"
    assert probe["backbone_finite_gradients"]
    assert probe["backbone_gradient_parameter_count"] > 0
    assert probe["backbone_gradient_norm"] > 0
    result = train_model(model, prepared, lists, config, tmp_path / "real")
    assert result["status"] == "completed"
    assert result["optimizer_steps"] == 1
    assert len(score_model(model, prepared, set(lists.fit))) == 3
    prepared.labels.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="known-normal"):
        train_model(TinyModel(), prepared, lists, config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_probe_snapshots_on_cpu_and_snapshot_failure_is_reported(monkeypatch):
    from src.service.modeling import training

    prepared, lists, config = fixture()
    model = TinyModel()
    model.weight.grad = torch.ones_like(model.weight)
    original_grad = model.weight.grad.clone()
    observed = []
    original_snapshot = training._cpu_snapshot

    def snapshot(tensor):
        result = original_snapshot(tensor)
        assert result.device.type == "cpu"
        assert result.data_ptr() != tensor.data_ptr()
        observed.append(result)
        return result

    monkeypatch.setattr(training, "_cpu_snapshot", snapshot)
    assert probe_fit(model, prepared, lists, config)["status"] == "completed"
    assert len(observed) == len(model.state_dict()) + 1
    torch.testing.assert_close(model.weight.grad, original_grad)

    def fail_snapshot(tensor):
        raise RuntimeError("synthetic snapshot OOM")

    monkeypatch.setattr(training, "_cpu_snapshot", fail_snapshot)
    report = probe_fit(model, prepared, lists, config)
    assert report["status"] == "failed"
    assert "snapshot OOM" in report["error"]
    assert report["restoration_status"] == "not_required"
    torch.testing.assert_close(model.weight.grad, original_grad)


def test_probe_restore_failure_is_reported(monkeypatch):
    prepared, lists, config = fixture()
    model = TinyModel()

    def fail_restore(state):
        raise RuntimeError("synthetic restore error")

    monkeypatch.setattr(model, "load_state_dict", fail_restore)
    report = probe_fit(model, prepared, lists, config)
    assert report["status"] == "failed"
    assert report["restoration_status"] == "failed"
    assert "synthetic restore error" in report["restoration_error"]
