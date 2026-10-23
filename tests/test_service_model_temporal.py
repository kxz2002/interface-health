import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.service_temporal_model import validate_temporal_config
from src.service.modeling.availability import availability_prior_logits, fit_availability
from src.service.modeling.temporal import (
    TemporalPredictor,
    lag_features,
    score_temporal,
    temporal_inputs,
    temporal_loss,
    train_temporal,
)
from tests.test_service_model_training import fixture as training_fixture


def fixture():
    prepared, _, _ = training_fixture(6)
    prepared.logs["masks"] = {sample_id: True for sample_id in prepared.sample_ids}
    config = json.loads(Path("configs/service/tt_temporal_predictor_gpu_b_v1.json").read_text())
    config["training"].update(device="cpu", epochs=2, hidden_dim=4, batch_size=3)
    return prepared, config


@pytest.mark.parametrize("length", [4, 8])
def test_temporal_context_is_past_only_and_lag_if_has_identical_information(length):
    prepared, _ = fixture()
    inputs = temporal_inputs(prepared, length)
    assert inputs["history_ids"]["fit-003"] == ["fit-000", "fit-001", "fit-002"]
    assert inputs["history_ids"]["test"] == []
    assert inputs["history_ids"]["calibration"] == []
    assert lag_features(inputs).shape == (8, length * 28 + 27)
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[4, "cpu_cores"] = 9999
    changed.logs["masks"]["fit-004"] = False
    other = temporal_inputs(changed, length)
    index = inputs["ids"].index("fit-003")
    np.testing.assert_array_equal(inputs["context"][index], other["context"][index])
    np.testing.assert_array_equal(inputs["values"][index], other["values"][index])


@pytest.mark.parametrize("length", [4, 8])
@pytest.mark.parametrize(
    "boundary", ["case_id", "canonical_service_id", "split", "gap", "unknown_fit"]
)
def test_temporal_context_stops_at_case_service_partition_gap_and_fit_exclusion(boundary, length):
    prepared, _ = fixture()
    if boundary in {"case_id", "canonical_service_id"}:
        prepared.identity.loc[2, boundary] = "other"
    elif boundary == "split":
        prepared.splits.loc[2, "split"] = "test"
    elif boundary == "gap":
        prepared.identity.loc[2, "window_start_ms"] += 5000
    else:
        prepared.lists.fit = tuple(
            sample_id for sample_id in prepared.lists.fit if sample_id != "fit-002"
        )
        prepared.labels.loc[2, "label"] = None
    inputs = temporal_inputs(prepared, length)
    assert inputs["history_ids"]["fit-003"] == []


def test_temporal_loss_ignores_unobserved_values_and_scores_missing_modalities():
    prediction = torch.zeros(1, 12, requires_grad=True)
    target = torch.full((1, 12), float("nan"))
    mask = torch.zeros(1, 12, dtype=torch.bool)
    logits = torch.full((1, 3), 10.0)
    score, values, present = temporal_loss(prediction, logits, target, mask, torch.ones(1, 3))
    missing_score, _, _ = temporal_loss(prediction, logits, target, mask, torch.zeros(1, 3))
    assert values.item() == 0.0 and present.item() < 0.001
    assert missing_score.item() > score.item() + 9.0
    score.mean().backward()
    assert torch.isfinite(prediction.grad).all()


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_temporal_training_is_reproducible_and_ignores_test_values(seed):
    torch.set_num_threads(1)
    prepared, config = fixture()
    config["training"]["seed"] = seed
    inputs = temporal_inputs(prepared)
    model, history = train_temporal(prepared, inputs, config["training"])
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6, "cpu_cores"] = 9999
    repeat, _ = train_temporal(changed, temporal_inputs(changed), config["training"])
    assert len(history) == 2 and np.isfinite(history[-1]["mean_loss"])
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, repeat.state_dict()[name])
    result = score_temporal(model, inputs, {"test", "calibration"})
    assert set(result.sample_id) == {"test", "calibration"}
    assert np.isfinite(result.score).all()
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = TemporalPredictor(hidden_dim=config["training"]["hidden_dim"])
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    np.testing.assert_array_equal(
        score_temporal(restored, inputs, {"test", "calibration"}).score, result.score
    )
    prepared.labels.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="known-normal"):
        train_temporal(prepared, inputs, config["training"])


def test_temporal_config_rejects_invalid_optimization_and_protocol():
    _, config = fixture()
    validate_temporal_config(config)
    config["training"]["epochs"] = 0
    with pytest.raises(ValueError, match="positive integers"):
        validate_temporal_config(config)
    config["training"]["epochs"] = 2
    config["training"]["lr"] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        validate_temporal_config(config)


def test_temporal_bert_config_requires_frozen_revision_and_pooling():
    config = json.loads(Path("configs/service/tt_temporal_prior_bert_gpu_b_v1.json").read_text())
    validate_temporal_config(config)
    config["text_embedding"]["revision"] = "wrong"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_temporal_config(config)


@pytest.mark.parametrize("seed", [1, 2, 42])
def test_temporal_config_records_consistent_neural_and_forest_seeds(seed):
    _, config = fixture()
    config["training"]["seed"] = seed
    config["isolation_forest"]["random_state"] = seed
    validate_temporal_config(config)
    config["isolation_forest"]["random_state"] = 123
    with pytest.raises(ValueError, match="protocol"):
        validate_temporal_config(config)


def test_temporal_config_rejects_unplanned_seed():
    _, config = fixture()
    config["training"]["seed"] = 3
    config["isolation_forest"]["random_state"] = 3
    with pytest.raises(ValueError, match="seed"):
        validate_temporal_config(config)


def test_temporal_inputs_reject_nonboolean_masks_and_stale_statistics():
    prepared, _ = fixture()
    prepared.logs["masks"]["test"] = 1
    with pytest.raises(ValueError, match="boolean"):
        temporal_inputs(prepared)
    prepared.logs["masks"]["test"] = True
    prepared.logs["statistics"]["masks"]["s0"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        temporal_inputs(prepared)
    prepared.logs["statistics"]["state_hash"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        temporal_inputs(prepared)


def test_temporal_service_prior_is_normal_only_and_survives_missing_context():
    prepared, config = fixture()
    for index in range(2):
        prepared.identity.loc[index, "canonical_service_id"] = "other"
        prepared.logs["masks"][f"fit-{index:03}"] = False
    state = fit_availability(prepared, prepared.lists.fit)
    inputs = temporal_inputs(prepared, availability_prior=state)
    restored_inputs = temporal_inputs(prepared, availability_prior=json.loads(json.dumps(state)))
    np.testing.assert_array_equal(inputs["prior_logits"], restored_inputs["prior_logits"])
    assert lag_features(inputs).shape == (8, 142)
    prior = availability_prior_logits(state, prepared, ["fit-000", "test"])
    np.testing.assert_allclose(prior[0], [np.log(3), np.log(3), -np.log(3)], rtol=1e-6)
    np.testing.assert_allclose(prior[1], [np.log(5)] * 3, rtol=1e-6)
    prepared.logs["masks"]["test"] = False
    assert fit_availability(prepared, prepared.lists.fit) == state
    np.testing.assert_array_equal(availability_prior_logits(state, prepared, ["test"])[0], prior[1])
    model = TemporalPredictor(hidden_dim=config["training"]["hidden_dim"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        _, logits = model(torch.zeros(1, 4, 28), torch.as_tensor(prior[[1]]))
    torch.testing.assert_close(logits, torch.as_tensor(prior[[1]]))
    assert inputs["history_ids"]["test"] == []
    broken = copy.deepcopy(state)
    broken["fit_ids"] = broken["fit_ids"][:-1]
    with pytest.raises(ValueError, match="Incompatible"):
        temporal_inputs(prepared, availability_prior=broken)
    prepared.identity.loc[6, "canonical_service_id"] = "unseen"
    with pytest.raises(ValueError, match="Missing service"):
        availability_prior_logits(state, prepared, ["test"])


def test_global_prior_pools_normal_counts_once_and_ignores_test_masks():
    prepared, _ = fixture()
    prepared.identity.loc[0:1, "canonical_service_id"] = "other"
    for index in range(2):
        prepared.logs["masks"][f"fit-{index:03}"] = False
    state = fit_availability(prepared, prepared.lists.fit)
    inputs = temporal_inputs(prepared, availability_prior=state, availability_prior_scope="global")
    expected = np.array([np.log(7), np.log(7), np.log(5 / 3)], dtype=np.float32)
    np.testing.assert_allclose(inputs["prior_logits"], np.tile(expected, (8, 1)), rtol=1e-6)
    prepared.logs["masks"]["test"] = False
    np.testing.assert_allclose(
        availability_prior_logits(state, prepared, ["test"], scope="global")[0], expected
    )
    broken = copy.deepcopy(state)
    broken["counts"]["other"][0] += 1
    with pytest.raises(ValueError, match="count mismatch"):
        availability_prior_logits(broken, prepared, ["test"], scope="global")
    with pytest.raises(ValueError, match="scope"):
        availability_prior_logits(state, prepared, ["test"], scope="unknown")


def test_text_controls_keep_only_past_within_original_case_service_split():
    prepared, _ = fixture()
    vectors = np.arange(len(prepared.sample_ids) * 3, dtype=np.float32).reshape(-1, 3)
    original = temporal_inputs(prepared, text_embeddings=vectors)
    zero = temporal_inputs(prepared, text_embeddings=vectors, text_control="zero")
    reverse = temporal_inputs(prepared, text_embeddings=vectors, text_control="reverse_past")
    assert not zero["text_context"].any()
    index = original["ids"].index("fit-003")
    past = [original["ids"].index(sample_id) for sample_id in ["fit-002", "fit-001", "fit-000"]]
    np.testing.assert_array_equal(reverse["text_context"][index, 1:], vectors[past])
    np.testing.assert_array_equal(reverse["context"], original["context"])
    assert reverse["history_ids"] == original["history_ids"]
    assert reverse["text_control_summary"]["changed_windows"] > 0
    changed = vectors.copy()
    changed[index:] += 100000
    other = temporal_inputs(prepared, text_embeddings=changed, text_control="reverse_past")
    np.testing.assert_array_equal(reverse["text_context"][index], other["text_context"][index])
    with pytest.raises(ValueError, match="requires"):
        temporal_inputs(prepared, text_control="zero")
    with pytest.raises(ValueError, match="control"):
        temporal_inputs(prepared, text_embeddings=vectors, text_control="unknown")


@pytest.mark.parametrize("control", ["zero", "reverse_past"])
def test_text_control_config_training_and_scoring(control):
    filename = "zero" if control == "zero" else "reverse"
    config = json.loads(
        Path(f"configs/service/tt_temporal_prior_bert_{filename}_gpu_b_v1.json").read_text()
    )
    validate_temporal_config(config)
    prepared, _ = fixture()
    vectors = np.arange(len(prepared.sample_ids) * 3, dtype=np.float32).reshape(-1, 3)
    inputs = temporal_inputs(prepared, text_embeddings=vectors, text_control=control)
    config["training"].update(device="cpu", epochs=2, hidden_dim=4)
    model, history = train_temporal(prepared, inputs, config["training"])
    scores = score_temporal(model, inputs, ["test"])
    assert len(history) == 2
    assert np.isfinite(scores.score).all()


def test_temporal_text_context_uses_only_past_cached_embeddings():
    prepared, config = fixture()
    cached = np.arange(len(prepared.sample_ids) * 3, dtype=np.float32).reshape(-1, 3)
    inputs = temporal_inputs(prepared, text_embeddings=cached)
    assert inputs["text_context"].shape == (8, 4, 3)
    assert inputs["text_embedding_dim"] == 3
    index = inputs["ids"].index("fit-003")
    assert inputs["text_context"][index, -1].tolist() == cached[3].tolist()
    changed = cached.copy()
    changed[index] += 10000
    other = temporal_inputs(prepared, text_embeddings=changed)
    np.testing.assert_array_equal(inputs["text_context"][index], other["text_context"][index])
    model = TemporalPredictor(input_dim=28, hidden_dim=4, text_dim=3)
    prediction, logits = model(
        torch.zeros(2, 4, 28),
        torch.zeros(2, 3),
        torch.zeros(2, 4, 3),
    )
    assert prediction.shape == (2, 12) and logits.shape == (2, 3)
    assert lag_features(inputs).shape == (8, 151)


def test_temporal_text_embeddings_require_one_finite_vector_per_sample():
    prepared, _ = fixture()
    with pytest.raises(ValueError, match="text embeddings"):
        temporal_inputs(prepared, text_embeddings=np.zeros((len(prepared.sample_ids) - 1, 3)))
    bad = np.zeros((len(prepared.sample_ids), 3), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="text embeddings"):
        temporal_inputs(prepared, text_embeddings=bad)


def test_text_projection_preserves_absent_history_and_log_masks():
    model = TemporalPredictor(text_dim=3)
    context = torch.zeros(2, 4, 28)
    text = torch.ones(2, 4, 3)
    with torch.no_grad():
        absent = model(context, text_context=text)
        zero = model(context, text_context=torch.zeros_like(text))
    for actual, expected in zip(absent, zero):
        torch.testing.assert_close(actual, expected)
    context[:, -1, -2:] = 1
    with torch.no_grad():
        present = model(context, text_context=text)
        zero = model(context, text_context=torch.zeros_like(text))
    assert not torch.equal(present[0], zero[0])


def test_temporal_text_projection_trains_and_checkpoint_replays():
    prepared, config = fixture()
    cached = np.arange(len(prepared.sample_ids) * 3, dtype=np.float32).reshape(-1, 3)
    inputs = temporal_inputs(prepared, text_embeddings=cached)
    model, _ = train_temporal(prepared, inputs, config["training"])
    assert model.text_projection.weight.grad is not None
    assert torch.isfinite(model.text_projection.weight.grad).all()
    restored = TemporalPredictor(hidden_dim=4, text_dim=3)
    restored.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(
        score_temporal(model, inputs, {"test"}).score,
        score_temporal(restored, inputs, {"test"}).score,
    )


def test_frozen_text_embedding_pool_excludes_special_tokens_and_averages_groups():
    from src.service.modeling.text_embeddings import pool_log_messages

    class FakeBackbone:
        def __call__(self, input_ids, attention_mask):
            hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
            return type("Output", (), {"last_hidden_state": hidden})()

    messages = {
        "input_ids": [[101, 2, 3, 102], [101, 4, 6, 102], [101, 8, 10, 102]],
        "attention_mask": [[1, 1, 1, 1]] * 3,
        "special_mask": [[1, 0, 0, 1]] * 3,
        "group_ids": ["a", "a", "b"],
    }
    result = pool_log_messages(FakeBackbone(), messages, torch.device("cpu"))
    expected = torch.tensor([(3.75 + 9.0) / 2, (3.75 + 9.0) / 2])
    torch.testing.assert_close(result, expected)


def test_frozen_text_cache_uses_offline_loader_without_fitting_or_gradients(monkeypatch):
    import sys
    from types import SimpleNamespace

    from src.service.modeling.text_embeddings import build_frozen_text_embeddings
    from tests.test_service_model_baselines import TinyBackbone, baseline_fixture

    prepared, _, _ = baseline_fixture()
    backbone = TinyBackbone()

    def load(model, **kwargs):
        assert model == "fixed-model"
        assert kwargs == dict(revision="fixed-sha", local_files_only=True)
        return backbone

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)),
    )
    for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]:
        prepared.logs[name]["test"] = []
    ids = sorted(prepared.sample_ids)
    vectors = build_frozen_text_embeddings(
        prepared, ids, model_name="fixed-model", revision="fixed-sha", device="cpu"
    )
    assert vectors.shape == (len(ids), 2)
    np.testing.assert_array_equal(vectors[ids.index("test")], [0, 0])
    assert not backbone.training
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in backbone.parameters()
    )
