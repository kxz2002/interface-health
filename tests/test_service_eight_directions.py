import copy
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_service_eight_directions import canonical_json_value, verify_tracking
from scripts.service_eight_directions import restore_checkpoint, validate_config
from scripts.service_temporal_model import validate_temporal_config
from src.service.modeling.eight_directions import (
    DIRECTIONS,
    ExplorationPredictor,
    candidate_loss,
    fit_prototypes,
    fused_score,
    fusion_scales,
    mechanism_components,
    prototype_distances,
    score_candidate,
    train_candidate,
)
from src.service.modeling.temporal import (
    GroupedPredictor,
    TemporalPredictor,
    multiscale_context,
    score_temporal,
    temporal_inputs,
    train_temporal,
)
from tests.test_service_model_temporal import fixture


def configuration():
    config = json.loads(Path("configs/service/gru_eight_directions_v1.json").read_text())
    config["training"].update(device="cpu", epochs=2, batch_size=3)
    config["pretraining_epochs"] = 1
    return config


def test_frozen_protocol_rejects_unplanned_changes():
    config = json.loads(Path("configs/service/gru_eight_directions_v1.json").read_text())
    validate_config(config)
    config["fusion_auxiliary_weight"] = 0.9
    with pytest.raises(ValueError, match="frozen"):
        validate_config(config)


@pytest.mark.parametrize("name", ["multiscale", "grouped"])
def test_standalone_candidates_accept_frozen_schemas(name):
    path = Path(f"configs/service/tt_temporal_prior_{name}_gpu_b_v1.json")
    config = json.loads(path.read_text())
    validate_temporal_config(config)
    if name == "multiscale":
        config["feature_mode"] = "base"
    else:
        config["training"]["architecture"] = "shared_gru"
    with pytest.raises(ValueError, match="protocol"):
        validate_temporal_config(config)


def test_multiscale_uses_only_declared_context_and_preserves_missingness():
    context = torch.zeros(1, 4, 28)
    context[..., 0] = torch.tensor([1.0, 3.0, 6.0, 10.0])
    context[..., 12] = 1
    context[..., -1] = 1
    expanded = multiscale_context(context)
    assert expanded.shape == (1, 4, 76)
    torch.testing.assert_close(expanded[0, :, 27], torch.tensor([0.0, 2.0, 3.0, 4.0]))
    torch.testing.assert_close(expanded[0, :, 51], torch.tensor([0.0, 0.0, 2.5, 3.5]))
    context[0, 1, 12] = 0
    changed = multiscale_context(context)
    assert changed[0, 1, 27] == 0 and changed[0, 2, 27] == 0
    assert changed[0, 3, 51] == 0
    altered = context.clone()
    altered[0, 3, 0] = 999
    torch.testing.assert_close(multiscale_context(altered)[:, :3], changed[:, :3])


def test_multiscale_feature_adapter_does_not_expand_history():
    prepared, _ = fixture()
    base = temporal_inputs(prepared)
    expanded = temporal_inputs(prepared, feature_mode="multiscale")
    assert expanded["history_ids"] == base["history_ids"]
    np.testing.assert_array_equal(
        expanded["context"], multiscale_context(torch.from_numpy(base["context"]))
    )
    prepared.lists.fit = tuple(
        sample_id for sample_id in prepared.lists.fit if sample_id != "fit-002"
    )
    altered = temporal_inputs(prepared, feature_mode="multiscale")
    assert np.count_nonzero(altered["context"][altered["ids"].index("fit-003")]) == 0


def test_grouped_branches_receive_only_their_own_modality_and_parameter_budget():
    model = GroupedPredictor(28)
    base = TemporalPredictor()
    assert sum(p.numel() for p in model.parameters()) <= 1.5 * sum(
        p.numel() for p in base.parameters()
    )
    seen = []
    handle = model.branches[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    context = torch.zeros(2, 4, 28)
    model(context)
    context[..., 3:8] = 999
    model(context)
    handle.remove()
    torch.testing.assert_close(seen[0], seen[1])


@pytest.mark.parametrize("direction", ["baseline", *DIRECTIONS])
def test_candidates_train_normal_only_and_checkpoint_roundtrip(direction, tmp_path):
    torch.set_num_threads(1)
    prepared, _ = fixture()
    config = configuration()
    inputs = temporal_inputs(prepared)
    model, history = train_candidate(prepared, inputs, direction, config, 42)
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6, "cpu_cores"] = 10000
    repeat, _ = train_candidate(changed, temporal_inputs(changed), direction, config, 42)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, repeat.state_dict()[name])
    result = score_candidate(model, inputs, {"test", "calibration"})
    assert np.isfinite(result.score).all()
    assert len(history) == (3 if direction == "masked_pretraining" else 2)
    path = tmp_path / "model.pt"
    torch.save(dict(model_direction=direction, config=config, model_state=model.state_dict()), path)
    restored, _ = restore_checkpoint(path, "cpu")
    np.testing.assert_array_equal(
        score_candidate(restored, inputs, {"test", "calibration"}).score, result.score
    )


def test_baseline_wrapper_reproduces_original_training_and_scores():
    torch.set_num_threads(1)
    prepared, _ = fixture()
    config = configuration()
    inputs = temporal_inputs(prepared)
    original, _ = train_temporal(prepared, inputs, {**config["training"], "seed": 42})
    wrapper, _ = train_candidate(prepared, inputs, "baseline", config, 42)
    for name, parameter in original.state_dict().items():
        assert torch.equal(parameter, wrapper.base.state_dict()[name])
    np.testing.assert_array_equal(
        score_candidate(wrapper, inputs, {"test"}).score,
        score_temporal(original, inputs, {"test"}).score,
    )


def test_dual_head_does_not_invent_missing_history_targets():
    config = configuration()
    model = ExplorationPredictor("dual_head", config)
    batch = dict(
        context=torch.zeros(2, 4, 28),
        prior_logits=torch.zeros(2, 3),
        values=torch.ones(2, 12),
        observed=torch.ones(2, 12, dtype=torch.bool),
        availability=torch.ones(2, 3),
    )
    original = candidate_loss(model, batch)
    with torch.no_grad():
        model.delta_head.bias.fill_(100)
    changed = candidate_loss(model, batch)
    torch.testing.assert_close(original[0], changed[0])
    assert torch.count_nonzero(changed[3]) == 0


def test_prototypes_use_fit_ids_and_mechanisms_require_valid_history():
    torch.set_num_threads(1)
    prepared, _ = fixture()
    config = configuration()
    inputs = temporal_inputs(prepared)
    model, _ = train_candidate(prepared, inputs, "baseline", config, 1)
    memory = fit_prototypes(model, prepared, inputs, config, 1)
    assert memory["fit_ids"] == sorted(prepared.lists.fit)
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6, "cpu_cores"] = 10000
    assert fit_prototypes(model, changed, temporal_inputs(changed), config, 1) == memory
    assert np.isfinite(prototype_distances(model, inputs, {"test"}, memory)).all()
    assert np.count_nonzero(mechanism_components(inputs, {"test"})) == 0
    values = np.array([[1.0, 0.0], [2.0, 1.0]])
    scales = fusion_scales(values, 0.001)
    assert np.isfinite(fused_score(values, scales, 0.2)).all()


def test_tcn_encoding_is_causal():
    model = ExplorationPredictor("tcn_gru", configuration())
    context = torch.randn(2, 4, 28)
    original = model.encode(context)[0]
    context[:, 3] = 100
    torch.testing.assert_close(original[:, :3], model.encode(context)[0][:, :3])


def tracking_fixture(direction="baseline"):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE runs (run_uuid TEXT, status TEXT)")
    connection.execute("CREATE TABLE metrics (run_uuid TEXT, key TEXT, step INTEGER, value REAL)")
    connection.execute("INSERT INTO runs VALUES ('example', 'FINISHED')")
    config = configuration()
    phases = (
        [("pretraining", config["pretraining_epochs"])] if direction == "masked_pretraining" else []
    ) + [("prediction", config["training"]["epochs"])]
    records = (
        []
        if direction in {"prototype", "mechanism"}
        else [
            dict(epoch=epoch, phase=phase, mean_loss=0.25)
            for phase, epochs in phases
            for epoch in range(1, epochs + 1)
        ]
    )
    histories = {scope: copy.deepcopy(records) for scope in ["inner", "outer"]}
    metrics = [
        ("example", f"{scope}_{record['phase']}_fit_loss", record["epoch"] + offset, 0.25)
        for scope, entries in histories.items()
        for record in entries
        for offset in [
            (100 if scope == "outer" else 0)
            + (
                config["pretraining_epochs"]
                if direction == "masked_pretraining" and record["phase"] == "prediction"
                else 0
            )
        ]
    ]
    connection.executemany(
        "INSERT INTO metrics VALUES (?, ?, ?, ?)",
        metrics
        + [
            ("example", "main_auroc", 250, 0.8),
            ("example", "main_ap", 250, 0.6),
        ],
    )
    manifest = dict(mlflow_run_id="example", run_id=f"{direction}-seed1")
    report = dict(main=dict(auroc=0.8, ap=0.6))
    return connection, manifest, histories, config, report


def test_tracking_audit_rejects_missing_epochs_and_metric_drift():
    connection, manifest, histories, config, report = tracking_fixture()
    assert verify_tracking(connection, manifest, histories, config, report) == 4
    connection.execute("UPDATE metrics SET value=0.1 WHERE key='main_ap'")
    with pytest.raises(AssertionError):
        verify_tracking(connection, manifest, histories, config, report)
    connection.execute("DELETE FROM metrics WHERE key='inner_prediction_fit_loss'")
    with pytest.raises(ValueError, match="epoch"):
        verify_tracking(connection, manifest, histories, config, report)
    connection.close()


@pytest.mark.parametrize("direction", ["baseline", "masked_pretraining", "prototype", "mechanism"])
def test_tracking_audit_accepts_only_complete_frozen_schedules(direction):
    connection, manifest, histories, config, report = tracking_fixture(direction)
    expected = (
        0
        if direction in {"prototype", "mechanism"}
        else 6 if direction == "masked_pretraining" else 4
    )
    assert verify_tracking(connection, manifest, histories, config, report) == expected
    connection.close()


@pytest.mark.parametrize(
    "mutation", ["truncate", "missing_scope", "empty_scope", "duplicate", "wrong_phase", "reorder"]
)
def test_tracking_audit_rejects_incomplete_or_invalid_histories(mutation):
    connection, manifest, histories, config, report = tracking_fixture("masked_pretraining")
    if mutation == "truncate":
        histories["inner"].pop()
        connection.execute("DELETE FROM metrics WHERE key='inner_prediction_fit_loss' AND step=3")
    elif mutation == "missing_scope":
        del histories["outer"]
    elif mutation == "empty_scope":
        histories["outer"] = []
    elif mutation == "duplicate":
        histories["inner"].append(copy.deepcopy(histories["inner"][-1]))
    elif mutation == "wrong_phase":
        histories["inner"][0]["phase"] = "prediction"
    else:
        histories["inner"].reverse()
    with pytest.raises(ValueError, match="schedule"):
        verify_tracking(connection, manifest, histories, config, report)
    connection.close()


@pytest.mark.parametrize("direction", ["baseline", "prototype"])
def test_tracking_audit_rejects_unexpected_mlflow_training_records(direction):
    connection, manifest, histories, config, report = tracking_fixture(direction)
    connection.execute(
        "INSERT INTO metrics VALUES ('example', 'outer_prediction_fit_loss', 999, 0.25)"
    )
    with pytest.raises(ValueError, match="epoch count"):
        verify_tracking(connection, manifest, histories, config, report)
    connection.close()


def test_tracking_audit_rejects_duplicate_main_metrics():
    connection, manifest, histories, config, report = tracking_fixture()
    connection.execute("INSERT INTO metrics VALUES ('example', 'main_ap', 250, 0.6)")
    with pytest.raises(ValueError, match="duplicated"):
        verify_tracking(connection, manifest, histories, config, report)
    connection.close()


def test_preprocessing_audit_compares_full_state_in_its_persisted_json_representation():
    rebuilt = dict(
        states=dict(numeric=dict(fit_ids=("normal-a", "normal-b"), mean=1.0)),
        preprocessor_hash="same",
    )
    stored = json.loads(json.dumps(rebuilt))
    assert stored != rebuilt
    assert stored == canonical_json_value(rebuilt)
    altered = copy.deepcopy(rebuilt)
    altered["states"]["numeric"]["mean"] = 2.0
    assert stored != canonical_json_value(altered)
