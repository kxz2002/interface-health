import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.service_temporal_model import validate_temporal_config
from src.service.modeling.temporal import (
    ServiceConditionedPredictor,
    ShortLongPredictor,
    TemporalPredictor,
    build_temporal_model,
    restore_temporal_model,
    score_temporal,
    temporal_inputs,
    temporal_model_metadata,
    train_temporal,
)
from tests.test_service_model_temporal import fixture


def test_service_bias_zero_initialization_unknown_and_prior():
    torch.manual_seed(42)
    original = TemporalPredictor()
    torch.manual_seed(42)
    conditioned = ServiceConditionedPredictor(["a"])
    context = torch.randn(2, 4, 28)
    prior = torch.randn(2, 3)
    expected, expected_logits = original(context, prior)
    actual, logits = conditioned(context, prior, service_indices=torch.tensor([0, -1]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    with torch.no_grad():
        conditioned.service_bias.fill_(2)
    actual, logits = conditioned(context, prior, service_indices=torch.tensor([0, -1]))
    torch.testing.assert_close(actual[0], expected[0] + 2)
    torch.testing.assert_close(actual[1], expected[1])
    torch.testing.assert_close(logits, expected_logits)


@pytest.mark.parametrize("name,length", [("service_bias", 4), ("short_long", 8)])
@pytest.mark.parametrize("seed", [1, 2, 42])
def test_architecture_fit_only_and_checkpoint_roundtrip(name, length, seed):
    torch.set_num_threads(1)
    prepared, _ = fixture()
    config = json.loads(
        Path(f"configs/service/tt_temporal_prior_gru_{name}_gpu_b_v1.json").read_text()
    )
    validate_temporal_config(config)
    settings = {**config["training"], "seed": seed, "device": "cpu", "epochs": 2}
    inputs = temporal_inputs(prepared, length)
    model, _ = train_temporal(prepared, inputs, settings)
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6, "cpu_cores"] = 9999
    changed.identity.loc[changed.identity.sample_id.eq("test"), "canonical_service_id"] = "unseen"
    changed_inputs = temporal_inputs(changed, length)
    repeat, _ = train_temporal(changed, changed_inputs, settings)
    for key, parameter in model.state_dict().items():
        assert torch.equal(parameter, repeat.state_dict()[key])
    metadata = temporal_model_metadata(model)
    assert "unseen" not in metadata["service_vocabulary"]
    buffer = io.BytesIO()
    torch.save(
        dict(
            model_state=model.state_dict(),
            architecture_metadata=metadata,
            config=config,
            settings=settings,
        ),
        buffer,
    )
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    restored = restore_temporal_model(checkpoint, inputs)
    np.testing.assert_array_equal(
        score_temporal(restored, inputs, {"test"}).score,
        score_temporal(model, inputs, {"test"}).score,
    )
    assert np.isfinite(score_temporal(model, changed_inputs, {"test"}).score).all()
    checkpoint["architecture_metadata"]["architecture"] = "shared_gru"
    with pytest.raises(ValueError, match="metadata mismatch"):
        restore_temporal_model(checkpoint, inputs)
    wrong = copy.deepcopy(config)
    wrong["schema_version"] = "service-temporal-prior-v1"
    with pytest.raises(ValueError, match="configuration keys"):
        validate_temporal_config(wrong)


def test_short_long_branches_receive_frozen_history_lengths():
    model = ShortLongPredictor()
    seen = {}
    model.short_recurrent.register_forward_pre_hook(
        lambda module, args: seen.update(short=args[0].clone())
    )
    model.long_recurrent.register_forward_pre_hook(
        lambda module, args: seen.update(long=args[0].clone())
    )
    context = torch.randn(2, 8, 28)
    model(context)
    torch.testing.assert_close(seen["short"], context[:, -4:])
    torch.testing.assert_close(seen["long"], context)
    with pytest.raises(ValueError, match="eight"):
        model(context[:, :4])


def test_legacy_checkpoint_without_architecture_metadata_restores():
    prepared, config = fixture()
    inputs = temporal_inputs(prepared)
    model, _ = train_temporal(prepared, inputs, config["training"])
    restored = restore_temporal_model(dict(config=config, model_state=model.state_dict()), inputs)
    for key, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[key])
