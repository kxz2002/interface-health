import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.service.modeling.config import load_config
from src.service.modeling.encoders import TextEncoder
from src.service.modeling.model import build_model, initialize_center


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def batch():
    return dict(
        resource=torch.randn(2, 3),
        resource_mask=torch.ones(2, 3, dtype=torch.bool),
        execution=torch.randn(2, 5),
        execution_mask=torch.ones(2, 5, dtype=torch.bool),
        logs=[
            dict(
                input_ids=torch.tensor([[1, 2, 3], [1, 4, 3]]),
                attention_mask=torch.ones(2, 3, dtype=torch.bool),
                special_mask=torch.tensor([[True, False, True]] * 2),
                group_ids=["a", "b"],
            )
            for _ in range(2)
        ],
        stats=torch.randn(2, 4),
        stats_mask=torch.ones(2, 4, dtype=torch.bool),
    )


@pytest.mark.parametrize("fusion", ["transformer_cls", "transformer_mean", "concat_svdd"])
def test_all_available_branches_receive_gradients(fusion):
    model = build_model({"model": {"fusion": fusion}}, TinyBackbone())
    inputs = batch()
    initialize_center(model, [inputs])
    model(inputs)["score"].mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize("fusion", ["transformer_cls", "transformer_mean", "concat_svdd"])
def test_missing_token_does_not_influence_output(fusion):
    model = build_model({"model": {"fusion": fusion}}, TinyBackbone()).eval()
    inputs = batch()
    inputs["resource_mask"][0] = False
    inputs["logs"][0] = None
    inputs["stats_mask"][0] = False
    initialize_center(model, [inputs])
    expected = model(inputs)["score"]
    inputs["resource"][0] = float("nan")
    inputs["stats"][0] = float("nan")
    torch.testing.assert_close(model(inputs)["score"], expected)
    inputs["execution_mask"][0] = False
    with pytest.raises(ValueError, match="unavailable"):
        model(inputs)


def test_center_fixed_and_reload_score():
    model = build_model({}, TinyBackbone())
    inputs = batch()
    with pytest.raises(RuntimeError, match="center"):
        model(inputs)
    initialize_center(model, [inputs])
    assert model.training
    assert (model.center.abs() >= 0.1).all()
    assert not model.center.requires_grad
    original = model.center.clone()
    model(inputs)["score"].mean().backward()
    torch.testing.assert_close(model.center, original)
    restored = build_model({}, TinyBackbone())
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model.eval()(inputs)["score"], restored.eval()(inputs)["score"])
    with pytest.raises(RuntimeError, match="already"):
        initialize_center(model, [inputs])


def test_reconstruction_learns_and_ignores_unobserved_values():
    torch.manual_seed(42)
    model = build_model(
        {"model": {"fusion": "concat_svdd", "auxiliary_losses": {"reconstruction": 1.0}}},
        TinyBackbone(),
    )
    inputs = batch()
    inputs["resource_mask"][0, 0] = False
    initialize_center(model, [inputs])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    initial = model(inputs)["score"].mean().detach()
    for step in range(60):
        optimizer.zero_grad()
        model(inputs)["aux_losses"]["reconstruction"].backward()
        optimizer.step()
    model.eval()
    expected = model(inputs)["score"].detach()
    assert expected.mean() < initial * 0.1
    inputs["resource"][0, 0] = float("nan")
    torch.testing.assert_close(model(inputs)["score"], expected)


@pytest.mark.parametrize("weight", [0, -1, True, "1", float("inf"), float("nan")])
def test_invalid_reconstruction_weight_is_rejected(tmp_path, weight):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"auxiliary_losses": {"reconstruction": weight}}}))
    with pytest.raises(ValueError, match="weight"):
        load_config(path)


def test_reconstruction_config_loads(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"auxiliary_losses": {"reconstruction": 1.0}}}))
    assert load_config(path)["model"]["auxiliary_losses"] == {"reconstruction": 1.0}


def test_equal_message_template_pooling():
    encoder = TextEncoder(TinyBackbone())
    messages = batch()["logs"][0]
    messages["input_ids"] = torch.tensor([[1, 2, 3], [1, 4, 3], [1, 6, 3]])
    messages["attention_mask"] = torch.ones(3, 3, dtype=torch.bool)
    messages["special_mask"] = torch.tensor([[True, False, True]] * 3)
    messages["group_ids"] = ["a", "a", "b"]
    expected = (
        encoder.backbone.embedding.weight[2:5:2].mean(0) + encoder.backbone.embedding.weight[6]
    ) / 2
    torch.testing.assert_close(encoder.pool_messages(messages), expected)
    messages["counts"] = [100, 1, 900]
    torch.testing.assert_close(encoder.pool_messages(messages), expected)
    stats = torch.zeros(1, 4)
    masks = torch.ones(1, 4, dtype=torch.bool)
    first = encoder([messages], stats, masks).token
    stats[0, 0] = 10
    assert not torch.equal(first, encoder([messages], stats, masks).token)


def test_mean_fusion_excludes_missing_outputs():
    model = build_model({"model": {"fusion": "transformer_mean"}}, TinyBackbone()).eval()

    class IdentityTransformer(nn.Module):
        def forward(self, tokens, src_key_padding_mask):
            assert src_key_padding_mask.tolist() == [[False, True, True]]
            return tokens.masked_fill(src_key_padding_mask[:, :, None], 10000)

    model.transformer = IdentityTransformer()
    inputs = batch()
    inputs = {key: value[:1] for key, value in inputs.items()}
    inputs["execution_mask"][:] = False
    inputs["logs"] = [None]
    expected_token = model.resource_encoder(inputs["resource"], inputs["resource_mask"]).token
    expected = model.projection(expected_token + model.modality_embedding.weight[0])
    torch.testing.assert_close(model.encode(inputs)["representation"], expected)


def test_no_body_messages_cannot_be_made_valid_by_statistics():
    encoder = TextEncoder(TinyBackbone())
    messages = batch()["logs"][0]
    messages["special_mask"][:] = True
    result = encoder(
        [messages], torch.full((1, 4), float("nan")), torch.ones(1, 4, dtype=torch.bool)
    )
    assert not result.valid.any()
    assert torch.equal(result.token, torch.zeros(1, 64))


def test_new_linear_layers_have_no_bias_and_layer_norm_has_affine_parameters():
    model = build_model({}, TinyBackbone())
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not name.startswith("text_encoder.backbone"):
            assert module.bias is None, name
        if isinstance(module, nn.LayerNorm):
            assert module.weight is not None and module.bias is not None


def test_nonfinite_observed_values_fail_and_empty_fit_preserves_mode():
    model = build_model({}, TinyBackbone()).train()
    inputs = batch()
    inputs["resource"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        model.encode(inputs)
    with pytest.raises(ValueError, match="Empty fit"):
        initialize_center(model, [])
    assert model.training
    assert not model.center_initialized
