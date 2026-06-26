import pytest
import torch
import torch.nn as nn

from src.models.deep_svdd import DeepSVDD


def test_svdd_init_center_none():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    assert model.center is None


def test_svdd_init_center_from_data():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(16, 18)
    model.init_center(x)
    assert model.center is not None
    assert model.center.shape == (32,)


def test_svdd_scores_positive():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(8, 18)
    model.init_center(x)
    scores = model.score(x)
    assert scores.shape == (8,)
    assert (scores >= 0).all()


def test_svdd_no_bias_in_encoder():
    """Deep SVDD 要求 encoder 无 bias，否则超球退化。"""
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    for name, param in model.named_parameters():
        if "bias" in name:
            pytest.fail(f"encoder 不应有 bias，但发现: {name}")


def test_svdd_loss_shape():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(8, 18)
    model.init_center(x)
    loss = model.svdd_loss(x)
    assert loss.shape == ()  # scalar


def test_svdd_center_is_registered_buffer():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    # PyTorch None buffers live in _buffers but are omitted from named_buffers()
    assert "center" in model._buffers
    assert model._buffers["center"] is None  # None before init_center

    x = torch.randn(16, 18)
    model.init_center(x)
    # After init_center the buffer is a real tensor — named_buffers() now includes it
    buffers_after = dict(model.named_buffers())
    assert "center" in buffers_after
    assert isinstance(buffers_after["center"], torch.Tensor)


def test_svdd_center_moves_with_model():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(16, 18)
    model.init_center(x)
    model.to("cpu")
    assert model.center.device == next(model.parameters()).device


def test_svdd_center_in_state_dict():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(16, 18)
    model.init_center(x)
    sd = model.state_dict()
    assert "center" in sd

    # model2 must have a non-None center buffer before load_state_dict so
    # PyTorch can copy the tensor into the existing slot (None buffers are
    # excluded from state_dict and therefore treated as unexpected keys).
    model2 = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    model2.init_center(torch.randn(16, 18))
    model2.load_state_dict(sd)
    assert model2.center is not None
    assert torch.allclose(model2.center, model.center)


def test_svdd_score_before_init_raises():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)
    x = torch.randn(8, 18)
    with pytest.raises(RuntimeError):
        model.score(x)


def test_svdd_eps_anti_collapse():
    model = DeepSVDD(input_dim=18, hidden_dim=64, rep_dim=32)

    # nn.Module wrapper required — PyTorch rejects plain callables as submodules
    class ZeroEncoder(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 32)

    model.encoder = ZeroEncoder()
    model.init_center(torch.randn(16, 18))
    assert model.center.abs().min().item() >= 0.1
