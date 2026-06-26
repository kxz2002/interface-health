import pytest
import torch

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
