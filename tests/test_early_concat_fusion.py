import pytest
import torch

from src.fusion.early_concat import EarlyConcatFusion


def test_concat_output_dim_18():
    fusion = EarlyConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3}
    )
    assert fusion.output_dim == 18


def test_concat_forward_shape():
    fusion = EarlyConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3}
    )
    batch = {
        "endpoint_red": torch.randn(4, 10),
        "service_metric": torch.randn(4, 5),
        "service_log": torch.randn(4, 3),
    }
    out = fusion(batch)
    assert out.shape == (4, 18)


def test_concat_order_is_deterministic():
    """输出顺序必须固定（endpoint_red, service_metric, service_log），否则下游模型层会错位。"""
    fusion = EarlyConcatFusion(
        modality_dims={"endpoint_red": 2, "service_metric": 1, "service_log": 1}
    )
    batch = {
        "endpoint_red": torch.tensor([[1.0, 2.0]]),
        "service_metric": torch.tensor([[3.0]]),
        "service_log": torch.tensor([[4.0]]),
    }
    out = fusion(batch)
    assert torch.equal(out, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))


def test_missing_modality_key_raises():
    """modality_dims 缺少 MODALITY_ORDER 中的 key 时应抛 ValueError。"""
    with pytest.raises(ValueError):
        EarlyConcatFusion(
            modality_dims={"endpoint_red": 10, "service_metric": 5}
        )  # missing service_log


def test_extra_modality_key_raises():
    """modality_dims 含多余 key 时应抛 ValueError。"""
    with pytest.raises(ValueError):
        EarlyConcatFusion(
            modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3, "extra": 2}
        )
