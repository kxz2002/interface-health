import pytest
import torch

from src.fusion.independent_concat import IndependentConcatFusion


def test_output_dim_is_2x_branch_dim():
    fusion = IndependentConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    assert fusion.output_dim == 32


def test_forward_shape():
    fusion = IndependentConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    batch = {
        "endpoint_red": torch.randn(4, 10),
        "service_metric": torch.randn(4, 5),
        "service_log": torch.randn(4, 3),
    }
    out = fusion(batch)
    assert out.shape == (4, 32)


def test_forward_is_deterministic_given_fixed_weights():
    """固定权重后同输入应产出同输出（非随机性冒烟测试）。"""
    fusion = IndependentConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    batch = {
        "endpoint_red": torch.randn(2, 10),
        "service_metric": torch.randn(2, 5),
        "service_log": torch.randn(2, 3),
    }
    fusion.eval()
    out1 = fusion(batch)
    out2 = fusion(batch)
    assert torch.equal(out1, out2)


def test_missing_modality_key_raises():
    with pytest.raises(ValueError):
        IndependentConcatFusion(
            modality_dims={"endpoint_red": 10, "service_metric": 5}, branch_dim=16
        )  # missing service_log


def test_extra_modality_key_raises():
    with pytest.raises(ValueError):
        IndependentConcatFusion(
            modality_dims={
                "endpoint_red": 10,
                "service_metric": 5,
                "service_log": 3,
                "extra": 2,
            },
            branch_dim=16,
        )


def test_ep_branch_uses_only_endpoint_red_dim():
    """e_ep 分支的 Linear 输入维度应等于 endpoint_red 的维度，与 service 侧维度无关。"""
    fusion = IndependentConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    assert fusion.ep_encoder[0].in_features == 10


def test_svc_branch_uses_concat_of_metric_and_log_dim():
    """e_svc 分支的 Linear 输入维度应等于 service_metric + service_log 维度之和。"""
    fusion = IndependentConcatFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    assert fusion.svc_encoder[0].in_features == 8
