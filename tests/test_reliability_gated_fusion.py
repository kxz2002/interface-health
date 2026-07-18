import numpy as np
import pandas as pd
import pytest
import torch

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER
from src.fusion.reliability_gate import ReliabilityGatedFusion

MODALITY_DIMS = {"endpoint_red": 3, "service_metric": 2, "service_log": 1}


def _fitted_stats(endpoints=("epA", "epB")) -> EndpointBaselineStats:
    red_cols = [f"endpoint_red__f{i}" for i in range(3)]
    svc_cols = [f"service_metric__g{i}" for i in range(2)] + ["service_log__h0"]
    rng = np.random.default_rng(0)
    rows = []
    for ep in endpoints:
        for _ in range(30):
            row = {"endpoint_key": ep}
            for c in red_cols + svc_cols:
                row[c] = float(rng.normal(10.0, 2.0))
            rows.append(row)
    stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    stats.fit(pd.DataFrame(rows))
    return stats


def _fusion(endpoints=("epA", "epB")) -> ReliabilityGatedFusion:
    stats = _fitted_stats(endpoints)
    id_to_key = {i: ep for i, ep in enumerate(endpoints)}
    return ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key=id_to_key,
        branch_dim=4,
    )


def test_output_dim_equals_branch_dim():
    fusion = _fusion()
    assert fusion.output_dim == 4


def test_forward_shape_with_endpoint_id():
    fusion = _fusion()
    batch = {
        "endpoint_red": torch.randn(5, 3),
        "service_metric": torch.randn(5, 2),
        "service_log": torch.randn(5, 1),
    }
    endpoint_id = torch.tensor([0, 1, 0, 1, 0])
    out = fusion(batch, endpoint_id=endpoint_id)
    assert out.shape == (5, 4)


def test_missing_endpoint_id_defaults_to_uniform_weights():
    """endpoint_id=None 时退化为均匀权重 [0.5, 0.5]，不查表——这是向后兼容
    L0/L1/L2 调用方式（那些调用不传 endpoint_id）的关键行为，也是稀疏/未知
    endpoint 场景下的兜底。"""
    fusion = _fusion()
    fusion.eval()
    batch = {
        "endpoint_red": torch.randn(3, 3),
        "service_metric": torch.randn(3, 2),
        "service_log": torch.randn(3, 1),
    }
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id=None)
    torch.testing.assert_close(w_ep, torch.full((3,), 0.5))
    torch.testing.assert_close(w_svc, torch.full((3,), 0.5))


def test_gate_weights_sum_to_one():
    fusion = _fusion()
    batch = {
        "endpoint_red": torch.randn(8, 3),
        "service_metric": torch.randn(8, 2),
        "service_log": torch.randn(8, 1),
    }
    endpoint_id = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id=endpoint_id)
    torch.testing.assert_close(w_ep + w_svc, torch.ones(8))
    assert (w_ep >= 0).all() and (w_ep <= 1).all()


def test_missing_modality_key_raises():
    stats = _fitted_stats()
    with pytest.raises(ValueError):
        ReliabilityGatedFusion(
            modality_dims={"endpoint_red": 3, "service_metric": 2},
            endpoint_baseline_stats=stats,
            id_to_endpoint_key={0: "epA", 1: "epB"},
            branch_dim=4,
        )


def test_extra_modality_key_raises():
    stats = _fitted_stats()
    with pytest.raises(ValueError):
        ReliabilityGatedFusion(
            modality_dims={**MODALITY_DIMS, "extra": 2},
            endpoint_baseline_stats=stats,
            id_to_endpoint_key={0: "epA", 1: "epB"},
            branch_dim=4,
        )


def test_unknown_endpoint_id_in_batch_raises():
    """batch 里出现 fit 时未映射到任何 endpoint_key 的 id，必须显式报错——
    与 EndpointBaselineStats.branch_stats 对未知 endpoint 报 KeyError 的策略
    一致，不能在这一层被静默吞掉退化成均匀权重（那会和"故意不传 endpoint_id"
    的合法退化路径混淆，掩盖真正的数据契约错误）。"""
    fusion = _fusion()
    batch = {
        "endpoint_red": torch.randn(1, 3),
        "service_metric": torch.randn(1, 2),
        "service_log": torch.randn(1, 1),
    }
    with pytest.raises(KeyError):
        fusion(batch, endpoint_id=torch.tensor([99]))


def test_scalar_deviation_mode_uses_1d_input_to_gate():
    """消融1：dev 换成'总偏离幅度标量'（去掉分支区分）——gate_mlp 输入应从 6 维
    降到 2 维（每分支 1 个标量：总 z-score 范数），验证'该信哪个分支'这个能力
    被移除后 gate_mlp 的实际输入维度确实变了（不是只加了个没用的开关）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"},
        branch_dim=4,
        deviation_mode="scalar",
    )
    assert fusion.gate_mlp[0].in_features == 2


def test_independent_sigmoid_mode_weights_do_not_sum_to_one():
    """消融2：softmax 换两个独立 sigmoid——权重和不再恒为1，验证'竞争性归一化'
    确实被换成了'独立开关'语义（否则消融变体和原实现在数值上无法区分）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"},
        branch_dim=4,
        gate_normalization="independent_sigmoid",
    )
    batch = {
        "endpoint_red": torch.randn(6, 3),
        "service_metric": torch.randn(6, 2),
        "service_log": torch.randn(6, 1),
    }
    endpoint_id = torch.tensor([0, 1, 0, 1, 0, 1])
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id)
    assert not torch.allclose(w_ep + w_svc, torch.ones(6))


def test_fixed_uniform_gate_disables_learning():
    """消融3：固定 [0.5, 0.5]（禁用门控）——下界对照，gate_mlp 应不存在
    可训练参数参与该路径（权重恒定，与输入无关）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"},
        branch_dim=4,
        gate_normalization="fixed_uniform",
    )
    batch = {
        "endpoint_red": torch.randn(4, 3) * 100,
        "service_metric": torch.randn(4, 2) * 100,
        "service_log": torch.randn(4, 1) * 100,
    }
    endpoint_id = torch.tensor([0, 1, 0, 1])
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id)
    torch.testing.assert_close(w_ep, torch.full((4,), 0.5))
    torch.testing.assert_close(w_svc, torch.full((4,), 0.5))
