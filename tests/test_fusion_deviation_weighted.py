import numpy as np
import pandas as pd
import pytest
import torch

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.deviation_weighted import DeviationWeightedFusion

MODALITY_DIMS = {"endpoint_red": 3, "service_metric": 2, "service_log": 1}
RED_COLS = [f"endpoint_red__f{i}" for i in range(3)]
SVC_COLS = [f"service_metric__g{i}" for i in range(2)] + ["service_log__h0"]


def _fitted_stats(endpoints=("epA", "epB")) -> EndpointBaselineStats:
    rng = np.random.default_rng(0)
    rows = []
    for ep in endpoints:
        for _ in range(30):
            row = {"endpoint_key": ep}
            for c in RED_COLS + SVC_COLS:
                row[c] = float(rng.normal(10.0, 2.0))
            rows.append(row)
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(pd.DataFrame(rows))
    return stats


def _fusion(endpoints=("epA", "epB"), **kwargs) -> DeviationWeightedFusion:
    stats = _fitted_stats(endpoints)
    id_to_key = {i: ep for i, ep in enumerate(endpoints)}
    return DeviationWeightedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key=id_to_key,
        red_cols=RED_COLS,
        svc_cols=SVC_COLS,
        **kwargs,
    )


def test_zero_learnable_parameters():
    """零可学习参数是本类相对 L0 的唯一自变量（module docstring / spec §2.2 /
    plan 背景说明）——若未来有人误加 nn.Linear/gate_mlp，必须在这里立刻报错，
    而不是让实验结果悄悄失去单变量对比的有效性。"""
    fusion = _fusion()
    assert sum(p.numel() for p in fusion.parameters()) == 0


def test_output_dim_equals_ep_plus_svc_dim():
    fusion = _fusion()
    assert fusion.output_dim == 6  # 3(ep) + 2(svc_metric) + 1(svc_log)


def test_forward_shape_with_endpoint_id():
    fusion = _fusion()
    batch = {
        "endpoint_red": torch.randn(5, 3),
        "service_metric": torch.randn(5, 2),
        "service_log": torch.randn(5, 1),
    }
    endpoint_id = torch.tensor([0, 1, 0, 1, 0])
    out = fusion(batch, endpoint_id=endpoint_id)
    assert out.shape == (5, 6)


def test_weighted_formula_matches_manual_zscore_sigmoid():
    """手算验证 weighted = feature * sigmoid(|z| - 2.0)，z=(feature-mean)/std——
    逐特征、非分支级，且两分支公式一致（同一个 threshold=2.0，复用 RG 的既有
    显著性标准，见 spec §2.3）。"""
    fusion = _fusion()
    fusion.eval()
    stats = fusion._baseline
    ep_mean, ep_std = stats.branch_stats("epA", "ep")
    svc_mean, svc_std = stats.branch_stats("epA", "svc")

    raw_ep = torch.tensor([[15.0, 15.0, 15.0]])
    raw_metric = torch.tensor([[15.0, 15.0]])
    raw_log = torch.tensor([[15.0]])
    batch = {
        "endpoint_red": raw_ep,
        "service_metric": raw_metric,
        "service_log": raw_log,
    }
    out = fusion(batch, endpoint_id=torch.tensor([0]))

    raw_svc = np.array([15.0, 15.0, 15.0])
    z_ep = (raw_ep[0].numpy() - ep_mean) / ep_std
    z_svc = (raw_svc - svc_mean) / svc_std
    w_ep = 1.0 / (1.0 + np.exp(-(np.abs(z_ep) - 2.0)))
    w_svc = 1.0 / (1.0 + np.exp(-(np.abs(z_svc) - 2.0)))
    expected = np.concatenate([raw_ep[0].numpy() * w_ep, raw_svc * w_svc])
    np.testing.assert_allclose(out[0].numpy(), expected, rtol=1e-5)


def test_degenerate_column_weight_fixed_to_one():
    """f0 在整个 fit 集合上恒为 5.0（零方差，触发全局退化 fallback，
    std 兜底为 1e-9）。eval 时 f0 仍恰好等于均值 5.0（该列未真正偏离）——
    若未特殊处理，naive z=(5.0-5.0)/1e-9=0，sigmoid(0-2.0)≈0.119 会错误地
    把这个合法值衰减；退化列权重固定为1的处理应让该列输出严格等于输入，
    不受人为兜底 std 影响（spec §2.4）。"""
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(30):
        row = {"endpoint_key": "epA", "endpoint_red__f0": 5.0}
        for c in RED_COLS[1:] + SVC_COLS:
            row[c] = float(rng.normal(10.0, 2.0))
        rows.append(row)
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(pd.DataFrame(rows))
    assert "endpoint_red__f0" in stats.degenerate_columns("ep")

    fusion = DeviationWeightedFusion(
        modality_dims=MODALITY_DIMS,
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA"},
        red_cols=RED_COLS,
        svc_cols=SVC_COLS,
    )
    fusion.eval()
    batch = {
        "endpoint_red": torch.tensor([[5.0, 10.0, 10.0]]),
        "service_metric": torch.tensor([[10.0, 10.0]]),
        "service_log": torch.tensor([[10.0]]),
    }
    out = fusion(batch, endpoint_id=torch.tensor([0]))
    assert out[0, 0].item() == pytest.approx(5.0)


def test_missing_endpoint_id_degrades_to_plain_concat():
    """endpoint_id=None 时全部特征权重恒为1，退化为纯 concat——等价 L0
    EarlyConcatFusion 的行为（spec §2.5），是向后兼容 L0/L1/L2 调用方式的
    兜底路径。"""
    fusion = _fusion()
    fusion.eval()
    batch = {
        "endpoint_red": torch.randn(3, 3),
        "service_metric": torch.randn(3, 2),
        "service_log": torch.randn(3, 1),
    }
    out = fusion(batch, endpoint_id=None)
    expected = torch.cat(
        [batch["endpoint_red"], batch["service_metric"], batch["service_log"]], dim=-1
    )
    torch.testing.assert_close(out, expected)


def test_missing_modality_key_raises():
    stats = _fitted_stats()
    with pytest.raises(ValueError):
        DeviationWeightedFusion(
            modality_dims={"endpoint_red": 3, "service_metric": 2},
            endpoint_baseline_stats=stats,
            id_to_endpoint_key={0: "epA", 1: "epB"},
            red_cols=RED_COLS,
            svc_cols=SVC_COLS,
        )


def test_extra_modality_key_raises():
    stats = _fitted_stats()
    with pytest.raises(ValueError):
        DeviationWeightedFusion(
            modality_dims={**MODALITY_DIMS, "extra": 2},
            endpoint_baseline_stats=stats,
            id_to_endpoint_key={0: "epA", 1: "epB"},
            red_cols=RED_COLS,
            svc_cols=SVC_COLS,
        )


def test_unknown_endpoint_id_in_batch_raises():
    """batch 里出现 fit 时未映射到任何 endpoint_key 的 id，必须显式报错——
    与 EndpointBaselineStats.branch_stats 对未知 endpoint 报 KeyError 的策略
    一致（与 RG 同款约定），不能静默退化成全1权重掩盖数据契约错误。"""
    fusion = _fusion()
    batch = {
        "endpoint_red": torch.randn(1, 3),
        "service_metric": torch.randn(1, 2),
        "service_log": torch.randn(1, 1),
    }
    with pytest.raises(KeyError):
        fusion(batch, endpoint_id=torch.tensor([99]))
