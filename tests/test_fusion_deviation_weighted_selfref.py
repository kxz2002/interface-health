import time
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.fusion.base import MODALITY_ORDER
from src.fusion.deviation_weighted_selfref import SelfReferentialDeviationFusion

CONFIGS_DIR = str(Path(__file__).parents[1] / "configs")

MODALITY_DIMS = {"endpoint_red": 3, "service_metric": 2, "service_log": 1}


def _batch(batch_size: int, dims: dict[str, int] | None = None, seed: int = 0):
    dims = dims or MODALITY_DIMS
    g = torch.Generator().manual_seed(seed)
    return {m: torch.randn(batch_size, dims[m], generator=g) for m in MODALITY_ORDER}


def test_weighted_formula_matches_manual_sigmoid():
    """逐值手算 out = sigmoid((|x| - threshold) / scale) * x——与
    DeviationWeightedFusion 公式完全相同，区别只在 x 本身就是 v2 per-case
    z-score 归一化后的偏离量，不再查 EndpointBaselineStats sidecar。"""
    fusion = SelfReferentialDeviationFusion(MODALITY_DIMS, threshold=2.0, scale=1.0)
    fusion.eval()
    batch = _batch(4, seed=1)
    out = fusion(batch)

    x = torch.cat([batch[m] for m in MODALITY_ORDER], dim=-1)
    expected = torch.sigmoid((x.abs() - 2.0) / 1.0) * x
    torch.testing.assert_close(out, expected)
    # 再用 numpy 独立算一遍，防 torch 公式抄自己。
    w = 1.0 / (1.0 + np.exp(-(x.abs().numpy() - 2.0)))
    np.testing.assert_allclose(out.numpy(), x.numpy() * w, rtol=1e-6)


def test_construct_with_only_modality_dims_no_sidecar():
    """无 sidecar 构造：只给 modality_dims 即可实例化——不需要 contract_dir、
    EndpointBaselineStats、id_to_endpoint_key、red_cols/svc_cols（v2 per-case
    归一化消掉了 entry 018 的全部 sidecar 耦合）。configs/fusion 下的 yaml 经
    Hydra 实例化也只吃 modality_dims 一个运行时 kwarg。"""
    fusion = SelfReferentialDeviationFusion(MODALITY_DIMS)
    assert isinstance(fusion, SelfReferentialDeviationFusion)

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/deviation_weighted_selfref")
    fusion_hydra = hydra.utils.instantiate(cfg.fusion, modality_dims=MODALITY_DIMS)
    assert isinstance(fusion_hydra, SelfReferentialDeviationFusion)
    assert fusion_hydra.output_dim == 6


def test_endpoint_id_none_equals_tensor():
    """endpoint_id 传 None 与传张量结果必须相同——本类不使用任何 endpoint 信息，
    权重只依赖特征值自身（与旧 DWF「None 退化为纯 concat」的语义不同：旧类的
    参照系是 per-endpoint sidecar，本类的参照系是特征自身，endpoint 维度恒无关）。"""
    fusion = SelfReferentialDeviationFusion(MODALITY_DIMS)
    fusion.eval()
    batch = _batch(5, seed=2)
    endpoint_id = torch.tensor([0, 7, 3, 99, 1])

    out_none = fusion(batch, endpoint_id=None)
    out_tensor = fusion(batch, endpoint_id=endpoint_id)
    torch.testing.assert_close(out_none, out_tensor)


def test_output_dim_equals_ep_plus_svc_dim():
    dims = {"endpoint_red": 4, "service_metric": 5, "service_log": 6}
    fusion = SelfReferentialDeviationFusion(dims)
    assert fusion.output_dim == 4 + 5 + 6
    batch = _batch(3, dims=dims)
    assert fusion(batch).shape == (3, 15)


def test_gradient_flows_through_to_downstream_and_input():
    """本类零可学习参数，但梯度必须能穿过它回传：① 下游 SVDD（此处用等价的
    Linear 头代替）的参数拿到非零梯度；② 输入特征本身也能收到梯度。entry 012
    的同源教训是 optimizer 漏接 fusion 子模块——零参数类不存在漏接问题，但
    forward 里若误用 .detach()/numpy/原地破坏图的写法，下游梯度会被静默截断。"""
    fusion = SelfReferentialDeviationFusion(MODALITY_DIMS)
    assert sum(p.numel() for p in fusion.parameters()) == 0

    head = torch.nn.Linear(fusion.output_dim, 1)
    batch = _batch(8, seed=3)
    for t in batch.values():
        t.requires_grad_(True)

    loss = head(fusion(batch, endpoint_id=torch.arange(8))).pow(2).mean()
    loss.backward()

    assert head.weight.grad is not None
    assert torch.count_nonzero(head.weight.grad) > 0
    for m in MODALITY_ORDER:
        assert batch[m].grad is not None
        assert torch.count_nonzero(batch[m].grad) > 0


def test_forward_is_vectorized_no_per_sample_python_loop():
    """大 batch 下不得有 per-sample Python 循环（旧 DWF forward 逐行查
    EndpointBaselineStats，v2 类应整批向量化）。shape 断言 + 计时兜底：
    与显式逐行 Python 参考实现比加速比，阈值故意放宽（≥20x 且绝对时 <1s），
    防 CI 共享机抖动误红，而真写成逐行循环会慢约两个数量级、必然触红。"""
    dims = {"endpoint_red": 64, "service_metric": 32, "service_log": 32}
    fusion = SelfReferentialDeviationFusion(dims)
    fusion.eval()
    batch = _batch(16384, dims=dims, seed=4)
    x = torch.cat([batch[m] for m in MODALITY_ORDER], dim=-1)

    with torch.no_grad():
        fusion(batch)  # warmup
        t0 = time.perf_counter()
        out = fusion(batch)
        t_vec = time.perf_counter() - t0
    assert out.shape == (16384, 128)

    def _python_loop_reference(x: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in range(x.shape[0]):
            rows.append(torch.sigmoid((x[i].abs() - 2.0) / 1.0) * x[i])
        return torch.stack(rows)

    with torch.no_grad():
        _python_loop_reference(x[:128])  # warmup
        t0 = time.perf_counter()
        ref = _python_loop_reference(x)
        t_loop = time.perf_counter() - t0
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-6)

    assert t_vec < 1.0, f"向量化 forward 耗时 {t_vec:.3f}s，疑似存在 per-sample Python 循环"
    assert t_loop > 20 * t_vec, (
        f"向量化仅比逐行 Python 循环快 {t_loop / max(t_vec, 1e-9):.1f}x，"
        "疑似 forward 退化为 per-sample 循环"
    )


def test_invalid_modality_keys_raise():
    with pytest.raises(ValueError):
        SelfReferentialDeviationFusion({"endpoint_red": 3, "service_metric": 2})
    with pytest.raises(ValueError):
        SelfReferentialDeviationFusion({**MODALITY_DIMS, "extra": 1})
