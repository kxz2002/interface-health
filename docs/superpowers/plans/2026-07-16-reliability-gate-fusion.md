# Deviation-Conditioned Reliability Gate Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 reliability-gate 融合机制——一个新的 `FusionModule` 子类 `ReliabilityGatedFusion`，其门控输入是"两分支各自相对自身 normal 基线的偏离摘要"而非原始特征，softmax 竞争性输出 `[w_ep, w_svc]` 决定该信哪个分支；配套新增 `EndpointBaselineStats` 组件产出偏离量计算所需的 per-endpoint 双分支 normal-only mean/std；同时扩容训练池（吸收故障 case 的 baseline 阶段行，838→~7359 行）并同步收紧 eval 防泄漏。跑通 Contract v1 训练/评估，产出与 L0/L1/L2 baseline 的多 seed AUROC/AUPRC 对比 + 2×2 训练池归因 + 路由消融。

**Architecture:**
- `EndpointBaselineStats`（新增，`src/data/endpoint_baseline_stats.py`）：`fit(df)` 只在 `train_fit` 上跑，按 `endpoint_key` 分组算 RED 分支（10维 `endpoint_red__*`）与 SVC 分支（8维 `service_metric__*`+`service_log__*`）各自的 mean/std；退化 group（全 NaN/零方差）延续 `Normalizer._is_degenerate` 同款判定跳过；稀疏 endpoint 用 shrinkage-to-global 兜底（可开关）；`save/load` JSON sidecar，不 bake 进主 parquet。
- `ReliabilityGatedFusion`（新增，`src/fusion/reliability_gate.py`，继承 `FusionModule`）：`forward(modality_dict, endpoint_id=None)`——用 `EndpointBaselineStats` 算 `dev_ep`/`dev_svc`（每分支2~3维：z-score范数+最大单特征z-score+偏离特征占比）→ `softmax(MLP([dev_ep;dev_svc]))` 得 `[w_ep, w_svc]` → `e_ep=enc_ep(raw_ep)`/`e_svc=enc_svc(raw_svc)` → `z = w_ep*value_ep(e_ep) + w_svc*value_svc(e_svc)`。`endpoint_id` 缺省时退化为均匀权重（不查表，向后兼容）。
- 管线接入（改动而非新增）：`build_contract.py` 新增 `endpoint_id` int 列（8个endpoint字符串→整数确定性映射）+ `train_pool_mask`（normal_mask | baseline行）+ `source_phase` 可审计列 + 调用 `EndpointBaselineStats.fit(train_fit).save(...)`；`REQUIRED_ID_COLUMNS` 追加 `endpoint_id`；`contract_dataloader.py::_row_to_sample` 的 meta 追加 `endpoint_id`；`train_baseline_v0.py::_collate` 追加堆叠 `endpoint_id`，训练/推理循环把它传给 `fusion(...)`。
- 新增 Hydra config：`configs/fusion/reliability_gate.yaml`。

**Tech Stack:** PyTorch (`nn.Module`), Hydra (`hydra.utils.instantiate`), pandas, pytest, conda env `interface`, DVC

**Spec:** `docs/superpowers/specs/2026-07-16-reliability-gate-fusion-design.md`

---

## 背景说明（供实施者理解代码库约定，执行前必读）

- **前置 bug 已修复**：spec §6 风险1点名的 `endpoint_red__client_latency_p95`/`latency_divergence` 等列 1e9~1e13 异常值已在 PR#13（history entry 013）修复——`Normalizer._is_degenerate` 判定条件从"仅NaN"扩展到"NaN或零方差"，退化 group 统一跳过归一化保留原值。当前 `artifacts/contract_v1/` 已是修复后重新生成的产物，可放心使用。**本计划的偏离量计算（z-score）依赖这份修复**：若 `EndpointBaselineStats` 的 mean/std 统计也遇到同款零方差退化（某 endpoint 某分支列在 `train_fit` 里只有单一取值，std=0），必须复用同样的"跳过"哲学（下方 Task 1 会显式处理，不是重新发明）。
- 所有 fusion 模块继承 `src/fusion/base.py` 的 `FusionModule` 抽象类：必须实现 `forward(modality_dict: dict[str, torch.Tensor]) -> torch.Tensor` 和 `output_dim` property。`MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")` 是模块级常量。给 `forward` 增加可选 `endpoint_id` 参数对现有 L0/L1/L2 是向后兼容的加法式改动（Python 允许子类签名比抽象方法少参数，调用方不传该参数走默认值）——已核对 L0/L1/L2 三个现有实现的 `forward` 签名完全一致（只接收 `modality_dict`），不会被这个改动破坏。
- `Normalizer`（`src/data/normalization.py`）现有实现按 `endpoint_key`/`service_name` groupby 算 **min/max**（用于线性归一化），JSON sidecar 模式（`save`/`load`），退化判定 `_is_degenerate(lo, hi)`。`EndpointBaselineStats` 是新写的类，可参考其 groupby-by-endpoint + sidecar 设计模式，但统计量语义不同（mean/std，非min/max），且需要 RED/SVC 双分支拆分（Normalizer 逐列独立，没有"分支"概念）——不能直接复用 `Normalizer` 类本身。
- Contract v1 现有列名约定：RED 分支特征列 = `endpoint_red__trace_request_count/latency_p50/latency_p95/error_rate/5xx_rate/client_request_count/client_latency_p95/client_error_rate/client_5xx_rate/latency_divergence`（10列，见 `configs/contract/v1.yaml`）；SVC 分支 = `service_metric__cpu_usage_rate/memory_usage_ratio/net_rx_error_rate/net_tx_error_rate/process_count`（5列）+ `service_log__event_rate/error_ratio/template_diversity`（3列），共8列。8 个 endpoint 的规范列表来自 `configs/contract/endpoint_to_service.yaml` 的 key 集合（`sorted()` 后得到确定性顺序，用于 `endpoint_id` 映射）。
- 已用 mini fixture 实测跑通 `build_contract.py --config v1.yaml`：真实 `merged_v2.yaml` 数据的 `artifacts/contract_v1/train_fit.parquet` 现有 838 行，`eval_all.parquet` 现有 13632 行（`anomaly_type` 覆盖 27 种故障类型 + Normal，`phase` 分布 inject=6875/baseline=5411/recover=1110/normal=236）。Task 6 的训练池扩容会把 `baseline` 的 5411 行吸收进训练池，`eval_all` 需同步收紧排除这些行，具体见 Task 6。

---

## Task 1: `EndpointBaselineStats`（per-endpoint 双分支 normal-only mean/std）

**Files:**
- Create: `src/data/endpoint_baseline_stats.py`
- Test: `tests/test_endpoint_baseline_stats.py`

**背景**：spec 组件1。这是 reliability gate 计算 `dev_ep`/`dev_svc` 偏离量的统计量来源——每个 endpoint 一份，分别记录 RED 分支10列和 SVC 分支8列的 mean/std，只在 `train_fit` 上拟合（防泄漏纪律与 `Normalizer` 一致），拟合后固定、不参与 backprop。稀疏 endpoint（如 `POST:/api/v1/users/login` 在真实数据里只有17行）std 估计不稳，用 shrinkage-to-global 兜底：`std_shrunk = (n/(n+k)) * std_local + (k/(n+k)) * std_global`，`k` 为可调超参（先设默认值10，无强理论依据，作为消融维度）。

- [ ] **Step 1: 写失败的测试**

`tests/test_endpoint_baseline_stats.py`：
```python
import numpy as np
import pandas as pd
import pytest

from src.data.endpoint_baseline_stats import EndpointBaselineStats

RED_COLS = [f"endpoint_red__f{i}" for i in range(3)]
SVC_COLS = [f"service_metric__g{i}" for i in range(2)]


def _synth_df(n_per_ep: dict[str, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ep, n in n_per_ep.items():
        for _ in range(n):
            row = {"endpoint_key": ep}
            for c in RED_COLS + SVC_COLS:
                row[c] = float(rng.normal(loc=10.0, scale=2.0))
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_produces_mean_std_per_endpoint_per_branch():
    df = _synth_df({"ep1": 50, "ep2": 50})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    for ep in ("ep1", "ep2"):
        ep_mean, ep_std = stats.branch_stats(ep, "ep")
        svc_mean, svc_std = stats.branch_stats(ep, "svc")
        assert ep_mean.shape == (len(RED_COLS),)
        assert ep_std.shape == (len(RED_COLS),)
        assert svc_mean.shape == (len(SVC_COLS),)
        assert svc_std.shape == (len(SVC_COLS),)
        assert (ep_std > 0).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_endpoint_baseline_stats.py -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'src.data.endpoint_baseline_stats'`

- [ ] **Step 3: 补充退化 group / shrinkage / 未知 endpoint 三个失败测试**

追加到 `tests/test_endpoint_baseline_stats.py`：
```python
def test_degenerate_group_zero_std_falls_back_to_global_std():
    """某 endpoint 某列在 fit 集合里零方差（std=0），复用 Normalizer 同款
    退化哲学：不能直接拿 std=0 去做 z-score（除零），必须退化处理——这里退化
    策略是整列直接用 global std（等价于 shrinkage k→inf 的极限），而非跳过该列
    （偏离量计算不能有 NaN 分量，跳过会破坏 dev 向量的固定维度）。"""
    df = _synth_df({"ep-normal": 50})
    degenerate_col = RED_COLS[0]
    df2 = pd.concat([df, _synth_df({"ep-degenerate": 5}, seed=1)], ignore_index=True)
    df2.loc[df2["endpoint_key"] == "ep-degenerate", degenerate_col] = 7.0  # 常数

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df2)
    ep_mean, ep_std = stats.branch_stats("ep-degenerate", "ep")
    idx = RED_COLS.index(degenerate_col)
    global_std = df2[degenerate_col].std()
    assert ep_std[idx] == pytest.approx(global_std, rel=1e-6)


def test_shrinkage_pulls_sparse_endpoint_toward_global():
    """稀疏 endpoint（n 远小于 shrinkage_k）的 std 应明显偏向 global std，
    而非纯局部估计——防止小样本方差估计噪声主导偏离量计算。"""
    df = _synth_df({"ep-dense": 500, "ep-sparse": 3}, seed=2)
    stats = EndpointBaselineStats(
        red_cols=RED_COLS, svc_cols=SVC_COLS, shrinkage_k=10, use_shrinkage=True
    )
    stats.fit(df)
    _, sparse_std = stats.branch_stats("ep-sparse", "ep")
    _, dense_std = stats.branch_stats("ep-dense", "ep")
    global_std = df[RED_COLS[0]].std()
    idx = 0
    # 稀疏 endpoint 的 shrunk std 应比 dense endpoint 的局部 std 更接近 global_std
    assert abs(sparse_std[idx] - global_std) < abs(dense_std[idx] - global_std)


def test_shrinkage_disabled_uses_pure_local_stats():
    df = _synth_df({"ep-sparse": 3}, seed=3)
    stats_on = EndpointBaselineStats(
        red_cols=RED_COLS, svc_cols=SVC_COLS, shrinkage_k=10, use_shrinkage=True
    )
    stats_off = EndpointBaselineStats(
        red_cols=RED_COLS, svc_cols=SVC_COLS, use_shrinkage=False
    )
    stats_on.fit(df)
    stats_off.fit(df)
    _, std_on = stats_on.branch_stats("ep-sparse", "ep")
    _, std_off = stats_off.branch_stats("ep-sparse", "ep")
    local_std = df[RED_COLS[0]].std()
    assert std_off[0] == pytest.approx(local_std, rel=1e-6)
    assert std_on[0] != pytest.approx(local_std, rel=1e-3)  # shrinkage 应改变取值


def test_unknown_endpoint_raises_key_error():
    """fit 时未见过的 endpoint_id 查表必须显式报错，不能静默返回 global 统计量——
    调用方（ReliabilityGatedFusion）需要知道这是数据契约错误还是正常的未知 endpoint
    退化路径，二者语义不同，不该在这一层混淆。"""
    df = _synth_df({"ep1": 50})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    with pytest.raises(KeyError):
        stats.branch_stats("ep-never-seen", "ep")


def test_save_load_roundtrip(tmp_path):
    df = _synth_df({"ep1": 50, "ep2": 30})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    path = tmp_path / "endpoint_baseline_stats.json"
    stats.save(path)
    loaded = EndpointBaselineStats.load(path)
    for ep in ("ep1", "ep2"):
        for branch in ("ep", "svc"):
            m1, s1 = stats.branch_stats(ep, branch)
            m2, s2 = loaded.branch_stats(ep, branch)
            np.testing.assert_allclose(m1, m2)
            np.testing.assert_allclose(s1, s2)
```

- [ ] **Step 4: 跑测试确认全部失败**

Run: `conda run -n interface python -m pytest tests/test_endpoint_baseline_stats.py -v`
Expected: FAIL（同 Step 2 的 ModuleNotFoundError，7 个测试全部收集失败）

- [ ] **Step 5: 写 `src/data/endpoint_baseline_stats.py`**

```python
"""每 endpoint 一份的 RED/SVC 双分支 normal-only mean/std 统计量。

供 ReliabilityGatedFusion 计算偏离量（z-score）使用。只在 train_fit 上 fit，
拟合后固定、不参与 backprop——与 Normalizer 的防泄漏纪律一致（build_contract.py
的 fit_df 收窄到 train_fit 约定，见该文件 L369-381 的注释）。

退化处理与 src/data/normalization.py::_is_degenerate 同一哲学但落点不同：
Normalizer 遇到零方差退化时"跳过归一化保留原值"；这里偏离量计算不允许 NaN
分量（会破坏 dev 向量固定维度、传播进 softmax），故退化列改为直接退化到
global std（等价于 shrinkage k→inf 的极限），而非跳过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Branch = Literal["ep", "svc"]

_DEGENERATE_STD_EPS = 1e-9


@dataclass
class _EndpointBranchStats:
    mean: np.ndarray
    std: np.ndarray


class EndpointBaselineStats:
    def __init__(
        self,
        red_cols: list[str],
        svc_cols: list[str],
        shrinkage_k: float = 10.0,
        use_shrinkage: bool = True,
    ):
        self._red_cols = list(red_cols)
        self._svc_cols = list(svc_cols)
        self._shrinkage_k = shrinkage_k
        self._use_shrinkage = use_shrinkage
        self._stats: dict[str, dict[Branch, _EndpointBranchStats]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        branch_cols: dict[Branch, list[str]] = {"ep": self._red_cols, "svc": self._svc_cols}
        global_stats = {
            branch: (df[cols].mean().to_numpy(), df[cols].std(ddof=0).to_numpy())
            for branch, cols in branch_cols.items()
        }

        self._stats = {}
        for ep, group in df.groupby("endpoint_key"):
            n = len(group)
            per_branch: dict[Branch, _EndpointBranchStats] = {}
            for branch, cols in branch_cols.items():
                local_mean = group[cols].mean().to_numpy()
                local_std = group[cols].std(ddof=0).to_numpy()
                g_mean, g_std = global_stats[branch]

                degenerate = np.isnan(local_std) | (local_std < _DEGENERATE_STD_EPS)
                std = local_std.copy()
                mean = local_mean.copy()
                # 退化列（全 NaN / 零方差）直接退化到 global 统计量，
                # 不参与 shrinkage 混合（shrinkage 对一个未定义的局部估计值取
                # 加权平均没有意义），也不保留 NaN（下游 z-score 计算会传染 NaN）。
                std[degenerate] = g_std[degenerate]
                mean[degenerate] = np.where(
                    np.isnan(mean[degenerate]), g_mean[degenerate], mean[degenerate]
                )

                if self._use_shrinkage:
                    w = n / (n + self._shrinkage_k)
                    non_degenerate = ~degenerate
                    std[non_degenerate] = (
                        w * std[non_degenerate] + (1 - w) * g_std[non_degenerate]
                    )
                    mean[non_degenerate] = (
                        w * mean[non_degenerate] + (1 - w) * g_mean[non_degenerate]
                    )

                per_branch[branch] = _EndpointBranchStats(mean=mean, std=std)
            self._stats[str(ep)] = per_branch

    def branch_stats(self, endpoint_key: str, branch: Branch) -> tuple[np.ndarray, np.ndarray]:
        if endpoint_key not in self._stats:
            raise KeyError(
                f"endpoint_key={endpoint_key!r} 未出现在 fit 集合（train_fit）中，"
                "无法计算偏离量——检查上游 endpoint_id 映射是否与 fit 时一致"
            )
        s = self._stats[endpoint_key][branch]
        return s.mean, s.std

    def save(self, path: str | Path) -> None:
        payload = {
            "red_cols": self._red_cols,
            "svc_cols": self._svc_cols,
            "shrinkage_k": self._shrinkage_k,
            "use_shrinkage": self._use_shrinkage,
            "stats": {
                ep: {
                    branch: {"mean": s.mean.tolist(), "std": s.std.tolist()}
                    for branch, s in per_branch.items()
                }
                for ep, per_branch in self._stats.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EndpointBaselineStats":
        raw = json.loads(Path(path).read_text())
        obj = cls(
            red_cols=raw["red_cols"],
            svc_cols=raw["svc_cols"],
            shrinkage_k=raw["shrinkage_k"],
            use_shrinkage=raw["use_shrinkage"],
        )
        obj._stats = {
            ep: {
                branch: _EndpointBranchStats(
                    mean=np.array(s["mean"]), std=np.array(s["std"])
                )
                for branch, s in per_branch.items()
            }
            for ep, per_branch in raw["stats"].items()
        }
        return obj
```

- [ ] **Step 6: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_endpoint_baseline_stats.py -v`
Expected: PASS（7 个测试全绿）

- [ ] **Step 7: Commit**

```bash
git add src/data/endpoint_baseline_stats.py tests/test_endpoint_baseline_stats.py
git commit -m "[Feature]: 新增 EndpointBaselineStats，per-endpoint 双分支 normal-only mean/std"
```

---

## Task 2: `ReliabilityGatedFusion`（核心融合模块）

**Files:**
- Create: `src/fusion/reliability_gate.py`
- Test: `tests/test_reliability_gated_fusion.py`

**背景**：spec 组件2 + §2.1 数据流。门控输入活在"异常度量空间"（两分支各自的偏离摘要 `dev_ep`/`dev_svc`），而非 L2 的原始特征空间——这是与 L2 `GatedFusion` 的核心架构区别（见 spec §2.2 对比表）。`endpoint_id` 用于查 `EndpointBaselineStats` 的 mean/std 表，缺省时退化为均匀权重 `[0.5, 0.5]`（不查表，向后兼容 L0/L1/L2 的调用方式）。

数据流：
```
raw_ep(10维)/raw_svc(8维)
  → z_ep = (raw_ep - mean_ep[endpoint_id]) / std_ep[endpoint_id]
  → dev_ep = [||z_ep||_2, max(|z_ep|), mean(|z_ep|>threshold)]   # 3维
  → 同理算 dev_svc（3维）
  → [w_ep, w_svc] = softmax(gate_mlp([dev_ep; dev_svc]))         # 竞争性权重
  → e_ep = ep_encoder(raw_ep); e_svc = svc_encoder(raw_svc)      # 沿用 L1 编码器
  → z = w_ep * value_ep(e_ep) + w_svc * value_svc(e_svc)
```

- [ ] **Step 1: 写失败的测试（基础形状 + 接口兼容 + softmax 性质）**

`tests/test_reliability_gated_fusion.py`：
```python
import numpy as np
import pytest
import torch

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.reliability_gate import ReliabilityGatedFusion

MODALITY_DIMS = {"endpoint_red": 3, "service_metric": 2, "service_log": 1}


def _fitted_stats(endpoints=("epA", "epB")) -> EndpointBaselineStats:
    import pandas as pd

    red_cols = ["endpoint_red__f0", "endpoint_red__f1", "endpoint_red__f2"]
    svc_cols = ["service_metric__g0", "service_metric__g1", "service_log__h0"]
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_reliability_gated_fusion.py -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'src.fusion.reliability_gate'`

- [ ] **Step 3: 补充剩余测试（缺省 endpoint_id 退化、softmax 性质、接口校验、手算公式）**

追加到 `tests/test_reliability_gated_fusion.py`：
```python
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
```

- [ ] **Step 4: 跑测试确认全部失败**

Run: `conda run -n interface python -m pytest tests/test_reliability_gated_fusion.py -v`
Expected: FAIL（同 Step 2）

- [ ] **Step 5: 写实现 `src/fusion/reliability_gate.py`**

```python
"""Deviation-Conditioned Reliability Gate Fusion（reliability-aware 模态路由门控）。

与 L2 GatedFusion 的核心区别：门控输入是"两分支各自相对自身 normal 基线的
偏离摘要"（dev_ep/dev_svc，活在异常度量空间），而非原始特征拼接（L2 活在
特征空间）；归一化用 softmax（竞争性二选一）而非逐维 sigmoid（独立开关）；
融合形式是对称加权 w_ep*value_ep + w_svc*value_svc，而非 L2 的非对称
e_ep + gate*value(e_svc)。详见 spec §2.2 对比表。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule

# |z|>此阈值算作"该特征显著偏离"，用于 dev 向量第3维（偏离特征占比）。
# 2.0 对应约 95% 正态分位数，无强理论依据，是消融维度（spec §4.3 未点名但可延伸）。
_DEVIATION_THRESHOLD = 2.0


class ReliabilityGatedFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        endpoint_baseline_stats: EndpointBaselineStats,
        id_to_endpoint_key: dict[int, str],
        branch_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._branch_dim = branch_dim
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        self.ep_encoder = nn.Sequential(nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU())
        self.svc_encoder = nn.Sequential(nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU())
        self.value_ep = nn.Linear(branch_dim, branch_dim)
        self.value_svc = nn.Linear(branch_dim, branch_dim)
        # dev_ep(3) + dev_svc(3) = 6 维输入，输出 2 维 logits 供 softmax。
        self.gate_mlp = nn.Sequential(
            nn.Linear(6, branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 2),
        )

    def _deviation_summary(self, raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = (raw - mean) / std
        norm = z.norm(dim=-1)
        max_abs = z.abs().max(dim=-1).values
        frac_exceed = (z.abs() > _DEVIATION_THRESHOLD).float().mean(dim=-1)
        return torch.stack([norm, max_abs, frac_exceed], dim=-1)

    def gate_weights(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = modality_dict["endpoint_red"].shape[0]
        if endpoint_id is None:
            half = torch.full((batch_size,), 0.5, device=modality_dict["endpoint_red"].device)
            return half, half

        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        dev_eps, dev_svcs = [], []
        for i in range(batch_size):
            eid = int(endpoint_id[i].item())
            if eid not in self._id_to_key:
                raise KeyError(f"endpoint_id={eid} 不在 id_to_endpoint_key 映射中")
            key = self._id_to_key[eid]
            ep_mean, ep_std = self._baseline.branch_stats(key, "ep")
            svc_mean, svc_std = self._baseline.branch_stats(key, "svc")
            device = modality_dict["endpoint_red"].device
            dev_eps.append(
                self._deviation_summary(
                    modality_dict["endpoint_red"][i],
                    torch.as_tensor(ep_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(ep_std, dtype=torch.float32, device=device),
                )
            )
            dev_svcs.append(
                self._deviation_summary(
                    svc_in[i],
                    torch.as_tensor(svc_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(svc_std, dtype=torch.float32, device=device),
                )
            )
        dev = torch.cat([torch.stack(dev_eps), torch.stack(dev_svcs)], dim=-1)
        w = torch.softmax(self.gate_mlp(dev), dim=-1)
        return w[:, 0], w[:, 1]

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        w_ep, w_svc = self.gate_weights(modality_dict, endpoint_id)
        e_ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        e_svc = self.svc_encoder(svc_in)
        return w_ep.unsqueeze(-1) * self.value_ep(e_ep) + w_svc.unsqueeze(-1) * self.value_svc(e_svc)

    @property
    def output_dim(self) -> int:
        return self._branch_dim
```

- [ ] **Step 6: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_reliability_gated_fusion.py -v`
Expected: PASS（7 个测试全绿）

- [ ] **Step 7: Commit**

```bash
git add src/fusion/reliability_gate.py tests/test_reliability_gated_fusion.py
git commit -m "[Feature]: 新增 ReliabilityGatedFusion，异常度量空间 softmax 路由门控"
```

---

## Task 3: `configs/fusion/reliability_gate.yaml` + Hydra instantiate 测试

**背景**：`ReliabilityGatedFusion` 比 L0/L1/L2 多两个运行时才知道的构造参数——`endpoint_baseline_stats`（Task 4 产出的 sidecar，训练脚本运行时加载）和 `id_to_endpoint_key`（同样运行时从 schema/sidecar 派生）。这两个不进 yaml（与 `modality_dims` 同类，属于"运行时关键字参数"，见现有 `configs/fusion/*.yaml` 只放 `_target_`+自身超参的惯例）。本 Task 只验证 `_target_`+`branch_dim`+`dropout` 三个静态超参能通过 Hydra 正确 instantiate，运行时参数用测试里手工构造的 fixture 补上（不依赖 Task 4/5 的管线改动，保持 Task 间独立可测）。

**Files:**
- Create: `configs/fusion/reliability_gate.yaml`
- Test: `tests/test_hydra_instantiate.py`（追加）

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_hydra_instantiate.py`：
```python
def test_fusion_reliability_gate_instantiate_returns_reliability_gated_fusion():
    from src.data.endpoint_baseline_stats import EndpointBaselineStats
    from src.fusion.reliability_gate import ReliabilityGatedFusion

    stats = EndpointBaselineStats(red_cols=[f"endpoint_red__f{i}" for i in range(10)],
                                   svc_cols=[f"service_metric__g{i}" for i in range(5)]
                                   + [f"service_log__h{i}" for i in range(3)])
    import pandas as pd

    rows = [{"endpoint_key": "epA", **{c: 1.0 for c in stats._red_cols + stats._svc_cols}}
            for _ in range(20)]
    stats.fit(pd.DataFrame(rows))

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/reliability_gate")
    fusion = hydra.utils.instantiate(
        cfg,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA"},
    )
    assert isinstance(fusion, ReliabilityGatedFusion)
    assert fusion.output_dim == 16  # yaml 里 branch_dim 默认值
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_hydra_instantiate.py -v -k reliability_gate`
Expected: FAIL——`hydra.errors.MissingConfigException: Cannot find primary config 'fusion/reliability_gate'`

- [ ] **Step 3: 写 config 文件**

`configs/fusion/reliability_gate.yaml`：
```yaml
_target_: src.fusion.reliability_gate.ReliabilityGatedFusion
branch_dim: 16
dropout: 0.1
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_hydra_instantiate.py -v -k reliability_gate`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/fusion/reliability_gate.yaml tests/test_hydra_instantiate.py
git commit -m "[Feature]: 新增 configs/fusion/reliability_gate.yaml，支持 hydra instantiate"
```

---

## Task 4: `build_contract.py` 新增 `endpoint_id` 列 + 落盘 `EndpointBaselineStats` sidecar

**背景**：spec 组件3.1/3.3。`endpoint_id` 是 8 个 endpoint 字符串→整数的确定性映射（`sorted(endpoint_to_service.keys())` 派生），baked 进主 parquet（不是 sidecar——它和 `endpoint_key` 一样是逐行天然存在的标识列，不是"per-endpoint 统计量"）。`EndpointBaselineStats` 则相反：统计量在 contract build 阶段算好存成 sidecar json，不 bake 进主 parquet 的行（spec §3.1 明确排除，8×N 小矩阵逐行复制违反数据规范化）。两者都只在 v1 路径新增，v0 路径不受影响。

**Files:**
- Modify: `src/contracts/contract_v0.py`（`REQUIRED_ID_COLUMNS` 追加 `endpoint_id`）
- Modify: `scripts/build_contract.py`
- Test: `tests/test_build_contract_v1_endpoint_id.py`（新建）

- [ ] **Step 1: 写失败的测试**

`tests/test_build_contract_v1_endpoint_id.py`：
```python
import json
import subprocess
import sys
import yaml
from pathlib import Path

import pandas as pd

from src.data.endpoint_baseline_stats import EndpointBaselineStats

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable, "scripts/build_contract.py",
            "--config", str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset", str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir", str(out_dir), "--seed", "42",
        ],
        check=True, cwd=str(REPO_ROOT),
    )


def test_endpoint_id_column_present_and_matches_sorted_mapping(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    expected_mapping = {ep: i for i, ep in enumerate(sorted(ep_to_svc.keys()))}

    for name in ("train_fit", "eval_all"):
        df = pd.read_parquet(out_dir / f"{name}.parquet")
        assert "endpoint_id" in df.columns
        for _, row in df.iterrows():
            assert row["endpoint_id"] == expected_mapping[row["endpoint_key"]]


def test_endpoint_baseline_stats_sidecar_written_and_loadable(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    sidecar = out_dir / "endpoint_baseline_stats.json"
    assert sidecar.exists()
    stats = EndpointBaselineStats.load(sidecar)

    # sidecar 覆盖的 endpoint 必须是 train_fit 里实际出现过的 endpoint_key 子集
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    fit_endpoints = set(train_fit["endpoint_key"].unique())
    assert set(stats._stats.keys()) == fit_endpoints


def test_endpoint_id_not_added_in_v0(tmp_path):
    """v0 路径不受影响——endpoint_id 是 v1 新增列，不应该悄悄出现在 v0 产物里
    （否则 v0 的既有契约测试的列集合断言会被意外改变行为）。"""
    out_dir = tmp_path / "contract_v0"
    subprocess.run(
        [
            sys.executable, "scripts/build_contract.py",
            "--config", str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset", str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir", str(out_dir), "--seed", "42",
        ],
        check=True, cwd=str(REPO_ROOT),
    )
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "endpoint_id" not in train.columns
    assert not (out_dir / "endpoint_baseline_stats.json").exists()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_build_contract_v1_endpoint_id.py -v`
Expected: FAIL——`endpoint_id` 列不存在，`endpoint_baseline_stats.json` 不存在

- [ ] **Step 3: `src/contracts/contract_v0.py` 追加 `endpoint_id`**

只对 v1 生效，故不能直接塞进 `REQUIRED_ID_COLUMNS`（v0 校验也会用到这个常量，会导致 v0 产物突然被要求有这一列）。改为新增独立常量：

```python
# v1 专属：ReliabilityGatedFusion 查 EndpointBaselineStats 用的整数 endpoint 标识，
# 不进 v0 校验（REQUIRED_ID_COLUMNS 保持不变，v0 产物无需这一列）。
REQUIRED_ID_COLUMNS_V1_EXTRA = ["endpoint_id"]
```

在 `validate_contract_df` 里不强制校验这个新常量（v0/v1 共用同一个校验函数，不应该分裂出版本分支逻辑——这是本次改动明确不做的事，`endpoint_id` 的存在性校验放在 Task 4 新测试里单独覆盖，不污染 contract_v0 的通用校验路径）。

- [ ] **Step 4: `build_contract.py` 新增 `endpoint_id` 列 + sidecar 落盘**

在 `main()` 里，`ep_to_svc` 加载之后新增确定性映射：
```python
    endpoint_id_map = {ep: i for i, ep in enumerate(sorted(ep_to_svc.keys()))}
```

在 `_attach_identity_columns` 调用处（`_process_one_case` 内）追加 `endpoint_id` 列——但只有 v1 需要，且映射表要传进去。改 `_process_one_case` 签名追加 `endpoint_id_map: dict[str, int] | None = None` 参数，在 `_attach_identity_columns` 内部（或调用后）追加：
```python
    if endpoint_id_map is not None:
        ep_df["endpoint_id"] = ep_df["endpoint_key"].map(endpoint_id_map)
```
`main()` 里调用 `_process_one_case` 时，仅当 `cfg.contract_version == "v1"` 才传 `endpoint_id_map=endpoint_id_map`，否则传 `None`（v0 路径列不存在，保持 `test_endpoint_id_not_added_in_v0` 断言成立）。

在 `_write_v1` 里，`fit_df`（即 `parts["train_fit"]`，Step 6 已在 main() 里算出）确定后、写 parquet 之前，新增 sidecar 落盘：
```python
    from src.data.endpoint_baseline_stats import EndpointBaselineStats

    red_cols = [c for c in feature_cols if c.startswith("endpoint_red__")]
    svc_cols = [c for c in feature_cols if c.startswith(("service_metric__", "service_log__"))]
    baseline_stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    baseline_stats.fit(fit_df)
    baseline_stats.save(out / "endpoint_baseline_stats.json")
```
这段代码接在 `main()` 里 `fit_df` 变量已经算出之后（v1 分支，紧邻 `normalizer.fit(fit_df)` 调用处），复用同一个 `fit_df`——不重新计算 split，避免与 `normalizer.fit` 用的 `train_fit` 产生细微不一致（两者必须是同一份 DataFrame）。

- [ ] **Step 5: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_build_contract_v1_endpoint_id.py -v`
Expected: PASS（3 个测试全绿）

- [ ] **Step 6: 跑既有 contract 测试确认无回归**

Run: `conda run -n interface python -m pytest tests/test_build_contract_smoke.py tests/test_contract_v1_split.py tests/test_contract_v0_schema.py -v`
Expected: PASS（v0 路径未受影响；v1 既有测试不受新增列干扰）

- [ ] **Step 7: Commit**

```bash
git add src/contracts/contract_v0.py scripts/build_contract.py tests/test_build_contract_v1_endpoint_id.py
git commit -m "[Feature]: Contract v1 新增 endpoint_id 列 + EndpointBaselineStats sidecar 落盘"
```

---

## Task 5: `contract_dataloader.py` / `train_baseline_v0.py` 传递 `endpoint_id` 给 fusion

**背景**：spec 组件3.3。`endpoint_id` 列已在 parquet 里（Task 4），现在要打通 dataset → collate → fusion 调用三层管道。v0 数据没有这一列，故 `_row_to_sample` 要做存在性检查而非硬编码假设列存在（v0/v1 共用同一个 `ContractDataset` 类）。

**Files:**
- Modify: `src/data/contract_dataloader.py`
- Modify: `scripts/train_baseline_v0.py`
- Test: `tests/test_contract_dataloader.py`（若不存在则新建；先检查是否已有同名文件）

- [ ] **Step 1: 写失败的测试**

先检查 `tests/` 下是否已有 `test_contract_dataloader.py`——若有则在其中追加，若无则新建：
```python
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.contract_dataloader import ContractDataset


def _write_fixture(tmp_path: Path, with_endpoint_id: bool) -> tuple[Path, Path]:
    cols = {
        "sample_id": ["s1", "s2"],
        "endpoint_key": ["epA", "epB"],
        "phase": ["normal", "normal"],
        "is_anomaly": [False, False],
        "f__x": [1.0, 2.0],
    }
    if with_endpoint_id:
        cols["endpoint_id"] = [0, 1]
    df = pd.DataFrame(cols)
    pq_path = tmp_path / "data.parquet"
    df.to_parquet(pq_path, index=False)
    schema = {"feature_groups": {"f": {"columns": ["f__x"]}}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    return pq_path, schema_path


def test_row_to_sample_includes_endpoint_id_when_present(tmp_path):
    pq_path, schema_path = _write_fixture(tmp_path, with_endpoint_id=True)
    ds = ContractDataset(parquet_path=pq_path, schema_path=schema_path, nan_strategy="zero")
    sample = ds[0]
    assert sample["meta"]["endpoint_id"] == 0


def test_row_to_sample_omits_endpoint_id_when_absent():
    """v0 数据没有 endpoint_id 列时，meta 里不应出现这个 key（而不是填 None 之类的
    占位符）——下游 _collate 靠这个 key 是否存在判断要不要堆叠传给 fusion。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        pq_path, schema_path = _write_fixture(Path(td), with_endpoint_id=False)
        ds = ContractDataset(parquet_path=pq_path, schema_path=schema_path, nan_strategy="zero")
        sample = ds[0]
        assert "endpoint_id" not in sample["meta"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_contract_dataloader.py -v`
Expected: FAIL——`KeyError: 'endpoint_id'`（第一个测试）

- [ ] **Step 3: 改 `_row_to_sample`（`src/data/contract_dataloader.py:104-114`）**

```python
    def _row_to_sample(self, row: pd.Series) -> dict:
        tensors = {
            name: torch.tensor(row[cols].values.astype(float), dtype=torch.float32)
            for name, cols in self._groups.items()
        }
        label = {
            "phase": row["phase"],
            "is_anomaly": bool(row["is_anomaly"]),
        }
        meta = {"sample_id": row["sample_id"], "endpoint_key": row["endpoint_key"]}
        if "endpoint_id" in row.index:
            meta["endpoint_id"] = int(row["endpoint_id"])
        return {**tensors, "label": label, "meta": meta}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_contract_dataloader.py -v`
Expected: PASS（2 个测试全绿）

- [ ] **Step 5: 改 `scripts/train_baseline_v0.py` 的 `_collate`（约第40-45行）+ 训练/推理循环**

`_collate` 需要处理"batch 里有的样本有 `endpoint_id`、有的没有"这种不可能发生但要显式排除的情形——同一个 parquet 内所有行的列集合是一致的（v0/v1 不混跑），故用第一个样本判断整批是否含 `endpoint_id` 即可，不必逐样本检查：

```python
def _collate(batch: list[dict]) -> dict:
    """聚合 ContractDataset point 样本：modality tensor 堆叠，meta/label 保持 list。

    endpoint_id 仅在 v1 数据（含该列）时存在于 meta；v0 数据不产出这个 key，
    batch 里也不会出现，此时 out 不含 "endpoint_id"，下游 fusion 调用走
    endpoint_id=None 的默认参数路径（向后兼容）。
    """
    out: dict = {m: torch.stack([s[m] for s in batch]) for m in MODALITY_ORDER}
    out["sample_id"] = [s["meta"]["sample_id"] for s in batch]
    out["is_anomaly"] = [s["label"]["is_anomaly"] for s in batch]
    if "endpoint_id" in batch[0]["meta"]:
        out["endpoint_id"] = torch.tensor([s["meta"]["endpoint_id"] for s in batch])
    return out
```

`_train`/`_infer` 里两处 `fusion({m: batch[m] for m in MODALITY_ORDER})` 调用，改为透传 `endpoint_id`：
```python
            x = fusion(
                {m: batch[m] for m in MODALITY_ORDER}, endpoint_id=batch.get("endpoint_id")
            )
```
`main()` 里 `svdd.init_center(fusion({m: init_batch[m] for m in MODALITY_ORDER}))` 同样加上 `endpoint_id=init_batch.get("endpoint_id")`。

**`ReliabilityGatedFusion` 的运行时构造参数补上**：`main()` 里 `hydra.utils.instantiate(cfg.fusion, modality_dims=modality_dims)` 这一行，对 `ReliabilityGatedFusion` 还需要额外的 `endpoint_baseline_stats`/`id_to_endpoint_key` 两个关键字参数（Task 3 已确认它们不进 yaml）。这两个参数对 L0/L1/L2 是多余的——`hydra.utils.instantiate` 遇到目标类 `__init__` 不接受的多余关键字参数会报 `TypeError`，不能无条件传。改为按 `cfg.fusion._target_` 条件分支：

```python
    fusion_kwargs = {"modality_dims": modality_dims}
    if cfg.fusion._target_ == "src.fusion.reliability_gate.ReliabilityGatedFusion":
        from src.data.endpoint_baseline_stats import EndpointBaselineStats

        baseline_stats = EndpointBaselineStats.load(contract_dir / "endpoint_baseline_stats.json")
        ep_to_svc = yaml.safe_load(
            (Path(__file__).parents[1] / "configs/contract/endpoint_to_service.yaml").read_text()
        )
        id_to_endpoint_key = {i: ep for i, ep in enumerate(sorted(ep_to_svc.keys()))}
        fusion_kwargs["endpoint_baseline_stats"] = baseline_stats
        fusion_kwargs["id_to_endpoint_key"] = id_to_endpoint_key
    fusion = hydra.utils.instantiate(cfg.fusion, **fusion_kwargs)
```
需要在文件顶部加 `import yaml`。这里没有另建一个"fusion factory"抽象——按 `_target_` 字符串分支是当前唯一需要特殊运行时参数的融合机制，YAGNI，若未来出现第二个需要特殊参数的机制再抽象。

- [ ] **Step 6: 跑现有训练脚本测试确认无回归**

Run: `conda run -n interface python -m pytest tests/test_train_baseline_v0.py tests/test_e2e_smoke.py -v`
Expected: PASS（L0/L1/L2 路径不受影响，`endpoint_id=batch.get("endpoint_id")` 对 v0 数据取到 `None`，等价于不传）

- [ ] **Step 7: Commit**

```bash
git add src/data/contract_dataloader.py scripts/train_baseline_v0.py tests/test_contract_dataloader.py
git commit -m "[Feature]: dataloader/train 脚本打通 endpoint_id 传递给 fusion"
```

---

## Task 6: 训练池扩容（吸收故障 case 的 baseline 阶段行）+ `source_phase` 可审计列

**背景**：spec §3.4。缓解数据稀缺（838→~7359 行）：`train_pool_mask = normal_mask | ((~normal_mask) & (phase == "baseline"))`。**只吸收 baseline，不吸收 recover**（系统未稳定回正常态，分布未验证，保守排除）。防泄漏强制要求：`eval_all` 的构成必须从 `full[~normal_mask]` 收紧为 `full[~train_pool_mask]`——否则复现 v0 的 train⊆eval 泄漏 bug。新增 `source_phase` 列（`"normal_case"`/`"fault_baseline"`）标记来源，可审计。这个改动只影响 v1 分支（`_write_v1`），v0 不变。

已用真实数据验证：`eval_all.parquet` 里 `phase=="baseline"` 的行共 5411 行（跨27种故障类型），扩容后训练池约为 838+5411=6249（注：spec 估计 ~7359，实测口径略有差异，属正常——spec 写作时可能基于稍早的数据版本或不同的 case 筛选口径，实现时以实测真实值为准，不强行对齐 spec 估计数）。

**Files:**
- Modify: `scripts/build_contract.py`（`_write_v1` 及其调用处）
- Test: `tests/test_contract_v1_train_pool.py`（新建）

- [ ] **Step 1: 写失败的测试**

`tests/test_contract_v1_train_pool.py`：
```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable, "scripts/build_contract.py",
            "--config", str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset", str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir", str(out_dir), "--seed", "42",
        ],
        check=True, cwd=str(REPO_ROOT),
    )


def test_train_pool_includes_fault_baseline_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    train = pd.read_parquet(out_dir / "train.parquet")
    # mini fixture 含故障 case（如 Lv_P_DISKIO_preserve），其 baseline 阶段行
    # 现在应出现在 train.parquet 里（扩容前 train 只含 Normal case）。
    assert (train["anomaly_type"] != "Normal").any(), (
        "train.parquet 未吸收任何故障 case 的 baseline 行，训练池扩容未生效"
    )
    # 吸收的行必须确实是 baseline 阶段，不能混入 inject/recover
    non_normal = train[train["anomaly_type"] != "Normal"]
    assert (non_normal["phase"] == "baseline").all()


def test_train_pool_excludes_recover_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert not (train["phase"] == "recover").any()
    assert not (train["phase"] == "inject").any()


def test_source_phase_column_present_and_correct(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "source_phase" in train.columns
    normal_rows = train[train["anomaly_type"] == "Normal"]
    fault_rows = train[train["anomaly_type"] != "Normal"]
    assert (normal_rows["source_phase"] == "normal_case").all()
    if len(fault_rows):
        assert (fault_rows["source_phase"] == "fault_baseline").all()


def test_eval_all_excludes_train_pool_rows_no_leakage(tmp_path):
    """核心防泄漏不变量：train_pool 里的行（含新吸收的故障 baseline 行）
    必须同步从 eval_all 摘除，否则复现 v0 的 train⊆eval 泄漏 bug——这正是
    spec §3.4 点名要求的强制项，不是可选加固。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()

    # eval_all 不应再含任何故障 case 的 baseline 行（全部被吸收进训练池并摘除）
    fault_baseline_in_eval = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    assert len(fault_baseline_in_eval) == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_contract_v1_train_pool.py -v`
Expected: FAIL——当前 `train.parquet` 只含 `train_fit`（纯 Normal），`source_phase` 列不存在

- [ ] **Step 3: 改 `_write_v1`**

**关键澄清（避免实现时范围混淆）**：`train_fit`/`train_val`/`eval_normal_holdout` 三路时序切分（`split_normal_rows_temporal`）**只作用于 Normal 行**，本任务不改这个切分本身，也不改 `Normalizer`/`EndpointBaselineStats` 的 fit 范围（两者仍固定在 `parts["train_fit"]`，纯 Normal——reliability gate 的偏离量基线必须干净，不能被故障 case 的 baseline 行污染，这是 Task 4 已确认的行为，本任务不动）。本任务只改**训练循环实际消费的 `train.parquet` 里装什么**：从"纯 `train_fit` 的冗余拷贝"变成"`train_fit` ∪ 故障 case 的 baseline 行"。

```python
def _write_v1(out: Path, full: pd.DataFrame, normal_mask: pd.Series, seed: int) -> None:
    """v1：Normal 行按时间窗三路切分；train.parquet 额外吸收故障 case 的 baseline
    阶段行扩容训练池（不吸收 recover——系统未稳定回正常态，分布未验证，保守排除）；
    eval_all 同步从这些被吸收的行里摘除，防止复现 v0 的 train⊆eval 泄漏。
    """
    normal_df = full[normal_mask].reset_index(drop=True)
    anomaly_df = full[~normal_mask].reset_index(drop=True)
    parts = split_normal_rows_temporal(normal_df, seed=seed)

    parts["train_fit"].to_parquet(out / "train_fit.parquet", index=False)
    parts["train_val"].to_parquet(out / "train_val.parquet", index=False)
    parts["eval_normal_holdout"].to_parquet(out / "eval_normal_holdout.parquet", index=False)

    # 训练池扩容：train_fit（Normal，打标 normal_case）+ 故障 case 的 baseline
    # 阶段行（打标 fault_baseline）。source_phase 可审计，未来可按需排除/降权
    # （已知风险：inside_payment/preserve 两 endpoint 的故障 case baseline 期
    # 流量比 normal_v2 高 1.8~1.9x，见 spec §3.4/§6，本轮不处理，只留痕）。
    train_fit_labeled = parts["train_fit"].copy()
    train_fit_labeled["source_phase"] = "normal_case"
    fault_baseline_df = anomaly_df[anomaly_df["phase"] == "baseline"].copy()
    fault_baseline_df["source_phase"] = "fault_baseline"
    train_pool = pd.concat([train_fit_labeled, fault_baseline_df], ignore_index=True)
    train_pool.to_parquet(out / "train.parquet", index=False)

    # eval_all：故障 case 的 inject/recover 行（baseline 已被吸收进训练池，摘除）
    # + 仅 holdout 的 Normal 行。
    eval_all = pd.concat(
        [anomaly_df[anomaly_df["phase"] != "baseline"], parts["eval_normal_holdout"]],
        ignore_index=True,
    )
    eval_all.to_parquet(out / "eval_all.parquet", index=False)
    LOG.info(
        "v1 切分：train_fit=%d train_val=%d holdout=%d fault_baseline=%d train_pool=%d eval_all=%d",
        len(parts["train_fit"]), len(parts["train_val"]), len(parts["eval_normal_holdout"]),
        len(fault_baseline_df), len(train_pool), len(eval_all),
    )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_contract_v1_train_pool.py -v`
Expected: PASS（4 个测试全绿）

- [ ] **Step 5: 跑全部既有 contract/fusion 测试确认无回归**

Run: `conda run -n interface python -m pytest tests/ -v -k "contract or fusion or normalization or endpoint_baseline"`
Expected: PASS。重点关注 `test_contract_v1_split.py::test_build_contract_v1_smoke` 与
`test_build_contract_v1_normalizer_excludes_holdout`——前者验证 train/eval_all 互斥
不受本次改动破坏（train 新增的 fault_baseline 行与 eval_all 的 anomaly 非-baseline
行天然不交），后者验证 Normalizer 的 fit 范围仍严格锁在纯 Normal `train_fit`
（本任务未改动这条路径，理应保持通过）。

- [ ] **Step 6: 跑真实 `merged_v2` 数据验证扩容后行数量级符合预期**

Run: `dvc repro build_contract_v1`（若已有产物且此步改动会触发 DVC 检测到依赖变化重跑）
Expected: 日志打印 `train_pool` 行数应显著大于 838（实测口径下预计 6000~7000+区间，
不强行对齐 spec 估计的 7359——以真实重跑结果为准），`eval_all` 行数应从原 13632
减少（baseline 行被摘除，减少量 ≈ 故障 case baseline 行数）。

- [ ] **Step 7: Commit**

```bash
git add scripts/build_contract.py tests/test_contract_v1_train_pool.py
git commit -m "[Feature]: Contract v1 训练池扩容，吸收故障 case baseline 阶段行"
```

---

## Task 7: 端到端跑通 `reliability_gate` fusion，新增 `dvc.yaml` stage

**背景**：前6个 Task 已经把所有组件和管线接入拼齐，本 Task 是第一次真正端到端跑一遍训练/评估，确认没有 Task 间接口没对齐的问题（比如 Task 5 的 `id_to_endpoint_key` 派生逻辑与 Task 4 的 `endpoint_id_map` 是否一致——两处都用 `sorted(ep_to_svc.keys())`，必须是同一份确定性顺序，否则 `endpoint_id` 整数值与 sidecar 里的 endpoint_key 会错位）。

**Files:**
- Modify: `dvc.yaml`
- Test: `tests/test_e2e_reliability_gate_smoke.py`（新建）

- [ ] **Step 1: 写失败的 e2e smoke 测试**

`tests/test_e2e_reliability_gate_smoke.py`：
```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]


def test_reliability_gate_e2e_on_mini_fixture(tmp_path):
    contract_dir = tmp_path / "contract_v1"
    subprocess.run(
        [
            sys.executable, "scripts/build_contract.py",
            "--config", str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset", str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir", str(contract_dir), "--seed", "1",
        ],
        check=True, cwd=str(REPO_ROOT),
    )

    scores_path = tmp_path / "scores.parquet"
    subprocess.run(
        [
            sys.executable, "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}", f"out={scores_path}",
            "seed=1", "training.epochs=2", "fusion=reliability_gate",
        ],
        check=True, cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(scores_path)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_e2e_reliability_gate_smoke.py -v`
Expected: 大概率 FAIL 在 `id_to_endpoint_key`/`endpoint_baseline_stats` 加载或 endpoint_id dtype 相关报错——这是本 Task 存在的意义，具体报错内容取决于 Task 1-6 实现细节，实现者需要据实际报错调试，本步骤不预设具体错误消息。

- [ ] **Step 3: 调试直到通过**

常见问题排查方向（不是必然发生，供参考）：
- `endpoint_id` 列 dtype 若在 parquet 里被存成 `int64` 而 `_row_to_sample` 里 `int(row["endpoint_id"])` 处理没问题；若 pandas 读出来是 `numpy.int64`，`torch.tensor([...])` 构造 batch 张量时应能正常处理，无需特殊转换。
- `EndpointBaselineStats.load` 读到的 endpoint_key 集合必须是 `train_fit` 里出现的 endpoint（而非 `train_pool`——Task 6 扩容后的 train.parquet 含故障 case 的 endpoint，但 Task 4 的 sidecar fit 范围固定在纯 Normal `train_fit`，两者 endpoint 覆盖面理论上一致，因为 8 个 endpoint 在 Normal case 里全部出现，见 `split_v1.py` 的"每个 endpoint 在三份都出现"设计目标）；若 mini fixture 的某个 endpoint 恰好不在 `train_fit` 里（如窗数不足退化场景），`gate_weights` 里对该 endpoint 的样本会抛 `KeyError`——若 e2e 测试触发这个情形，说明 mini fixture 数据规模下 `EndpointBaselineStats` 覆盖不全，这是已知的稀疏数据边界情形（spec §4.5 稳健性维度），不是本 Task 要修的 bug，可以换一个覆盖更全的 mini fixture 场景或在测试里显式跳过该 endpoint 的样本断言，视实测情况决定，不预先规定解法。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_e2e_reliability_gate_smoke.py -v`
Expected: PASS

- [ ] **Step 5: 新增 `dvc.yaml` stage**

```yaml
  train_v1_reliability_gate:
    cmd: python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1
      out=artifacts/baseline_v1_reliability_gate/scores.parquet seed=42
      training.epochs=50 fusion=reliability_gate model=deep_svdd
    deps:
      - scripts/train_baseline_v0.py
      - configs/base.yaml
      - configs/fusion/reliability_gate.yaml
      - configs/model/deep_svdd.yaml
      - configs/training/default.yaml
      - configs/contract/endpoint_to_service.yaml
      - src/models/deep_svdd.py
      - src/fusion/reliability_gate.py
      - src/fusion/base.py
      - src/data/contract_dataloader.py
      - src/data/endpoint_baseline_stats.py
      - artifacts/contract_v1/train.parquet
      - artifacts/contract_v1/eval_all.parquet
      - artifacts/contract_v1/schema.json
      - artifacts/contract_v1/endpoint_baseline_stats.json
    outs:
      - artifacts/baseline_v1_reliability_gate/scores.parquet

  eval_v1_reliability_gate:
    cmd: python scripts/eval_baseline_v0.py
      --scores artifacts/baseline_v1_reliability_gate/scores.parquet
      --out artifacts/baseline_v1_reliability_gate/metrics.json
    deps:
      - scripts/eval_baseline_v0.py
      - src/contracts
      - artifacts/baseline_v1_reliability_gate/scores.parquet
    metrics:
      - artifacts/baseline_v1_reliability_gate/metrics.json:
          cache: false
```

- [ ] **Step 6: `dvc repro` 冒烟（真实数据）**

Run: `dvc repro build_contract_v1 train_v1_reliability_gate eval_v1_reliability_gate`
Expected: 三个 stage 跑通，`artifacts/baseline_v1_reliability_gate/metrics.json` 产出。这一步耗时较长（真实数据训练50 epoch），可先用较小 `training.epochs` override 快速验证管道通畅，再正式跑一遍记录数值（正式数值记录在 Task 9）。

- [ ] **Step 7: Commit**

```bash
git add dvc.yaml tests/test_e2e_reliability_gate_smoke.py
git commit -m "[Feature]: reliability_gate 端到端 dvc stage + smoke 测试"
```

---

## Task 8: 实验——多 seed 主对比 + 2×2 训练池归因 + 路由消融

**背景**：spec §4，回应两轮审稿要求（归因清晰、区分"路由 vs 幅度门控"、区分"机制 vs 数据量"）。本 Task 不写新代码，是用 Task 1-7 已经跑通的管线批量跑实验、记录结果。三块实验：

1. **§4.1 主对比**：ReliabilityGate vs L0/L1/L2，seed=1/2/3/42，报 AUROC/AUPRC ± std。
2. **§4.2 训练池扩容归因（2×2）**：{原始838行, 扩容~7000+行} × {L2, ReliabilityGate}，证明提升不是纯数据量带来的。原始838行版本 = 直接用 `train_fit.parquet` 而非扩容后的 `train.parquet` 训练（`contract_dir` 下手动切换/软链接，或新增一个 `--train-file` 式的 override——见 Step 3 的具体做法）。
3. **§4.3 路由消融**：`dev` 换成总偏离幅度标量（去分支区分）/ softmax 换两个独立 sigmoid / 固定 `[0.5,0.5]`（禁用门控）。这三个消融变体需要 `ReliabilityGatedFusion` 支持行为开关——若 Task 2 实现时未预留这些开关，本 Task 需要先给 `ReliabilityGatedFusion.__init__` 补充消融专用参数（见 Step 5）。

**Files:**
- Modify: `src/fusion/reliability_gate.py`（补充消融开关，若 Task 2 未预留）
- Create: `configs/fusion/reliability_gate_ablation_*.yaml`（3个消融变体配置）
- 产出: `artifacts/rg_seed{1,2,3,42}/metrics.json`、`artifacts/rg_838/metrics.json`、`artifacts/rg_ablation_*/metrics.json`

- [ ] **Step 1: 主对比——4 个 seed 跑 ReliabilityGate**

```bash
for seed in 1 2 3 42; do
  python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1 \
    out=artifacts/rg_seed${seed}/scores.parquet seed=${seed} training.epochs=50 \
    fusion=reliability_gate model=deep_svdd
  python scripts/eval_baseline_v0.py --scores artifacts/rg_seed${seed}/scores.parquet \
    --out artifacts/rg_seed${seed}/metrics.json
done
```
L0/L1/L2 的多 seed 结果若尚未跑齐（entry 011/012 只跑了单 seed），同样方式补跑 seed=1/2/3（seed=42 已有）。汇总 `overall.auroc`/`overall.auprc` 均值±std 到 Task 9 的 history entry。

- [ ] **Step 2: 读取并记录主对比结果**

```bash
for seed in 1 2 3 42; do
  echo "seed=${seed}:"; cat artifacts/rg_seed${seed}/metrics.json | python -c \
    "import json,sys; d=json.load(sys.stdin); print(d['overall'])"
done
```
把读到的真实数值记入下一步 history entry，不用占位符（与 entry 012 的既有约定一致）。

- [ ] **Step 3: 2×2 训练池归因——构造"原始838行"版本的 contract_dir**

`train.parquet` 现在是扩容后的训练池；`train_fit.parquet` 是未扩容的纯 Normal 838 行。最简单的做法是建一份平行 `contract_dir`，把 `train.parquet` 换成 `train_fit.parquet` 的内容，其余文件（`eval_all.parquet`/`schema.json`/`endpoint_baseline_stats.json`）复用同一份（后两者的 fit 范围本来就固定在 `train_fit`，天然与"原始838行"场景一致，不需要重算）：

```bash
mkdir -p artifacts/contract_v1_838
cp artifacts/contract_v1/{eval_all.parquet,schema.json,endpoint_baseline_stats.json,normalization_stats.json} artifacts/contract_v1_838/
cp artifacts/contract_v1/train_fit.parquet artifacts/contract_v1_838/train.parquet
```

```bash
for fusion in gated reliability_gate; do
  python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1_838 \
    out=artifacts/${fusion}_838/scores.parquet seed=42 training.epochs=50 fusion=${fusion}
  python scripts/eval_baseline_v0.py --scores artifacts/${fusion}_838/scores.parquet \
    --out artifacts/${fusion}_838/metrics.json
done
```
（扩容版的 L2/ReliabilityGate 结果直接复用 entry 012 的 `artifacts/l2/metrics.json` 和本 Task Step 1 的 `artifacts/rg_seed42/metrics.json`，2×2 表格四格齐了：{838, 扩容}×{L2, RG}。）

- [ ] **Step 4: 补充测试——`test_contract_v1_train_pool.py` 追加"838 版本可独立复现"回归**

```python
def test_contract_v1_838_variant_is_pure_normal(tmp_path):
    """2x2 归因实验依赖的手工构造 contract_v1_838：确认 train.parquet 替换为
    train_fit.parquet 后确实是纯 Normal（无 source_phase 列或全为 normal_case），
    防止归因实验的"原始838行"对照组混入扩容数据。"""
    out_dir = tmp_path / "contract_v1"
    subprocess.run(
        [sys.executable, "scripts/build_contract.py", "--config",
         str(REPO_ROOT / "configs/contract/v1.yaml"), "--dataset",
         str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
         "--out-dir", str(out_dir), "--seed", "42"],
        check=True, cwd=str(REPO_ROOT),
    )
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert (train_fit["anomaly_type"] == "Normal").all()
```
Run: `conda run -n interface python -m pytest tests/test_contract_v1_train_pool.py -v -k 838`
Expected: PASS（`train_fit.parquet` 本身在 Task 6 之后未被改动，此断言应直接成立，属确认性测试非新功能）

- [ ] **Step 5: 路由消融——给 `ReliabilityGatedFusion` 补充消融开关**

三个消融变体都是"改变门控输入/归一化方式"，不改融合公式本身（`w_ep*value_ep + w_svc*value_svc` 不变），故实现为构造参数开关而非新类，避免为消融变体各写一个几乎重复的 `FusionModule` 子类。

先写失败测试（追加到 `tests/test_reliability_gated_fusion.py`）：
```python
def test_scalar_deviation_mode_uses_1d_input_to_gate():
    """消融1：dev 换成'总偏离幅度标量'（去掉分支区分）——gate_mlp 输入应从 6 维
    降到 2 维（每分支 1 个标量：总 z-score 范数），验证'该信哪个分支'这个能力
    被移除后 gate_mlp 的实际输入维度确实变了（不是只加了个没用的开关）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS, endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"}, branch_dim=4,
        deviation_mode="scalar",
    )
    assert fusion.gate_mlp[0].in_features == 2


def test_independent_sigmoid_mode_weights_do_not_sum_to_one():
    """消融2：softmax 换两个独立 sigmoid——权重和不再恒为1，验证'竞争性归一化'
    确实被换成了'独立开关'语义（否则消融变体和原实现在数值上无法区分）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS, endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"}, branch_dim=4,
        gate_normalization="independent_sigmoid",
    )
    batch = {"endpoint_red": torch.randn(6, 3), "service_metric": torch.randn(6, 2),
              "service_log": torch.randn(6, 1)}
    endpoint_id = torch.tensor([0, 1, 0, 1, 0, 1])
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id)
    assert not torch.allclose(w_ep + w_svc, torch.ones(6))


def test_fixed_uniform_gate_disables_learning():
    """消融3：固定 [0.5, 0.5]（禁用门控）——下界对照，gate_mlp 应不存在
    可训练参数参与该路径（权重恒定，与输入无关）。"""
    stats = _fitted_stats()
    fusion = ReliabilityGatedFusion(
        modality_dims=MODALITY_DIMS, endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA", 1: "epB"}, branch_dim=4,
        gate_normalization="fixed_uniform",
    )
    batch = {"endpoint_red": torch.randn(4, 3) * 100, "service_metric": torch.randn(4, 2) * 100,
              "service_log": torch.randn(4, 1) * 100}
    endpoint_id = torch.tensor([0, 1, 0, 1])
    w_ep, w_svc = fusion.gate_weights(batch, endpoint_id)
    torch.testing.assert_close(w_ep, torch.full((4,), 0.5))
    torch.testing.assert_close(w_svc, torch.full((4,), 0.5))
```

Run: `conda run -n interface python -m pytest tests/test_reliability_gated_fusion.py -v -k "scalar_deviation or independent_sigmoid or fixed_uniform"`
Expected: FAIL——三个参数目前都不存在，`TypeError: unexpected keyword argument`

实现改动（`src/fusion/reliability_gate.py`）：`__init__` 新增 `deviation_mode: Literal["branch_aware", "scalar"] = "branch_aware"` 和 `gate_normalization: Literal["softmax", "independent_sigmoid", "fixed_uniform"] = "softmax"` 两个参数。`deviation_mode="scalar"` 时 `_deviation_summary` 只返回 `[norm]`（1维）而非 3 维，`gate_mlp` 第一层 `in_features` 相应从 6 改为 2；`gate_normalization` 影响 `gate_weights` 里最终归一化方式（`softmax`→`torch.softmax`；`independent_sigmoid`→对 2 个 logits 各自 `sigmoid` 不归一化；`fixed_uniform`→跳过 `gate_mlp` 前向直接返回常数 0.5，此时 `gate_mlp` 仍会被构造但不参与 forward 路径，等价于"存在但未被使用"，不必删除模块本身以保持代码路径统一）。

Run: `conda run -n interface python -m pytest tests/test_reliability_gated_fusion.py -v`
Expected: PASS（全部测试，含 Task 2 原有的 + 本 Step 新增的 3 个）

- [ ] **Step 6: 新增 3 个消融 config + 跑实验**

`configs/fusion/reliability_gate_ablation_scalar_dev.yaml`：
```yaml
_target_: src.fusion.reliability_gate.ReliabilityGatedFusion
branch_dim: 16
dropout: 0.1
deviation_mode: scalar
```
`configs/fusion/reliability_gate_ablation_indep_sigmoid.yaml`：同上但 `gate_normalization: independent_sigmoid`（`deviation_mode` 留默认 `branch_aware`）。
`configs/fusion/reliability_gate_ablation_fixed_uniform.yaml`：同上但 `gate_normalization: fixed_uniform`。

```bash
for variant in scalar_dev indep_sigmoid fixed_uniform; do
  python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1 \
    out=artifacts/rg_ablation_${variant}/scores.parquet seed=42 training.epochs=50 \
    fusion=reliability_gate_ablation_${variant} model=deep_svdd
  python scripts/eval_baseline_v0.py --scores artifacts/rg_ablation_${variant}/scores.parquet \
    --out artifacts/rg_ablation_${variant}/metrics.json
done
```

- [ ] **Step 7: Commit**

```bash
git add src/fusion/reliability_gate.py tests/test_reliability_gated_fusion.py \
  configs/fusion/reliability_gate_ablation_*.yaml tests/test_contract_v1_train_pool.py \
  artifacts/rg_seed*/metrics.json artifacts/*_838/metrics.json artifacts/rg_ablation_*/metrics.json
git commit -m "[Experiment]: ReliabilityGate 多seed主对比 + 2x2训练池归因 + 路由消融"
```
（同 entry 012 约定：只提交小文件 `metrics.json`，不提交 `scores.parquet` 大文件产物——先跑 `git status` 确认哪些被 `.gitignore` 排除。）

---

## Task 9: 可解释性证据 + 稳健性实验 + history entry

**背景**：spec §4.4/§4.5，论文卖点部分。按故障类型分层展示 gate 学到的 `w_ep`/`w_svc`，预期 HTTP 类故障 `w_ep` 高、资源类故障 `w_svc` 高（对应 §1.2 已验证的数据信号）。稳健性看 shrinkage-to-global 开/关对稀疏 endpoint（如 login）的影响。

**Files:**
- Create: `scripts/analyze_gate_weights.py`（一次性分析脚本，非管线常驻组件）
- Create: `history/entries/014-reliability-gate-fusion.md`
- Modify: `history/index.md`

- [ ] **Step 1: 写 `scripts/analyze_gate_weights.py`**

这是分析脚本不是训练管线的一部分，不需要走 TDD（无持续维护的正确性契约，一次性产出图表/表格用于论文），但仍需能独立运行验证：

```python
#!/usr/bin/env python
"""按 anomaly_type 分层统计 ReliabilityGatedFusion 学到的 w_ep/w_svc 均值，
用于 spec §4.4 可解释性证据（HTTP 类故障预期 w_ep 高，资源类故障预期 w_svc 高）。
一次性分析脚本，不是训练/评估管线的一部分。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.contract_dataloader import ContractDataset
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER
from src.fusion.reliability_gate import ReliabilityGatedFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", required=True)
    parser.add_argument("--checkpoint", required=False, help="若训练脚本保存了权重则传入")
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    schema = json.loads((contract_dir / "schema.json").read_text())
    modality_dims = {n: len(g["columns"]) for n, g in schema["feature_groups"].items()}
    baseline_stats = EndpointBaselineStats.load(contract_dir / "endpoint_baseline_stats.json")
    ep_to_svc = yaml.safe_load(
        Path("configs/contract/endpoint_to_service.yaml").read_text()
    )
    id_to_key = {i: ep for i, ep in enumerate(sorted(ep_to_svc.keys()))}

    fusion = ReliabilityGatedFusion(
        modality_dims=modality_dims, endpoint_baseline_stats=baseline_stats,
        id_to_endpoint_key=id_to_key, branch_dim=16,
    )
    if args.checkpoint:
        fusion.load_state_dict(torch.load(args.checkpoint))
    fusion.eval()

    ds = ContractDataset(
        parquet_path=contract_dir / "eval_all.parquet",
        schema_path=contract_dir / "schema.json", nan_strategy="mean",
        fit_on_parquet=contract_dir / "train.parquet",
    )
    eval_df = pd.read_parquet(contract_dir / "eval_all.parquet")

    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            batch = {m: sample[m].unsqueeze(0) for m in MODALITY_ORDER}
            eid = torch.tensor([sample["meta"]["endpoint_id"]])
            w_ep, w_svc = fusion.gate_weights(batch, eid)
            records.append({"sample_id": sample["meta"]["sample_id"],
                              "w_ep": w_ep.item(), "w_svc": w_svc.item()})

    weights_df = pd.DataFrame(records).merge(
        eval_df[["sample_id", "anomaly_type"]], on="sample_id"
    )
    summary = weights_df.groupby("anomaly_type")[["w_ep", "w_svc"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑分析脚本，记录可解释性结果**

Run: `python scripts/analyze_gate_weights.py --contract-dir artifacts/contract_v1`

注：本脚本用随机初始化权重跑通只验证管道通畅——真正有意义的可解释性结果需要传入 Task 8 训练出的 `reliability_gate` 模型权重（`--checkpoint`）。`train_baseline_v0.py` 当前**没有保存模型权重**（只写 `scores.parquet`），若要跑这一步的真实分析，需要先给训练脚本加一个可选的 `torch.save(fusion.state_dict(), ...)`（这是本 Task 唯一需要回头改训练脚本的地方，改动限定在"新增可选保存"，不改变现有默认行为、不破坏 Task 5-7 的既有测试）。

- [ ] **Step 3: 稳健性——shrinkage 开关对稀疏 endpoint 的影响**

```bash
python -c "
from src.data.endpoint_baseline_stats import EndpointBaselineStats
import pandas as pd
train = pd.read_parquet('artifacts/contract_v1/train_fit.parquet')
red_cols = [c for c in train.columns if c.startswith('endpoint_red__')]
svc_cols = [c for c in train.columns if c.startswith(('service_metric__','service_log__'))]
for use_shrinkage in (True, False):
    stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols, use_shrinkage=use_shrinkage)
    stats.fit(train)
    mean, std = stats.branch_stats('POST:/api/v1/users/login', 'ep')
    print(f'shrinkage={use_shrinkage}: login std[:3]={std[:3]}')
"
```
对比 login endpoint（真实数据仅17行）在 shrinkage 开/关下 std 估计的差异，记入 history entry。

- [ ] **Step 4: 写 `history/entries/014-reliability-gate-fusion.md`**

按 `history/entries/_template.md` 结构，仿照 entry 011/012/013 的写法（本计划已参考过它们的风格）。必须包含：
- **做了什么**：EndpointBaselineStats + ReliabilityGatedFusion + endpoint_id 管线接入 + 训练池扩容，Contract v1 上多 seed 实测结果（真实数值，不用占位符）。
- **关键决策**：softmax 而非 sigmoid 的理由（数据信号 HTTP/资源故障此消彼长）；偏离量表示而非原始特征作为门控输入的理由（机制约束+防过拟合）；只吸收 baseline 不吸收 recover 的理由；2x2 归因实验的构造方式（`train_fit.parquet` 替换 `train.parquet` 而非重新切分）。
- **坑/已知问题**：若 Task 7 调试过程中遇到实际问题（`endpoint_id` 映射错位等），如实记录；已知的 `inside_payment`/`preserve` 流量偏移风险（source_phase 可审计但本轮未处理）；HTTPPATCH/HTTPDELAY 等困难样本若消融/主对比中确认两模态弱偏离、gate 无优势，如实报告不掩盖（spec §6 风险2）。
- **遗留 TODO**：spec §7 明确排除项原样搬入（不做 sibling cross-attention、不做 endpoint 时序建模、不做 session-conditional 数值校正、不做逐特征全维门控输入）；SVDD 超球体距离反馈门控 plan B 留痕。

- [ ] **Step 5: 更新 `history/index.md`**

Entry 列表追加一行（编号014，日期为实施当天），影响域索引表追加：`src/fusion/`、`src/data/`（`endpoint_baseline_stats.py`）、`src/contracts/`（`endpoint_id`/训练池扩容）、`scripts/build_contract.py`、`scripts/train_baseline_v0.py`、`configs/fusion/`。

- [ ] **Step 6: Commit**

```bash
git add scripts/analyze_gate_weights.py history/entries/014-reliability-gate-fusion.md history/index.md
git commit -m "[Docs]: reliability gate 融合机制 history entry + 可解释性分析脚本"
```

---

## 最终验证

- [ ] **跑完整测试套件**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: PASS（Task 1-9 全部新增/修改测试 + 所有既有测试无回归，包括 v0/v1 既有 pipeline、L0/L1/L2 fusion、Normalizer）

- [ ] **v0/v1（L0-L2）既有 pipeline 冒烟（确认未被本轮改动破坏）**

Run: `dvc repro build_contract build_contract_v1 train_v0 eval_v0 train_v1 eval_v1`
Expected: 既有 stage 全部正常跑通，产出不受影响。**特别关注**：`build_contract_v1` 重跑后
`artifacts/contract_v1/eval_all.parquet` 行数应比之前减少（baseline 行被摘除进训练池），
若下游还有其他脚本/测试硬编码了旧的 `eval_all` 行数（如 13632），需要同步更新——搜索
`grep -rn "13632" tests/ scripts/ docs/` 排查。

- [ ] **reliability_gate 新链路端到端冒烟**

Run: `dvc repro train_v1_reliability_gate eval_v1_reliability_gate`
Expected: 跑通，`artifacts/baseline_v1_reliability_gate/metrics.json` 产出，AUROC/AUPRC
数值记入 history entry（若 Task 9 Step 4 写 entry 时这一步还没跑完，回头补上真实数值，
不要用占位符占坑后忘记回填）。

- [ ] **对照 spec §4 完成标准逐项自查**

- [ ] §4.1 主对比：4 个 seed 的 ReliabilityGate vs L0/L1/L2 结果表已产出
- [ ] §4.2 训练池归因 2×2：四格数值齐全，且能说明"提升不是纯数据量带来的"（或如实报告未能说明）
- [ ] §4.3 路由消融：3 个变体结果已产出，能回应"路由是否真有用"
- [ ] §4.4 可解释性：按故障类型的 w_ep/w_svc 分层表已产出
- [ ] §4.5 稳健性：shrinkage 开关对稀疏 endpoint 的影响已记录
- [ ] §6 已识别风险中"困难样本"一项如实报告（不掩盖 gate 在弱偏离样本上无优势的情形，若发生）
- [ ] history entry 014 已写，index.md 已更新，spec §7 明确排除项已搬入"遗留 TODO"
