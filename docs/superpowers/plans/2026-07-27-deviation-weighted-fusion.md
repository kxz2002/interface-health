# DeviationWeightedFusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `DeviationWeightedFusion`——一个新的 `FusionModule` 子类，对 `endpoint_red`/`service_metric+service_log` 两分支的每一维特征分别计算相对自身 endpoint normal 基线的 z-score，用 `sigmoid(|z|-threshold)` 逐特征加权后直接 concat（无编码器、无分支级门控、零可学习参数），验证"逐特征可信度加权能否救回信号淹没 gap"（entry 017：ABORT/REPLACE 单特征 oracle AUROC 0.99 vs naive fusion 0.3~0.5）。跑通 Contract v1（`v1_expanded_pool.yaml`）训练/评估，与 L0/RG/oracle 在 `by_anomaly_type` headline 指标上直接对比，按 spec §5 三档标准判定是否值得深化。

**Architecture:**
- `DeviationWeightedFusion`（新增，`src/fusion/deviation_weighted.py`，继承 `FusionModule`）：`forward(modality_dict, endpoint_id=None)`——按 `endpoint_id` 查 `EndpointBaselineStats.branch_stats` 拿到该 endpoint 的逐特征 mean/std，算 `z=(raw-mean)/std`，`w=sigmoid(|z|-threshold)`，`weighted=raw*w`；`EndpointBaselineStats.degenerate_columns(branch)` 命中的列权重强制为 1（不计算 sigmoid）；两分支 `weighted_ep`/`weighted_svc` 直接 concat，无编码器。`endpoint_id=None` 时全部权重为 1，退化为纯 concat（等价 L0）。`from_contract` 覆写：从 `contract_dir` 加载 `endpoint_baseline_stats.json` + `schema.json`（后者提供逐列列名，用于把 `degenerate_columns()` 返回的列名映射到张量里的位置索引），复用 `src/contracts/endpoint_id_mapping.py` 派生 `id_to_endpoint_key`——这套构造模式与 `ReliabilityGatedFusion.from_contract`（`src/fusion/reliability_gate.py:171-195`）几乎一样，只是多读一份 `schema.json`。
- 复用现有基础设施，零改动：`EndpointBaselineStats`（`src/data/endpoint_baseline_stats.py`，已提供 `branch_stats`/`degenerate_columns`/`fitted_endpoints`）、Contract v1 `v1_expanded_pool.yaml`（`fit_endpoint_baseline_stats=true`，已产出 `endpoint_baseline_stats.json`+`schema.json`）、`train_baseline_v0.py`（`fusion = hydra.utils.get_class(cfg.fusion._target_).from_contract(...)`，构造对训练脚本完全透明，无需 `_target_` 字符串分支）、`DeepSVDD` 打分公式。
- 管线接入：新增 `configs/fusion/deviation_weighted.yaml`；新增隔离的 `dvc_deviation_weighted/dvc.yaml`（`train_v1_deviation_weighted`+`eval_v1_deviation_weighted` 两个 stage，直接依赖 `dvc_reliability_gate/dvc.yaml` 的 `build_contract_v1_expanded` stage 已产出的 `artifacts/contract_v1_expanded/*`，不重新定义 build 阶段——与 RG 用同一份 contract 才能直接对比是 spec §4 的硬性要求）。

**Tech Stack:** PyTorch (`nn.Module`), Hydra (`hydra.utils.instantiate`), pandas, numpy, pytest, conda env `interface`, DVC

**Spec:** `docs/superpowers/specs/2026-07-27-deviation-weighted-fusion-design.md`

---

## 背景说明（供实施者理解代码库约定，执行前必读）

- **本轮是最小改动验证，不是最终方法**：spec §1.3/§8 明确排除可学习阈值、分支级门控、SVDD 打分公式改造、跨run污染切分修复、第三分支拆分。实现时严格照做，不要"顺手"加这些——若验证不达标，按 spec §5 直接停止，不做后续深化任务（这是用户的明确决定，不是实施者可以自行调整的空间）。
- **`DeviationWeightedFusion` 与 `ReliabilityGatedFusion` 的关键区别**（写代码时反复对照，避免误抄 RG 的分支级逻辑）：RG 门控输入是"分支偏离摘要"（3维标量：norm/max_abs/frac_exceed）经 softmax 竞争产出2个分支权重；本设计是**逐特征**独立 sigmoid 加权，没有 softmax、没有 gate_mlp、没有 `ep_encoder`/`svc_encoder`/`value_ep`/`value_svc`——整个类**没有一个 `nn.Linear`，零可学习参数**。唯一状态是 `EndpointBaselineStats` 引用和两个退化列 mask buffer。
- **`EndpointBaselineStats` 不做任何修改**（spec §3.1/§3.3 已确认）。它已提供：
  - `branch_stats(endpoint_key, branch) -> (mean: np.ndarray, std: np.ndarray)`（`branch` 是 `"ep"` 或 `"svc"`，各自逐列，`"svc"` 内部已经是 `service_metric`+`service_log` 合并列），未知 `endpoint_key` 抛 `KeyError`。
  - `degenerate_columns(branch) -> list[str]`：该分支在整个 fit 集合上全局退化（零方差/全 NaN，std 已兜底成 `_FALLBACK_STD_EPSILON=1e-9`）的**列名**列表（不是索引）。
  - `fitted_endpoints() -> set[str]`：fit 集合里出现过的 endpoint 全集。
- **列名→张量位置索引的映射从哪来**：`EndpointBaselineStats` 只暴露列名（`degenerate_columns()`），不暴露列的顺序列表（`_red_cols`/`_svc_cols` 是私有属性，本设计不触碰它们，避免依赖另一个模块的实现细节）。真正权威的列顺序来源是 contract 产物 `schema.json` 的 `feature_groups`（已验证真实产物 `artifacts/contract_v1_expanded/schema.json` 结构：`feature_groups.endpoint_red.columns`/`feature_groups.service_metric.columns`/`feature_groups.service_log.columns`，均为字符串列表，顺序即 `ContractDataset` 构造张量时的列顺序）。`DeviationWeightedFusion.__init__` 因此需要显式接收 `red_cols: list[str]`、`svc_cols: list[str]`（后者是 `service_metric` 列 + `service_log` 列拼接，与 `forward()` 里 `torch.cat([service_metric, service_log])` 的顺序严格一致），在构造时把 `degenerate_columns()` 的列名与这两份列表做集合比对，预算出两个 bool mask，注册为 buffer（不是 parameter，不参与优化器）。`from_contract` 负责从 `schema.json` 读出这两份列表并传入构造函数；单元测试里手工传入合成列表。
- **`endpoint_id_mapping.py` 复用方式与 RG 完全一致**：`from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key`，`_EP_TO_SVC_PATH = Path(__file__).resolve().parents[2] / "configs/contract/endpoint_to_service.yaml"`（`src/fusion/` 下 `parents[2]` 是 repo 根，与 `reliability_gate.py:33` 同款路径计算，直接照抄）。
- **`FusionModule.forward` 抽象签名已经包含 `endpoint_id` 参数**（`src/fusion/base.py:18-20`），`train_baseline_v0.py` 对所有 fusion 统一调用 `fusion({m: batch[m] for m in MODALITY_ORDER}, endpoint_id=batch.get("endpoint_id"))`（`_train`/`_infer`/`init_center` 三处，训练脚本本身**不需要任何改动**——`_collate` 已经在 v1 数据上产出 `endpoint_id` 张量，`from_contract` 分派机制已经支持"某些 fusion 需要额外运行时参数、某些不需要"，这套基础设施是 RG 落地时（entry 015）已经收束好的，本设计直接享用）。
- **`v1_expanded_pool.yaml` 是唯一可用 contract 配置**：`fit_endpoint_baseline_stats=true` 才会产出 `endpoint_baseline_stats.json`+触发 `endpoint_id` 列写入；`v1.yaml`（L0/L1/L2 用的）不产出这些，`from_contract` 在 sidecar 缺失时必须 fail fast（照抄 RG 的报错信息风格，指向"检查 `fit_endpoint_baseline_stats` 配置"）。
- **e2e 冒烟测试必须用 `tests/fixtures/nan_propagation_mini.yaml`，不是 `mini_dataset.yaml`**：`tests/test_e2e_reliability_gate_smoke.py` 已经踩过这个坑并写了详细注释——`mini_dataset.yaml` 在训练池扩容后会把某些 endpoint 的 `fault_baseline` 行吸收进 `train.parquet`/`eval_all`，但这些 endpoint 从未出现在该 fixture 的（单 Normal case、单 endpoint）`train_fit` 里，`EndpointBaselineStats` 只 fit `train_fit`，查表会抛 `KeyError`。`nan_propagation_mini.yaml` 的 Normal 与故障 case 共用同一 endpoint，不触发这个退化路径，本设计的 e2e 冒烟测试直接复用同一份 fixture。
- **真实数据现状**（已读取，供 Task 5 对比用，不用占位符）：`artifacts/baseline_v1_reliability_gate/metrics.json`（RG 在 `contract_v1_expanded` 上的现有结果，`n_samples=12683`）：`overall.auroc=0.5789`；`by_anomaly_type` 里 ABORT 四个 case 分别 `Lv_E_HTTPABORT_assurance=0.5937`/`order=0.8900`/`travel=0.2355`/`travel2=0.6689`，REPLACE 四个 case `assurance=0.7311`/`order=0.3149`/`travel=0.8354`/`travel2=0.7930`。oracle（entry 017 单特征 `client_error_rate`）ABORT/REPLACE ≈ 0.993/1.000。这些数字是 Task 5 判定"达标/部分达标/不达标"的直接对照基准。

---

## Task 1: `DeviationWeightedFusion` 核心类（逐特征加权 + 退化列处理 + endpoint_id 兜底）

**Files:**
- Create: `src/fusion/deviation_weighted.py`
- Test: `tests/test_fusion_deviation_weighted.py`

**背景**：spec §2.1/§2.2/§2.4/§2.5。本 Task 只实现"手工传入 `EndpointBaselineStats`+`id_to_endpoint_key`+`red_cols`+`svc_cols`"的直接构造路径（与 `tests/test_reliability_gated_fusion.py` 的 `_fusion()` fixture 同款风格），不涉及 `from_contract`（Task 3 单独做）。

- [ ] **Step 1: 写失败的测试（形状 + 手算公式正确性）**

`tests/test_fusion_deviation_weighted.py`：
```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_fusion_deviation_weighted.py -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'src.fusion.deviation_weighted'`

- [ ] **Step 3: 补充剩余测试（退化列固定权重1、endpoint_id=None 兜底、接口校验、unknown endpoint_id）**

追加到 `tests/test_fusion_deviation_weighted.py`：
```python
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
```

- [ ] **Step 4: 跑测试确认全部失败**

Run: `conda run -n interface python -m pytest tests/test_fusion_deviation_weighted.py -v`
Expected: FAIL（同 Step 2，全部因 `ModuleNotFoundError` 收集失败）

- [ ] **Step 5: 写实现 `src/fusion/deviation_weighted.py`**

```python
"""DeviationWeightedFusion——逐特征可信度加权融合（最小改动验证信号淹没假设）。

与 ReliabilityGatedFusion 的核心区别：没有 softmax/gate_mlp、没有编码器、零可
学习参数。每个原始特征独立按 sigmoid(|z|-threshold) 加权后直接 concat，z 是
该特征相对其所属 endpoint 自身 normal 基线的 z-score。是 L0 EarlyConcatFusion
的单变量对照（只多了"是否逐特征加权"这一个变量），详见 spec §2.2 对比表。
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
import torch.nn as nn
import yaml
from omegaconf import DictConfig

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule

# |z|>此阈值算作"该特征显著偏离"，直接复用 ReliabilityGatedFusion 的既有语义
# （_DEVIATION_THRESHOLD，见 src/fusion/reliability_gate.py:27），避免同一项目内
# 出现两套不一致的偏离显著性标准（spec §2.3）。
_DEVIATION_THRESHOLD = 2.0

# endpoint_to_service.yaml 是 id_to_endpoint_key 反查表的唯一权威来源，必须与
# build_contract.py 派生 endpoint_id 列时用的同一份 sorted() 逻辑一致
# (src/contracts/endpoint_id_mapping.py)。与 reliability_gate.py 同款路径计算。
_EP_TO_SVC_PATH = Path(__file__).resolve().parents[2] / "configs/contract/endpoint_to_service.yaml"


class DeviationWeightedFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        endpoint_baseline_stats: EndpointBaselineStats,
        id_to_endpoint_key: dict[int, str],
        red_cols: list[str],
        svc_cols: list[str],
        threshold: float = _DEVIATION_THRESHOLD,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        self._threshold = threshold
        self._ep_dim = modality_dims["endpoint_red"]
        self._svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        # 退化列（EndpointBaselineStats 在整个 fit 集合上全局零方差/全 NaN，std 已
        # 兜底为 epsilon）的权重强制为1，不参与 sigmoid(|z|-threshold) 计算——除以
        # epsilon 会把任何非零偏移放大成虚假的"显著偏离"信号，这不是信号淹没问题
        # 要救的真实信号（spec §2.4）。mask 在构造时算好、注册为 buffer（非
        # parameter，不参与优化器，但跟 .to(device) 一起搬）——这是本类唯一的状态，
        # 没有 nn.Linear、零可学习参数。
        degenerate_ep = set(endpoint_baseline_stats.degenerate_columns("ep"))
        degenerate_svc = set(endpoint_baseline_stats.degenerate_columns("svc"))
        self.register_buffer(
            "_degenerate_mask_ep",
            torch.tensor([c in degenerate_ep for c in red_cols], dtype=torch.bool),
        )
        self.register_buffer(
            "_degenerate_mask_svc",
            torch.tensor([c in degenerate_svc for c in svc_cols], dtype=torch.bool),
        )

    def _weight(self, raw: torch.Tensor, mean, std, degenerate_mask: torch.Tensor) -> torch.Tensor:
        z = (raw - torch.as_tensor(mean, dtype=torch.float32, device=raw.device)) / torch.as_tensor(
            std, dtype=torch.float32, device=raw.device
        )
        w = torch.sigmoid(z.abs() - self._threshold)
        return torch.where(degenerate_mask.to(raw.device), torch.ones_like(w), w)

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_ep = modality_dict["endpoint_red"]
        raw_svc = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)

        if endpoint_id is None:
            # 没有 endpoint 信息可用（L0/L1/L2 兼容调用路径）——全部特征权重为1，
            # 退化为纯 concat，等价 L0 EarlyConcatFusion（spec §2.5）。
            return torch.cat([raw_ep, raw_svc], dim=-1)

        weighted_eps, weighted_svcs = [], []
        for i in range(raw_ep.shape[0]):
            eid = int(endpoint_id[i].item())
            if eid not in self._id_to_key:
                raise KeyError(
                    f"endpoint_id={eid} 不在 id_to_endpoint_key 映射中，"
                    "检查上游 endpoint_id 编码是否与 fusion 构造时传入的映射一致"
                )
            key = self._id_to_key[eid]
            ep_mean, ep_std = self._baseline.branch_stats(key, "ep")
            svc_mean, svc_std = self._baseline.branch_stats(key, "svc")
            weighted_eps.append(
                raw_ep[i] * self._weight(raw_ep[i], ep_mean, ep_std, self._degenerate_mask_ep)
            )
            weighted_svcs.append(
                raw_svc[i] * self._weight(raw_svc[i], svc_mean, svc_std, self._degenerate_mask_svc)
            )
        return torch.cat([torch.stack(weighted_eps), torch.stack(weighted_svcs)], dim=-1)

    @property
    def output_dim(self) -> int:
        return self._ep_dim + self._svc_dim
```

- [ ] **Step 6: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_fusion_deviation_weighted.py -v`
Expected: PASS（7 个测试全绿）

- [ ] **Step 7: Commit**

```bash
git add src/fusion/deviation_weighted.py tests/test_fusion_deviation_weighted.py
git commit -m "[Feature]: 新增 DeviationWeightedFusion，逐特征偏离量加权融合"
```

---

## Task 2: `configs/fusion/deviation_weighted.yaml` + Hydra instantiate 测试

**背景**：与 RG 同理（见既有 `configs/fusion/reliability_gate.yaml`），`endpoint_baseline_stats`/`id_to_endpoint_key`/`red_cols`/`svc_cols` 是运行时才知道的构造参数，不进 yaml；yaml 只放 `_target_`+静态超参 `threshold`。

**Files:**
- Create: `configs/fusion/deviation_weighted.yaml`
- Test: `tests/test_hydra_instantiate.py`（追加）

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_hydra_instantiate.py`：
```python
def test_fusion_deviation_weighted_instantiate_returns_deviation_weighted_fusion():
    import pandas as pd

    from src.data.endpoint_baseline_stats import EndpointBaselineStats
    from src.fusion.deviation_weighted import DeviationWeightedFusion

    red_cols = [f"endpoint_red__f{i}" for i in range(10)]
    svc_cols = [f"service_metric__g{i}" for i in range(5)] + [
        f"service_log__h{i}" for i in range(3)
    ]
    stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    rows = [{"endpoint_key": "epA", **{c: 1.0 for c in red_cols + svc_cols}} for _ in range(20)]
    stats.fit(pd.DataFrame(rows))

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/deviation_weighted")
    fusion = hydra.utils.instantiate(
        cfg.fusion,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA"},
        red_cols=red_cols,
        svc_cols=svc_cols,
    )
    assert isinstance(fusion, DeviationWeightedFusion)
    assert fusion.output_dim == 18
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_hydra_instantiate.py -v -k deviation_weighted`
Expected: FAIL——`hydra.errors.MissingConfigException: Cannot find primary config 'fusion/deviation_weighted'`

- [ ] **Step 3: 写 config 文件**

`configs/fusion/deviation_weighted.yaml`：
```yaml
_target_: src.fusion.deviation_weighted.DeviationWeightedFusion
threshold: 2.0
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_hydra_instantiate.py -v -k deviation_weighted`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/fusion/deviation_weighted.yaml tests/test_hydra_instantiate.py
git commit -m "[Feature]: 新增 configs/fusion/deviation_weighted.yaml，支持 hydra instantiate"
```

---

## Task 3: `from_contract` 覆写（加载 baseline stats + schema 列名 + id 映射）

**Files:**
- Modify: `src/fusion/deviation_weighted.py`
- Modify: `tests/test_fusion_from_contract.py`

**背景**：spec §3.2。训练脚本调用 `hydra.utils.get_class(cfg.fusion._target_).from_contract(cfg.fusion, contract_dir=contract_dir, modality_dims=modality_dims)`（`scripts/train_baseline_v0.py:132-134`），本身不知道也不需要知道 `DeviationWeightedFusion` 需要额外的 `endpoint_baseline_stats`/`id_to_endpoint_key`/`red_cols`/`svc_cols`——这套解耦是 RG 落地时（entry 015）已经收束好的模式，本 Task 只是给新类照抄同一套。

- [ ] **Step 1: 写失败的测试**

在 `tests/test_fusion_from_contract.py` 顶部追加 import（该文件目前没有 `import json`）：
```python
import json
```
并追加：
```python
from src.fusion.deviation_weighted import DeviationWeightedFusion
```

追加测试函数：
```python
def test_deviation_weighted_from_contract_loads_baseline_stats_and_schema(tmp_path):
    """DeviationWeightedFusion 覆写 from_contract：从 contract_dir 读
    endpoint_baseline_stats.json + schema.json（后者提供逐列列名，RG 不需要这份，
    因为 RG 只算分支级偏离摘要，不需要知道单列名字；本设计需要把 degenerate_columns()
    返回的列名映射到张量位置，schema.json 的 feature_groups 是列顺序的权威来源）。"""
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    key = sorted(ep_to_svc)[0]
    dims = {"endpoint_red": 2, "service_metric": 1, "service_log": 1}
    schema = {
        "feature_groups": {
            "endpoint_red": {"columns": ["endpoint_red__a", "endpoint_red__b"]},
            "service_metric": {"columns": ["service_metric__c"]},
            "service_log": {"columns": ["service_log__d"]},
        }
    }
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    df = pd.DataFrame(
        {
            "endpoint_key": [key, key],
            "endpoint_red__a": [0.1, 0.2],
            "endpoint_red__b": [0.3, 0.5],
            "service_metric__c": [0.4, 0.6],
            "service_log__d": [0.2, 0.3],
        }
    )
    stats = EndpointBaselineStats(
        red_cols=["endpoint_red__a", "endpoint_red__b"],
        svc_cols=["service_metric__c", "service_log__d"],
    )
    stats.fit(df)
    stats.save(tmp_path / "endpoint_baseline_stats.json")

    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    fusion = DeviationWeightedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=dims)
    assert isinstance(fusion, DeviationWeightedFusion)
    assert fusion.output_dim == 4
    assert fusion._id_to_key == _derive_id_to_key(ep_to_svc)
    assert key in fusion._baseline.fitted_endpoints()


def test_deviation_weighted_from_contract_missing_sidecar_raises_clear_error(tmp_path):
    """与 RG 同款约定：sidecar 缺失必须 fail fast 且报错信息指向
    fit_endpoint_baseline_stats 配置项，不是裸 FileNotFoundError 只报路径。
    sidecar 检查在读 schema.json 之前，contract_dir 空目录即可触发。"""
    cfg = OmegaConf.create(
        {"_target_": "src.fusion.deviation_weighted.DeviationWeightedFusion", "threshold": 2.0}
    )
    with pytest.raises(FileNotFoundError, match="fit_endpoint_baseline_stats"):
        DeviationWeightedFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_fusion_from_contract.py -v -k deviation_weighted`
Expected: FAIL——`AttributeError: type object 'DeviationWeightedFusion' has no attribute 'from_contract'` 会走基类默认实现（不报 AttributeError，而是走 `FusionModule.from_contract` 默认路径，因为 `hydra.utils.instantiate` 会收到多余的 `endpoint_baseline_stats`/`red_cols`/`svc_cols` 关键字参数导致 `TypeError`），或直接因缺少这些参数报 `TypeError: __init__() missing required positional argument`

- [ ] **Step 3: 给 `src/fusion/deviation_weighted.py` 追加 `from_contract` 覆写**

在 `DeviationWeightedFusion` 类内追加（`output_dim` property 之后）：
```python
    @classmethod
    def from_contract(
        cls, cfg: DictConfig, *, contract_dir: Path, modality_dims: dict[str, int]
    ) -> "DeviationWeightedFusion":
        """从 contract_dir 加载 per-endpoint 基线统计量 + schema 列名，派生
        id_to_endpoint_key 反查表，交给 hydra.utils.instantiate 注入这些运行时对象。
        与 ReliabilityGatedFusion.from_contract 同一模式（entry 015 收束的
        构造解耦），训练脚本对"本类需要什么"一无所知即可正确构造。"""
        sidecar = Path(contract_dir) / "endpoint_baseline_stats.json"
        if not sidecar.exists():
            raise FileNotFoundError(
                f"{sidecar} 不存在——DeviationWeightedFusion 需要 contract 构建时开启"
                " fit_endpoint_baseline_stats=true（见 configs/contract/v1_expanded_pool.yaml），"
                f"检查 {contract_dir} 是否是用 fit_endpoint_baseline_stats=false 的配置"
                "（如 v1.yaml）构建的"
            )
        baseline_stats = EndpointBaselineStats.load(sidecar)
        schema = json.loads((Path(contract_dir) / "schema.json").read_text())
        red_cols = schema["feature_groups"]["endpoint_red"]["columns"]
        svc_cols = (
            schema["feature_groups"]["service_metric"]["columns"]
            + schema["feature_groups"]["service_log"]["columns"]
        )
        ep_to_svc = yaml.safe_load(_EP_TO_SVC_PATH.read_text())
        id_to_key = _derive_id_to_endpoint_key(ep_to_svc)
        return hydra.utils.instantiate(
            cfg,
            modality_dims=modality_dims,
            endpoint_baseline_stats=baseline_stats,
            id_to_endpoint_key=id_to_key,
            red_cols=red_cols,
            svc_cols=svc_cols,
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_fusion_from_contract.py -v`
Expected: PASS（含既有 L0/L1/L2/RG 测试 + 本 Task 新增 2 个，全绿，无回归）

- [ ] **Step 5: 跑 Task 1/2 既有测试确认无回归**

Run: `conda run -n interface python -m pytest tests/test_fusion_deviation_weighted.py tests/test_hydra_instantiate.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/fusion/deviation_weighted.py tests/test_fusion_from_contract.py
git commit -m "[Feature]: DeviationWeightedFusion.from_contract 加载 baseline stats + schema 列名"
```

---

## Task 4: DVC pipeline 接入 + e2e 冒烟测试

**背景**：spec §3.4/§4.1。`DeviationWeightedFusion` 与 RG 用**同一份** contract（`v1_expanded_pool.yaml` → `artifacts/contract_v1_expanded/`），这是能与 RG 直接对比的硬性前提（spec §4）。不重新定义 build 阶段——`dvc_reliability_gate/dvc.yaml` 的 `build_contract_v1_expanded` stage 已经产出这些文件，新建的 `dvc_deviation_weighted/dvc.yaml` 只加 train/eval 两个 stage，直接把 `artifacts/contract_v1_expanded/*` 列为 `deps`（DVC 的依赖图是仓库级全局的，不局限于单个 `dvc.yaml` 文件——只要路径的 hash 记录在 `dvc.lock` 里，跨文件引用同一产物是受支持的标准用法，`dvc repro` 能正确检测上游 stage 是否需要先跑）。

**Files:**
- Create: `dvc_deviation_weighted/dvc.yaml`
- Test: `tests/test_e2e_deviation_weighted_smoke.py`

- [ ] **Step 1: 写失败的 e2e smoke 测试**

`tests/test_e2e_deviation_weighted_smoke.py`：
```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]


def test_deviation_weighted_e2e_on_mini_fixture(tmp_path):
    # 复用 RG e2e 冒烟测试同款 fixture 选择理由（见 test_e2e_reliability_gate_smoke.py
    # 的详细注释）：DeviationWeightedFusion 同样需要 EndpointBaselineStats 覆盖
    # eval_all 里出现的全部 endpoint，nan_propagation_mini.yaml 的 Normal/故障 case
    # 共用同一 endpoint，不会触发稀疏 endpoint 覆盖不全的 KeyError 边界情形。
    contract_dir = tmp_path / "contract_v1"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1_expanded_pool.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/nan_propagation_mini.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    scores_path = tmp_path / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={scores_path}",
            "seed=1",
            "training.epochs=2",
            "fusion=deviation_weighted",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(scores_path)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_e2e_deviation_weighted_smoke.py -v`
Expected: FAIL——`hydra.errors.MissingConfigException` 或 `ConfigAttributeError`（`fusion=deviation_weighted` override 尚未在 `configs/fusion/` 下有效——若 Task 2 已完成，此步应能跑到 `from_contract`/训练逻辑，具体报错取决于 Task 1-3 实现细节是否已经落地；若本 Task 独立执行且 Task 1-3 已完成，直接跳到 Step 3 调试）

- [ ] **Step 3: 调试直到通过**

常见排查方向：确认 `configs/fusion/deviation_weighted.yaml` 存在（Task 2）；确认 `from_contract` 覆写已落地（Task 3）；确认 `nan_propagation_mini.yaml` fixture 的 endpoint 覆盖面在训练池扩容后仍与 `train_fit` 一致（若不一致会在 `forward()` 里对某些 batch 样本抛 `KeyError`，与 RG e2e 冒烟测试当年踩的坑同源，见背景说明）。

- [ ] **Step 4: 跑测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_e2e_deviation_weighted_smoke.py -v`
Expected: PASS

- [ ] **Step 5: 新增 `dvc_deviation_weighted/dvc.yaml`**

```yaml
# DeviationWeightedFusion 专属 pipeline，从根 dvc.yaml 隔离出来（同
# dvc_reliability_gate/dvc.yaml 的隔离模式，见该文件顶部注释）。复用
# dvc_reliability_gate/dvc.yaml 的 build_contract_v1_expanded stage 已产出的
# artifacts/contract_v1_expanded/*（与 RG 用同一份 contract，spec
# 2026-07-27-deviation-weighted-fusion-design.md §4 的硬性要求：必须同 contract
# 才能直接对比），本文件不重新定义 build 阶段。
# 触发：`dvc repro dvc_deviation_weighted/dvc.yaml`（需先确保
# artifacts/contract_v1_expanded/ 已由 `dvc repro dvc_reliability_gate/dvc.yaml`
# 产出）。DVC 3.67.1 强制 pipeline 文件名字面量为 dvc.yaml，故放子目录。
stages:
  train_v1_deviation_weighted:
    wdir: ..
    cmd: python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1_expanded
      out=artifacts/baseline_v1_deviation_weighted/scores.parquet seed=42
      training.epochs=50 fusion=deviation_weighted model=deep_svdd
    deps:
      - scripts/train_baseline_v0.py
      - configs/base.yaml
      - configs/fusion/deviation_weighted.yaml
      - configs/model/deep_svdd.yaml
      - configs/training/default.yaml
      - configs/contract/endpoint_to_service.yaml
      - src/models/deep_svdd.py
      - src/fusion/deviation_weighted.py
      - src/fusion/base.py
      - src/data/contract_dataloader.py
      - src/data/endpoint_baseline_stats.py
      - artifacts/contract_v1_expanded/train.parquet
      - artifacts/contract_v1_expanded/eval_all.parquet
      - artifacts/contract_v1_expanded/schema.json
      - artifacts/contract_v1_expanded/endpoint_baseline_stats.json
    outs:
      - artifacts/baseline_v1_deviation_weighted/scores.parquet

  eval_v1_deviation_weighted:
    wdir: ..
    cmd: python scripts/eval_baseline_v0.py
      --scores artifacts/baseline_v1_deviation_weighted/scores.parquet
      --out artifacts/baseline_v1_deviation_weighted/metrics.json
    deps:
      - scripts/eval_baseline_v0.py
      - src/contracts
      - artifacts/baseline_v1_deviation_weighted/scores.parquet
    metrics:
      - artifacts/baseline_v1_deviation_weighted/metrics.json:
          cache: false
```

- [ ] **Step 6: 验证跨文件依赖能被 DVC 正确解析**

Run: `dvc dag dvc_deviation_weighted/dvc.yaml:train_v1_deviation_weighted`
Expected: 输出的依赖图应能追溯到 `dvc_reliability_gate/dvc.yaml:build_contract_v1_expanded`（或至少确认 `artifacts/contract_v1_expanded/train.parquet` 等路径被识别为该 stage 的已知输出，不报"unknown dependency"类错误）。若 DVC 版本对跨文件 stage 引用有限制导致此步报错，退回方案：把 `train_v1_deviation_weighted`/`eval_v1_deviation_weighted` 两个 stage 直接追加进 `dvc_reliability_gate/dvc.yaml` 末尾（该文件顶部注释需同步更新说明它现在承载两个融合机制的实验 stage，不再是 RG 专属），不新建 `dvc_deviation_weighted/dvc.yaml`。

- [ ] **Step 7: Commit**

```bash
git add dvc_deviation_weighted/dvc.yaml tests/test_e2e_deviation_weighted_smoke.py
git commit -m "[Feature]: deviation_weighted 端到端 dvc stage + smoke 测试"
```

---

## Task 5: 主实验——真实数据训练评估 + 对照 spec §5 判定标准

**背景**：spec §4/§5。本 Task 不写新代码，是用 Task 1-4 跑通的管线在真实数据（`artifacts/contract_v1_expanded/`，已由 RG 链路产出，无需重新 build）上训练评估，headline 指标用 `by_anomaly_type`（spec §4 已确认理由：`overall` 跨run污染），对照 ABORT/REPLACE 两类 case 的 AUROC 判定达标/部分达标/不达标。

**Files:**
- 产出: `artifacts/baseline_v1_deviation_weighted/scores.parquet`、`artifacts/baseline_v1_deviation_weighted/metrics.json`

- [ ] **Step 1: 跑通训练+评估**

```bash
dvc repro dvc_deviation_weighted/dvc.yaml
```
Expected: `train_v1_deviation_weighted`/`eval_v1_deviation_weighted` 两个 stage 跑通（若 Task 4 Step 6 走了退回方案，改为 `dvc repro dvc_reliability_gate/dvc.yaml`），产出 `artifacts/baseline_v1_deviation_weighted/metrics.json`。若 DVC 检测到 `artifacts/contract_v1_expanded/` 尚不存在或已过期，会先跑 `build_contract_v1_expanded`——这是预期行为，不是本 Task 的 bug。

- [ ] **Step 2: 读取并记录 ABORT/REPLACE 分层 AUROC**

```bash
python -c "
import json
d = json.load(open('artifacts/baseline_v1_deviation_weighted/metrics.json'))
bt = d['stratified']['by_anomaly_type']
for k, v in sorted(bt.items()):
    if 'ABORT' in k or 'REPLACE' in k:
        print(k, v)
print('n_samples:', d['n_samples'])
"
```
把读到的真实数值记入下一步判定，不用占位符。

- [ ] **Step 3: 对照 spec §5 三档标准判定**

- **达标**：ABORT/REPLACE 各 case 的 AUROC 宏平均 ≥0.8，且 DELAY/PATCH（若数据里存在，如实读取 `by_anomaly_type` 里其余 case）无明显退步（允许持平/小幅波动）→ 记录"方向验证成功，值得深化"，下一步候选见 spec §2.6（可学习阈值等），但**本计划不包含深化任务**（用户已明确要求分开决策，深化是否做需要用户看到本 Task 结果后另行决定）。
- **部分达标**：有改善但离 oracle（0.993/1.000）明显有距离（如接近 0.4→0.6 区间）→ 记录"部分改善，需诊断卡在融合层还是打分层"，Task 6（诊断脚本）按此结果决定是否执行。
- **不达标**：与 L0/RG 基本无差异甚至更差（对照本文档"背景说明"里列出的 RG 现有数字：ABORT 0.594/0.890/0.236/0.669，REPLACE 0.731/0.315/0.835/0.793）→ 记录"不达标，不做后续深化"，直接进入 Task 7 写 history entry 如实报告，跳过 Task 6。

如实记录判定结果（三档之一）到本次会话/交给 Task 7 使用，不预设哪一档会命中。

---

## Task 6（条件执行，仅当 Task 5 判定为"部分达标"时执行）：逐特征权重诊断脚本

**背景**：spec §5"部分达标"分支要求的诊断手段，类比 `scripts/analyze_gate_weights.py`（RG 的门控权重按 anomaly_type 分层分析脚本）。**若 Task 5 判定为"达标"或"不达标"，跳过本 Task**，直接进入 Task 7。

**Files:**
- Create: `scripts/analyze_deviation_weights.py`（一次性分析脚本，非管线常驻组件）

- [ ] **Step 1: 写 `scripts/analyze_deviation_weights.py`**

```python
#!/usr/bin/env python
"""按 anomaly_type 分层统计 DeviationWeightedFusion 逐特征权重的均值，用于
诊断"部分达标"场景下瓶颈在融合层还是打分层——若 client_error_rate 对应特征列
的权重在故障样本上确实被推高到接近1（融合层已经放大了信号），但整体 AUROC
仍不够，说明瓶颈更可能在 DeepSVDD 等权 L2 距离打分公式（spec §2.6 排除项，
仅当此诊断指向该方向时才值得在未来轮次考虑）。一次性分析脚本，不是训练/评估
管线的一部分，直接构造 fusion（不经 Hydra cfg 中转），与
scripts/analyze_gate_weights.py 同款风格。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
from src.data.contract_dataloader import ContractDataset
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER
from src.fusion.deviation_weighted import DeviationWeightedFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", required=True)
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    schema = json.loads((contract_dir / "schema.json").read_text())
    modality_dims = {n: len(g["columns"]) for n, g in schema["feature_groups"].items()}
    red_cols = schema["feature_groups"]["endpoint_red"]["columns"]
    svc_cols = (
        schema["feature_groups"]["service_metric"]["columns"]
        + schema["feature_groups"]["service_log"]["columns"]
    )
    baseline_stats = EndpointBaselineStats.load(contract_dir / "endpoint_baseline_stats.json")
    ep_to_svc = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    id_to_key = _derive_id_to_endpoint_key(ep_to_svc)

    fusion = DeviationWeightedFusion(
        modality_dims=modality_dims,
        endpoint_baseline_stats=baseline_stats,
        id_to_endpoint_key=id_to_key,
        red_cols=red_cols,
        svc_cols=svc_cols,
    )
    fusion.eval()

    ds = ContractDataset(
        parquet_path=contract_dir / "eval_all.parquet",
        schema_path=contract_dir / "schema.json",
        nan_strategy="mean",
        fit_on_parquet=contract_dir / "train.parquet",
    )
    eval_df = pd.read_parquet(contract_dir / "eval_all.parquet")
    error_rate_idx = red_cols.index("endpoint_red__client_error_rate")

    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            batch = {m: sample[m].unsqueeze(0) for m in MODALITY_ORDER}
            eid = torch.tensor([sample["meta"]["endpoint_id"]])
            key = fusion._id_to_key[int(eid.item())]
            ep_mean, ep_std = fusion._baseline.branch_stats(key, "ep")
            w_ep = fusion._weight(
                batch["endpoint_red"][0], ep_mean, ep_std, fusion._degenerate_mask_ep
            )
            records.append(
                {
                    "sample_id": sample["meta"]["sample_id"],
                    "w_client_error_rate": w_ep[error_rate_idx].item(),
                }
            )

    weights_df = pd.DataFrame(records).merge(
        eval_df[["sample_id", "anomaly_type"]], on="sample_id"
    )
    summary = weights_df.groupby("anomaly_type")["w_client_error_rate"].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑分析脚本，记录结果**

Run: `conda run -n interface python scripts/analyze_deviation_weights.py --contract-dir artifacts/contract_v1_expanded`

把 ABORT/REPLACE 各 case 的 `w_client_error_rate` 均值记入 Task 7 的 history entry——若权重确实被推高（接近1）但 AUROC 仍不够，如实记录"融合层已放大信号，瓶颈更可能在 SVDD 打分层"，这是留给用户决定是否值得在未来轮次改造 SVDD 打分公式的依据（本计划不包含该项，spec §2.6 已排除）。

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_deviation_weights.py
git commit -m "[Chore]: 新增逐特征权重诊断脚本（DeviationWeightedFusion 部分达标场景用）"
```

---

## Task 7: history entry + 最终验证

**Files:**
- Create: `history/entries/018-deviation-weighted-fusion.md`
- Modify: `history/index.md`

- [ ] **Step 1: 写 `history/entries/018-deviation-weighted-fusion.md`**

按 `history/entries/_template.md` 结构，仿照 entry 014/016/017 的写法。必须包含：
- **做了什么**：`DeviationWeightedFusion`（逐特征 sigmoid 加权 + 退化列固定权重1 + `endpoint_id=None`兜底 = L0）+ `from_contract` 接入 + `dvc_deviation_weighted/dvc.yaml`，在 `contract_v1_expanded` 上的真实 ABORT/REPLACE `by_anomaly_type` AUROC 数值（Task 5 Step 2 读到的真实数字，不用占位符），与 L0/RG/oracle 对比表。
- **关键决策**：为什么排除分支级门控（避免复现 RG softmax 坍缩，见 spec §2.2）；为什么退化列固定权重1 而非跳过（spec §2.4，避免除零放大假信号）；为什么用 `by_anomaly_type` 而非 `overall` 作 headline（spec §4，跨run污染，archival 记录）；为什么 DVC pipeline 复用 RG 的 contract build 阶段而不重新 build（spec §4，必须同 contract 才能直接对比）。
- **坑/已知问题**：若 Task 4/5 调试中遇到实际问题（如 DVC 跨文件依赖解析、mini fixture 稀疏 endpoint 覆盖）如实记录；DELAY/PATCH（若在数据里存在困难样本）如实报告是否被拖累。
- **达标判定结果**（Task 5 三档之一，如实记录，不掩盖不达标的可能性）：若达标/部分达标，记录"下一步候选（可学习阈值/逐维可学习权重/改造SVDD打分公式）留给用户决定是否开新一轮设计"；若不达标，记录"不做后续深化，回头重新审视机制假设本身"（spec §5/§7 风险1 已预先声明接受这种结果）。
- **遗留 TODO**：spec §8 明确排除项原样搬入（不做分支级门控、不做可学习阈值、不改 SVDD 打分公式、不修复跨run污染切分代码、不新增第三分支）。

- [ ] **Step 2: 更新 `history/index.md`**

Entry 列表追加一行（编号018，日期为实施当天），影响域索引表追加：`src/fusion/`（`deviation_weighted.py`）、`configs/fusion/`、`dvc_deviation_weighted/`、`scripts/`（若 Task 6 执行了，追加 `analyze_deviation_weights.py`）。

- [ ] **Step 3: 跑完整测试套件确认无回归**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: PASS（Task 1-4 全部新增测试 + 所有既有测试无回归，包括 L0/L1/L2/RG fusion、Contract v0/v1、Normalizer、EndpointBaselineStats）

- [ ] **Step 4: 对照 spec 完成标准逐项自查**

- [ ] spec §4 训练/评估协议：用同一份 `v1_expanded_pool.yaml` contract，headline 用 `by_anomaly_type`，已在 Task 5 完成
- [ ] spec §5 成功判定：三档之一已确定并如实记录（不掩盖不达标可能性）
- [ ] spec §6 测试要求：5 项要求测试全部覆盖（加权公式数值正确性/退化列固定权重1/`endpoint_id=None`兜底/`output_dim`正确性/`modality_dims`校验）——对照 Task 1 的 7 个测试逐一核对
- [ ] spec §8 明确排除项：未在本计划任何 Task 中被违反（无分支级门控、无可学习阈值、无 SVDD 打分公式改动、无跨run污染切分代码修复、无第三分支拆分）
- [ ] history entry 018 已写，index.md 已更新

- [ ] **Step 5: Commit**

```bash
git add history/entries/018-deviation-weighted-fusion.md history/index.md
git commit -m "[Docs]: DeviationWeightedFusion history entry + 达标判定记录"
```

---

## 最终验证

- [ ] **跑完整测试套件**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: PASS（全部新增/修改测试 + 所有既有测试无回归）

- [ ] **既有 pipeline 冒烟（确认未被本轮改动破坏）**

Run: `dvc repro dvc_reliability_gate/dvc.yaml`
Expected: RG 链路照常跑通，产出不受影响（本计划未修改 `src/fusion/reliability_gate.py`/`src/data/endpoint_baseline_stats.py`/contract build 逻辑）。

- [ ] **deviation_weighted 新链路端到端冒烟（真实数据）**

Run: `dvc repro dvc_deviation_weighted/dvc.yaml`
Expected: 跑通，`artifacts/baseline_v1_deviation_weighted/metrics.json` 产出，数值与 Task 5/7 记录的一致（若此时重跑数值有变化，说明环境/随机种子未固定，需要排查，不应该发生——`seed=42` 已固定）。
