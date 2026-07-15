# Fusion Ablation Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 L0-L3 融合机制消融实验所需的三块基础设施——Hydra 可插拔 fusion/model、train/eval 无重叠的三路时序切分（Contract v1）、参数量对齐工具——不实现门控融合本身，不跑消融对比实验。

**Architecture:** 训练脚本从纯 argparse 改为 `@hydra.main` 入口，融合/模型通过 `hydra.utils.instantiate(cfg.fusion/model)` 构造，不再硬编码具体类；`build_contract.py` 新增 v1 产出路径，按每个 Normal case 内的时间窗时序切分（前 60%/中 20%/后 20%）把 Normal 行切成 `train_fit`/`train_val`/`eval_normal_holdout` 三个不重叠子集（每个 endpoint 在三份都出现，训练窗早于评估窗），v0 pipeline 原样保留并存；新增纯函数 `solve_hidden_dim_for_param_budget` 做二分查找，本轮只提供工具不接入任何训练脚本。

**Tech Stack:** Python, PyTorch, Hydra 1.3, OmegaConf, pandas, pytest, DVC

**Spec:** `docs/superpowers/specs/2026-07-14-fusion-ablation-infra-design.md`

---

## Task 1: 把 `MODALITY_ORDER` 从 `EarlyConcatFusion` 挪到 `src/fusion/base.py`

**背景**：现状 `MODALITY_ORDER` 是 `EarlyConcatFusion` 的类属性（`src/fusion/early_concat.py:11`），`scripts/train_baseline_v0.py:31` 靠 `EarlyConcatFusion.MODALITY_ORDER` 拿到模态列表——这把训练脚本和某一个具体融合子类绑死了。spec 要求把它挪到 `FusionModule` 所在的 `base.py`，做成模块级常量，任何融合子类都能共享，不绑定具体子类。

**Files:**
- Modify: `src/fusion/base.py`
- Modify: `src/fusion/early_concat.py:9-28`
- Test: `tests/test_early_concat_fusion.py`（已存在，需确认仍通过）

- [ ] **Step 1: 在 `base.py` 加模块级常量**

`src/fusion/base.py` 当前内容：
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
```

改成：
```python
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# 三个模态的固定顺序，所有融合机制共享（不绑定任何具体子类）。
# 训练脚本据此拼 batch/tensor，顺序错位会导致特征维度错位但不报错——务必保持全局唯一来源。
MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
```

- [ ] **Step 2: 改 `EarlyConcatFusion` 引用 `base.MODALITY_ORDER`**

`src/fusion/early_concat.py` 改为：
```python
from __future__ import annotations

import torch

from src.fusion.base import MODALITY_ORDER, FusionModule


class EarlyConcatFusion(FusionModule):
    """Early concatenation：直接拼接所有 modality 特征。顺序固定为 MODALITY_ORDER。"""

    MODALITY_ORDER = MODALITY_ORDER  # 向后兼容别名：Task 4 之前仍有代码用 EarlyConcatFusion.MODALITY_ORDER

    def __init__(self, modality_dims: dict[str, int]):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._dims = modality_dims

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([modality_dict[m] for m in MODALITY_ORDER], dim=-1)

    @property
    def output_dim(self) -> int:
        return sum(self._dims[m] for m in MODALITY_ORDER)
```

保留 `MODALITY_ORDER` 类属性别名的原因：Task 4 会把 `scripts/train_baseline_v0.py:31` 的 `EarlyConcatFusion.MODALITY_ORDER` 换成 `from src.fusion.base import MODALITY_ORDER` 直接导入。这里保留别名只是让 Task 1 单独提交时不破坏任何现有引用；Task 4 完成后如确认无其他引用，可选择性清理（不在本任务删除，避免过早断链）。

- [ ] **Step 3: 跑现有测试确认不破坏**

Run: `pytest tests/test_early_concat_fusion.py -v`
Expected: PASS（4 个测试全绿，行为未变，只是常量来源变了）

- [ ] **Step 4: Commit**

```bash
git add src/fusion/base.py src/fusion/early_concat.py
git commit -m "[Refactor]: MODALITY_ORDER 移至 fusion/base.py，脱离具体子类绑定"
```

---

## Task 2: 新增 `configs/fusion/concat.yaml` 和 `configs/model/deep_svdd.yaml`，加 Hydra instantiate 测试

**背景**：spec 组件 1 要求融合机制和模型都能通过 Hydra config-group 切换。这一步只加 config 文件 + 验证 `hydra.utils.instantiate` 能正确构造出对象，不改训练脚本（Task 4 才改训练脚本本身）。

**Files:**
- Create: `configs/fusion/concat.yaml`
- Create: `configs/model/deep_svdd.yaml`
- Test: `tests/test_hydra_instantiate.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_hydra_instantiate.py`：
```python
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

from src.fusion.early_concat import EarlyConcatFusion
from src.models.deep_svdd import DeepSVDD

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_DIR = str(REPO_ROOT / "configs")


def test_fusion_concat_instantiate_returns_early_concat_fusion():
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/concat")
    fusion = hydra.utils.instantiate(
        cfg,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
    )
    assert isinstance(fusion, EarlyConcatFusion)
    assert fusion.output_dim == 18


def test_model_deep_svdd_instantiate_returns_deep_svdd():
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="model/deep_svdd")
    svdd = hydra.utils.instantiate(cfg, input_dim=18)
    assert isinstance(svdd, DeepSVDD)
    assert svdd.rep_dim == 32  # deep_svdd.yaml 默认值


def test_fusion_output_dim_feeds_svdd_input_dim():
    """fusion.output_dim 必须能直接喂给 svdd 的 input_dim，不需要手动改代码对齐维度。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        fusion_cfg = compose(config_name="fusion/concat")
        model_cfg = compose(config_name="model/deep_svdd")
    fusion = hydra.utils.instantiate(
        fusion_cfg,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
    )
    svdd = hydra.utils.instantiate(model_cfg, input_dim=fusion.output_dim)
    assert svdd.encoder[0].in_features == fusion.output_dim
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_hydra_instantiate.py -v`
Expected: FAIL，报错类似 `hydra.errors.MissingConfigException: Cannot find primary config 'fusion/concat'`（config 文件还不存在）

- [ ] **Step 3: 写 config 文件**

`configs/fusion/concat.yaml`：
```yaml
_target_: src.fusion.early_concat.EarlyConcatFusion
```

`configs/model/deep_svdd.yaml`：
```yaml
_target_: src.models.deep_svdd.DeepSVDD
hidden_dim: 64
rep_dim: 32
```

注：`hidden_dim`/`rep_dim` 取值沿用 `train_baseline_v0.py` 现有 argparse 默认值（`--hidden-dim` 默认 64，`--rep-dim` 默认 32），保证 Task 4 迁移后行为不变。`modality_dims`（fusion）和 `input_dim`（model）不写进 yaml，因为它们是运行时才知道的值（依赖 schema.json），由调用方在 `hydra.utils.instantiate(cfg.fusion, modality_dims=...)` 时以关键字参数补上——Hydra 允许 instantiate 时追加 config 里没有的参数。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_hydra_instantiate.py -v`
Expected: PASS（3 个测试全绿）

- [ ] **Step 5: Commit**

```bash
git add configs/fusion/concat.yaml configs/model/deep_svdd.yaml tests/test_hydra_instantiate.py
git commit -m "[Feature]: 新增 fusion/concat.yaml 与 model/deep_svdd.yaml，支持 hydra.utils.instantiate"
```

---

## Task 3: `configs/base.yaml` 填充 `model`/`fusion` 默认值，新增 `contract_dir`/`out` 必填字段

**背景**：spec 组件 1 要求把 `model: ???` 正式填上默认值（修复 CLAUDE.md 声称走 `hydra.utils.instantiate(cfg.model)` 但实际从未填充的脱节），同时加 `fusion` config-group 默认值，并新增 `contract_dir: ???`、`out: ???` 两个必填字段替代原来的 `--contract-dir`/`--out` argparse 参数。`???` 是 OmegaConf 的"强制字段"标记——不显式填充就访问会抛 `MissingMandatoryValue`，这是仿照 `model: ???` 已有的约定。

**Files:**
- Modify: `configs/base.yaml`
- Test: `tests/test_base_config.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_base_config.py`：
```python
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import MissingMandatoryValue

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_DIR = str(REPO_ROOT / "configs")


def test_base_config_composes_with_fusion_and_model_defaults():
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="base", overrides=["contract_dir=/tmp/c", "out=/tmp/o.parquet"])
    assert cfg.fusion._target_ == "src.fusion.early_concat.EarlyConcatFusion"
    assert cfg.model._target_ == "src.models.deep_svdd.DeepSVDD"
    assert cfg.contract_dir == "/tmp/c"
    assert cfg.out == "/tmp/o.parquet"


def test_base_config_missing_contract_dir_raises():
    """contract_dir/out 未通过 CLI override 填充时，访问该字段必须报错，而不是静默返回 None。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="base")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.contract_dir


def test_base_config_fusion_switchable_via_override():
    """本轮只新增 concat，但必须验证 config-group 切换机制本身工作——
    这是消融实验用 fusion=xxx 切换机制的前提。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="base",
            overrides=["contract_dir=/tmp/c", "out=/tmp/o.parquet", "fusion=concat"],
        )
    assert cfg.fusion._target_ == "src.fusion.early_concat.EarlyConcatFusion"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_base_config.py -v`
Expected: FAIL——`compose(config_name="base")` 报错找不到 `fusion` config group（`configs/base.yaml` 的 `defaults` 里还没有 `fusion` 条目），且 `cfg.contract_dir` 不存在会是 `ConfigAttributeError` 而不是 `MissingMandatoryValue`

- [ ] **Step 3: 改 `configs/base.yaml`**

改成：
```yaml
defaults:
  - data: anomod
  - model: deep_svdd
  - fusion: concat
  - training: default
  - _self_

# 可复现性
seed: 42

# 输出目录，按时间戳组织
output_dir: outputs/${now:%Y-%m-%d_%H-%M-%S}

# Contract 数据目录与训练产物输出路径，替代原 argparse 的 --contract-dir/--out
contract_dir: ???
out: ???
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_base_config.py -v`
Expected: PASS（3 个测试全绿）

- [ ] **Step 5: Commit**

```bash
git add configs/base.yaml tests/test_base_config.py
git commit -m "[Feature]: base.yaml 填充 model/fusion 默认值，新增 contract_dir/out 必填字段"
```

---

## Task 4: `scripts/train_baseline_v0.py` 改为 Hydra entrypoint，同步更新调用方

**背景**：spec 组件 1 的核心改动——训练脚本从纯 argparse 改为 `@hydra.main` 装饰的入口，融合/模型不再硬编码 `from src.fusion.early_concat import EarlyConcatFusion`，改用 `hydra.utils.instantiate`。这一步会破坏所有用 argparse flag（`--contract-dir` 等）调用该脚本的现有测试，必须同步改成 Hydra override 语法（`contract_dir=...`）。

**已知的行为副作用**（不是 bug，是迁移到统一 config 后的必然结果，明确记录不掩盖）：`configs/training/default.yaml` 是这个仓库里第一次被真正消费——旧 argparse 脚本从未读过它。相对旧 argparse 默认值，有两处生效值变化：
- `batch_size`：旧默认 `--batch-size 64` → config 里 `batch_size: 256`。dvc override 命令只 override `training.epochs=50`，不 override batch_size，故实际生效 256。不额外加 `training.batch_size=64` 去掩盖。
- `weight_decay`：旧脚本手搓 `Adam(svdd.parameters(), lr=lr)`，`weight_decay=0` → config 里 `optimizer.weight_decay: 1e-4`。本任务改为用 `hydra.utils.instantiate(cfg.training.optimizer, params=svdd.parameters())` 消费**整份** optimizer config（带 lr + weight_decay），不再手搓、不再只挑 `lr` 一个字段——避免 config 与代码脱节（CLAUDE.md 点名的"配置改了但没进消费路径"坑）。生效 weight_decay 从 0 变 1e-4。

**本轮明确不消费的 config 字段**（划清边界，不是遗漏）：`training.scheduler`（lr 衰减，会改变收敛行为，本轮保持 baseline 无 scheduler）、`training.early_stopping`（依赖 val split 消费，是下一轮 train_val 接入的工作）。这两块留在 config 里但训练脚本不读，Task 5 的 v1 `train_val.parquet` 同理是为下一轮预留。`num_workers` 消费（传给 DataLoader）。

**Files:**
- Modify: `scripts/train_baseline_v0.py`（整体重写）
- Modify: `tests/test_train_baseline_v0.py`
- Modify: `tests/test_e2e_smoke.py:40-57, 220-253`
- Modify: `dvc.yaml`（`train_v0` stage）

- [ ] **Step 1: 改 `tests/test_train_baseline_v0.py` 为 Hydra override 语法**

把文件里两处 subprocess 调用的参数从 argparse flag 改为 Hydra override（`key=value`，无 `--`）。`_build_contract` helper 不变（那是 `build_contract.py`，仍是 argparse，本任务不改）。完整替换后的文件：

```python
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


def _build_contract(contract_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _train(contract_dir: Path, out: Path, seed: int = 42, epochs: int = 2) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={out}",
            f"seed={seed}",
            f"training.epochs={epochs}",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_train_baseline_v0_writes_scores_contract(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    validate_scores_df(df)
    assert "case_id" in df.columns
    assert "anomaly_type" in df.columns
    # 行数应与 eval_all 一致（每个评估样本输出一个 score）
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)


def test_train_baseline_v0_is_reproducible(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out1 = tmp_path / "scores1.parquet"
    out2 = tmp_path / "scores2.parquet"
    for out in (out1, out2):
        _train(contract_dir, out)

    df1 = pd.read_parquet(out1).sort_values("sample_id").reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_series_equal(df1["score"], df2["score"])


def test_train_baseline_v0_scores_carry_endpoint_label_columns(tmp_path):
    """scores.parquet 必须显式带上 is_endpoint_anomaly / label_granularity，
    否则四层 eval 分层无法计算——这是历史上 is_target_endpoint 被静默丢弃的同一种坑。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert "is_endpoint_anomaly" in df.columns
    assert "label_granularity" in df.columns

    merged = df.merge(
        eval_all[["sample_id", "is_endpoint_anomaly", "label_granularity"]],
        on="sample_id",
        suffixes=("_out", "_src"),
    )
    assert (
        merged["is_endpoint_anomaly_out"].astype(bool)
        == merged["is_endpoint_anomaly_src"].astype(bool)
    ).all()
    assert (merged["label_granularity_out"] == merged["label_granularity_src"]).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_train_baseline_v0.py -v`
Expected: FAIL——脚本仍是 argparse，`contract_dir=...` 这种写法会被 `argparse` 当成不认识的位置参数，报 `error: unrecognized arguments`

- [ ] **Step 3: 重写 `scripts/train_baseline_v0.py`**

完整替换为：
```python
#!/usr/bin/env python
"""Baseline v0 训练脚本：融合机制 + Deep SVDD 单类异常检测（Hydra 驱动）。

流程（One-Class 约定）：
1. 用 train.parquet（仅 Normal）训练，center 在正常表征上初始化
2. 推理 eval_all.parquet，输出每样本异常分数（距超球心距离²，higher=更异常）
3. 写出符合 scores_v0 契约的 parquet（含 case_id/anomaly_type 等诊断列）

eval_all 的特征 NaN 用 train.parquet 均值填补（fit_on_parquet），避免 eval 自身统计量泄漏。

融合机制与模型均通过 hydra.utils.instantiate 构造（cfg.fusion / cfg.model），
切换实现只需 CLI override（如 fusion=gated），不需要改这个脚本。

注意接口约束：fusion=xxx 可任意切换（都实现 forward/output_dim 统一接口）；
但 model=xxx 只对实现了 init_center/svdd_loss/score 三个方法的 One-Class 检测器成立——
本训练循环调这三个方法，换成非 SVDD 系模型（如 VAE）需另写训练循环，不能仅靠 config 切换。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.contracts import validate_scores_df
from src.data.contract_dataloader import ContractDataset
from src.fusion.base import MODALITY_ORDER, FusionModule
from src.models.deep_svdd import DeepSVDD
from src.utils.seed import set_seed

LOG = logging.getLogger(__name__)


def _collate(batch: list[dict]) -> dict:
    """聚合 ContractDataset point 样本：modality tensor 堆叠，meta/label 保持 list。"""
    out: dict = {m: torch.stack([s[m] for s in batch]) for m in MODALITY_ORDER}
    out["sample_id"] = [s["meta"]["sample_id"] for s in batch]
    out["is_anomaly"] = [s["label"]["is_anomaly"] for s in batch]
    return out


def _modality_dims(schema: dict) -> dict[str, int]:
    return {name: len(grp["columns"]) for name, grp in schema["feature_groups"].items()}


def _make_dataset(parquet_path: Path, schema_path: Path, fit_on: Path) -> ContractDataset:
    return ContractDataset(
        parquet_path=parquet_path,
        schema_path=schema_path,
        mode="point",
        nan_strategy="mean",
        fit_on_parquet=fit_on,
    )


def _train(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    svdd.train()
    for epoch in range(epochs):
        total = 0.0
        n_batches = 0
        for batch in loader:
            x = fusion({m: batch[m] for m in MODALITY_ORDER})
            loss = svdd.svdd_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        LOG.info("epoch %d/%d loss=%.6f", epoch + 1, epochs, total / max(n_batches, 1))


@torch.no_grad()
def _infer(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
) -> dict[str, float]:
    svdd.eval()
    scores: dict[str, float] = {}
    for batch in loader:
        x = fusion({m: batch[m] for m in MODALITY_ORDER})
        s = svdd.score(x)
        for sid, val in zip(batch["sample_id"], s.tolist()):
            scores[sid] = val
    return scores


@hydra.main(config_path="../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    set_seed(cfg.seed)

    # hydra-core>=1.3 默认 hydra.job.chdir=False，不切换工作目录，
    # 因此 contract_dir/out 的相对路径解析规则与旧 argparse 版本完全一致
    # （相对于脚本被调用时的 cwd，不是 outputs/ 产物目录）。
    contract_dir = Path(cfg.contract_dir)
    train_pq = contract_dir / "train.parquet"
    eval_pq = contract_dir / "eval_all.parquet"
    schema_path = contract_dir / "schema.json"
    schema = json.loads(schema_path.read_text())

    train_ds = _make_dataset(train_pq, schema_path, fit_on=train_pq)
    eval_ds = _make_dataset(eval_pq, schema_path, fit_on=train_pq)

    modality_dims = _modality_dims(schema)
    fusion = hydra.utils.instantiate(cfg.fusion, modality_dims=modality_dims)
    svdd = hydra.utils.instantiate(cfg.model, input_dim=fusion.output_dim)

    # 用全部 Normal 训练样本初始化超球心（One-Class：center 只见正常表征）
    LOG.info("初始化 SVDD 超球心，加载 %d 训练样本...", len(train_ds))
    if len(train_ds) > 10_000:
        LOG.warning(
            "训练集 %d 行，init_center 一次性加载全部数据到内存；如遇 OOM 请考虑增量初始化",
            len(train_ds),
        )
    init_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=_collate)
    init_batch = next(iter(init_loader))
    svdd.init_center(fusion({m: init_batch[m] for m in MODALITY_ORDER}))

    # instantiate 整份 optimizer config（lr + weight_decay 都来自 cfg.training.optimizer），
    # 不手搓、不只挑 lr——避免 weight_decay 等字段静默丢失。svdd.parameters() 作为
    # 运行时参数在 instantiate 时以 params= 补上（fusion 无可训练参数，只传 svdd）。
    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=svdd.parameters())

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    _train(fusion, svdd, train_loader, epochs=cfg.training.epochs, optimizer=optimizer)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=cfg.training.num_workers,
    )
    score_map = _infer(fusion, svdd, eval_loader)

    eval_df = pd.read_parquet(eval_pq)
    out_df = pd.DataFrame(
        {
            "sample_id": eval_df["sample_id"].astype(str),
            "score": eval_df["sample_id"].map(score_map).astype(float),
            "y_true": eval_df["is_anomaly"].astype(int),
            "case_id": eval_df["case_id"],
            "endpoint_key": eval_df["endpoint_key"],
            "phase": eval_df["phase"],
            "anomaly_type": eval_df["anomaly_type"],
            "anomaly_level": eval_df["anomaly_level"],
            "is_endpoint_anomaly": eval_df["is_endpoint_anomaly"].astype(int),
            "label_granularity": eval_df["label_granularity"],
        }
    )

    validate_scores_df(out_df)

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    LOG.info("写出 scores：%d 行 → %s", len(out_df), out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_train_baseline_v0.py -v`
Expected: PASS（3 个测试全绿）

- [ ] **Step 5: 改 `tests/test_e2e_smoke.py` 两处调用点**

第一处（`pipeline_out` fixture，约第 40-57 行）：
```python
    # Stage 2: train_baseline_v0
    scores_path = baseline_dir / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={scores_path}",
            "seed=42",
            "training.epochs=2",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
```

第二处（约第 220-253 行的 reproducibility 测试内 `run_pipeline` 函数）：
```python
        scores_path = out_dir / "scores.parquet"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_contract.py",
                "--config",
                str(REPO_ROOT / "configs/contract/v0.yaml"),
                "--dataset",
                str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
                "--out-dir",
                str(contract_dir),
                "--seed",
                "42",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/train_baseline_v0.py",
                f"contract_dir={contract_dir}",
                f"out={scores_path}",
                "seed=42",
                "training.epochs=2",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        return pd.read_parquet(scores_path)["score"].values
```

（`build_contract.py` 调用不变，仍是 argparse；只改 `train_baseline_v0.py` 那一段。）

- [ ] **Step 6: 跑 e2e 测试确认通过**

Run: `pytest tests/test_e2e_smoke.py -v`
Expected: PASS

- [ ] **Step 7: 改 `dvc.yaml` 的 `train_v0` stage**

```yaml
  train_v0:
    cmd: python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v0
      out=artifacts/baseline_v0/scores.parquet seed=42 training.epochs=50
      fusion=concat model=deep_svdd
    deps:
      - scripts/train_baseline_v0.py
      - configs/base.yaml
      - configs/fusion/concat.yaml
      - configs/model/deep_svdd.yaml
      - configs/training/default.yaml
      - src/models/deep_svdd.py
      - src/fusion/early_concat.py
      - src/fusion/base.py
      - src/data/contract_dataloader.py
      - artifacts/contract_v0/train.parquet
      - artifacts/contract_v0/eval_all.parquet
      - artifacts/contract_v0/schema.json
    outs:
      - artifacts/baseline_v0/scores.parquet
```

新增的 `configs/base.yaml`、`configs/fusion/concat.yaml`、`configs/model/deep_svdd.yaml`、`configs/training/default.yaml`、`src/fusion/base.py` 加入 `deps`——这些文件现在都会实际影响训练行为（Hydra config 树 + `MODALITY_ORDER` 来源），漏掉会导致改了 config 但 DVC 不重跑，属于 CLAUDE.md 提到的"配置改了但没进 deps"隐蔽坑。

- [ ] **Step 8: 跑现有 DVC 结构测试确认不破坏**

Run: `pytest tests/test_dvc_pipeline_v0.py -v`
Expected: PASS（该测试只检查 `build_contract` stage 的 `--dataset` 用法，不检查 `train_v0` 的具体 cmd 语法，故不受影响）

- [ ] **Step 9: 更新 `CLAUDE.md` 里过时的命令示例与描述**

spec 组件 1 明确要求修复 CLAUDE.md 与代码脱节。Task 4 落地后，CLAUDE.md 有两处会变过时，必须同步（否则下次照文档敲命令会直接报错）：
- `CLAUDE.md:97` 的 `python scripts/train_baseline_v0.py --contract-dir artifacts/contract_v0 --out ... --seed 42 --epochs 50` → 改为 Hydra override 语法：
  ```
  python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v0 out=artifacts/baseline_v0/scores.parquet seed=42 training.epochs=50 fusion=concat model=deep_svdd
  ```
- `CLAUDE.md:109` 的"模型通过 `hydra.utils.instantiate(cfg.model)` 实例化"——这句从"声称但未实现"变成"已实现"，可保留，但建议补一句 fusion 同理走 `cfg.fusion` instantiate、且 `contract_dir`/`out` 现在是 Hydra 字段而非 argparse flag。
（`build_contract.py`/`eval_baseline_v0.py` 两行命令不变，仍是 argparse，本轮不动。）

- [ ] **Step 10: 跑完整测试套件确认无回归**

Run: `pytest tests/ -v`
Expected: PASS（含 Task 1-4 新增/修改的所有测试；`test_dvc_pipeline_v0.py`、`test_schema_json_contract.py` 等未改动的测试应保持绿）

- [ ] **Step 11: Commit**

```bash
git add scripts/train_baseline_v0.py tests/test_train_baseline_v0.py tests/test_e2e_smoke.py dvc.yaml CLAUDE.md
git commit -m "[Refactor]: train_baseline_v0.py 改为 Hydra entrypoint，融合/模型走 instantiate"
```

---

## Task 5: Contract v1 — 三路时序切分（train_fit / train_val / eval_normal_holdout）

**背景**：spec 组件 2。现状 `build_contract.py` 产出 `train.parquet`（全部 Normal 行）+ `eval_all.parquet`（`= full`，全部行），后者逐行包含前者——训练时见过的 Normal 行原样进了评估集，违反 One-Class 评估"测试集正常样本必须未见过"的基本要求，导致 baseline 指标虚高。本任务新增 v1 产出路径，把 Normal 行切成 `train_fit`/`train_val`/`eval_normal_holdout` 三个互斥子集，`eval_all` 的 Normal 部分只保留 `eval_normal_holdout`。v0 pipeline 原样保留并存。

**切分维度决策：按时间窗切分，不按 service（与原 spec 和 entry 010 均偏离，已与用户确认）**。spec 组件 2 原写按 `service_name` 做 `GroupShuffleSplit`；entry 010 的坑 #2 进一步主张 group-aware / leave-service-out 切分，理由是"同一 service 下所有 endpoint 的 metric/log 特征被 left join 广播复制成完全相同，随机行切分会让模型学到 service identity 而非异常信号，构成泄漏式乐观偏差"。**本任务改为按 case 内时间窗时序切分，需要正面回应 010 那个担忧**：

- 为什么不用 leave-service-out：endpoint→service ~1:1（8 对 8），leave-service-out 等于让 `eval_normal_holdout` 里的 endpoint 训练时完全没见过，对 per-endpoint One-Class 检测变成考"对未见 endpoint 的泛化"，违背本研究"对已监控 endpoint 检测异常"的设想，holdout 正常样本仅因"没见过"就被打高分，指标失真。用户已确认放弃 leave-service-out。
- 为什么时序切分仍能挡住 010 的泄漏担忧：010 担忧的本质是"train 和 eval 出现**逐行相同**的广播特征行"。时序切分下 train 取早期窗、eval 取晚期窗，service 级广播特征随时间变化（不是常量），eval 的 holdout 行是**训练时未见过的时间窗**，FP 估计诚实。至于"模型可能学 service identity"——在 One-Class（只在 Normal 上 fit 超球心、无 service 分类目标）设定下，这不构成监督分类式泄漏；而"同 service 下 endpoint 无法区分"正是门控融合方法本身要解决的问题，属于建模目标，不该靠切分回避。
- 切分方式：每个 Normal case 独立按 `timestamp_window_ms` 排序，前 60% 窗 → `train_fit`，中 20% → `train_val`，后 20% → `eval_normal_holdout`。① 每个 endpoint 在三份都出现（无泛化偏移）；② 训练窗与评估窗时间分离；③ 仅切分边界处相邻两窗时序相关，泄漏远小于随机行切分。真实数据两 Normal case 各 108/156 窗、各含全部 8 endpoint（已实测原型验证：63.4/18.7/17.9%，8 endpoint 三份全覆盖，每 case 内 fit<val<hold 时序分离成立）。

**⚠️ 这个偏离必须写进 history entry**（最终验证段已列入）：entry 010 的遗留 TODO 明写"补 group-aware / leave-service-out 切分"，本轮改用时序切分是有意识的方法论决策，不是漏做——不记录会让未来查 010 的人误判 TODO 未完成，或误以为泄漏担忧未处理。

**关键设计决策（切分逻辑抽成纯函数）**：切分逻辑抽到独立纯函数 `src/contracts/split_v1.py::split_normal_rows_temporal`，不做任何 I/O，输入输出都是 DataFrame——能脱离 `build_contract.py` 的 subprocess 直接单元测试时序切分的互斥性、endpoint 覆盖、比例。**窗数不足退化**：某 case 时间窗数 < 3 时切不出三份（如 fixture 只 3 窗、或极短 case），按可用窗数尽力分配，不足的份为空——空份保留 schema，下游 concat/写 parquet 不报 KeyError。这不是测试专属分支：真实数据里也可能有极短 case。

**范围控制决策（v1 也写一份 `train.parquet`）**：v1 分支额外把 `train_fit` 的内容原样再写一份 `train.parquet`。spec 明确排除"改训练脚本消费 `train_val` 早停 / 跑 L0-L3"——那是下一轮。写这份冗余 `train.parquet` 让 `train_v1` dvc stage 今天就能复用 Task 4 改完的同一个训练脚本跑通，代价仅是 `artifacts/contract_v1/` 下多一个文件。`train_val` 的早停消费留给下一轮。

**Files:**
- Create: `src/contracts/split_v1.py`（时序切分纯函数）
- Create: `configs/contract/v1.yaml`
- Modify: `scripts/build_contract.py`（按 `cfg.contract_version` 分支）
- Test: `tests/test_contract_v1_split.py`
- Modify: `dvc.yaml`（新增 `build_contract_v1`/`train_v1`/`eval_v1`）

- [ ] **Step 1: 写失败的测试**

`tests/test_contract_v1_split.py`：
```python
import pandas as pd

from src.contracts.split_v1 import split_normal_rows_temporal

# 合成两个 Normal case，模拟真实结构：每个 case 多个时间窗，每窗含全部 8 endpoint。
# 真实数据两 case 各 108/156 窗（已实测），这里各用 100 窗，够切且比例干净。
_ENDPOINTS = [f"ep{i}" for i in range(8)]
_CASES = {"NormalA": 100, "NormalB": 100}


def _synth_normal_df() -> pd.DataFrame:
    rows = []
    for case_id, n_win in _CASES.items():
        for w in range(n_win):
            ts = 1_000 + w * 15_000  # 15s 一窗，单调递增
            for ep in _ENDPOINTS:
                rows.append({
                    "sample_id": f"{case_id}__{ep}__{ts}",
                    "case_id": case_id,
                    "endpoint_key": ep,
                    "timestamp_window_ms": ts,
                    "feat": float(w),
                })
    return pd.DataFrame(rows)


def test_split_three_way_sample_id_disjoint():
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    ids = {k: set(v["sample_id"]) for k, v in parts.items()}
    assert ids["train_fit"] & ids["train_val"] == set()
    assert ids["train_fit"] & ids["eval_normal_holdout"] == set()
    assert ids["train_val"] & ids["eval_normal_holdout"] == set()
    total = sum(len(v) for v in parts.values())
    assert total == len(_synth_normal_df())  # 无行丢失


def test_all_endpoints_present_in_every_split():
    """时序切分的核心目的：每个 endpoint 在三份里都出现，不因切分整体消失
    （这正是相对 service 切分的改进——避免 eval 出现训练时没见过的 endpoint）。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    for name, df in parts.items():
        assert set(df["endpoint_key"].unique()) == set(_ENDPOINTS), f"{name} 缺 endpoint"


def test_temporal_no_overlap_per_case():
    """每个 case 内：train_fit 的窗全部早于 train_val，train_val 全部早于 holdout。
    时序分离是"评估样本训练时未见过"的保证来源。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    for case_id in _CASES:
        fit_ts = parts["train_fit"].query("case_id == @case_id")["timestamp_window_ms"]
        val_ts = parts["train_val"].query("case_id == @case_id")["timestamp_window_ms"]
        hold_ts = parts["eval_normal_holdout"].query("case_id == @case_id")["timestamp_window_ms"]
        assert fit_ts.max() < val_ts.min()
        assert val_ts.max() < hold_ts.min()


def test_split_ratio_within_tolerance():
    """比例目标 60/20/20。时序切分按窗数分配，比例比 service 切分更均匀，容差 ±5 个百分点。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    total = sum(len(v) for v in parts.values())
    frac = {k: len(v) / total for k, v in parts.items()}
    assert abs(frac["train_fit"] - 0.60) <= 0.05
    assert abs(frac["train_val"] - 0.20) <= 0.05
    assert abs(frac["eval_normal_holdout"] - 0.20) <= 0.05


def test_too_few_windows_degrades_gracefully():
    """case 只有 1 个时间窗：切不出三份，全归 train_fit，另两个空，不抛异常。"""
    df = pd.DataFrame({
        "sample_id": [f"X__{ep}__1000" for ep in _ENDPOINTS],
        "case_id": "X", "endpoint_key": _ENDPOINTS,
        "timestamp_window_ms": 1000, "feat": 1.0,
    })
    parts = split_normal_rows_temporal(df, seed=42)
    assert len(parts["train_fit"]) == len(_ENDPOINTS)
    assert len(parts["train_val"]) == 0
    assert len(parts["eval_normal_holdout"]) == 0
    assert list(parts["train_val"].columns) == list(df.columns)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_contract_v1_split.py -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'src.contracts.split_v1'`

- [ ] **Step 3: 写 `src/contracts/split_v1.py`**

```python
"""Contract v1 三路时序切分：每个 Normal case 内按时间窗切成
train_fit / train_val / eval_normal_holdout 三个互斥子集。

修复 v0 的 train ⊆ eval_all 重叠问题——见 spec 组件 2。
按时间窗（而非 service）切分：每个 endpoint 在三份里都出现（无泛化偏移），
训练窗与评估窗时序分离（评估样本训练时未见过）。seed 参数当前不影响时序切分
（纯按时间排序，确定性），保留是为与 build_contract 的 seed 传参接口一致 + 未来扩展。
"""

from __future__ import annotations

import logging

import pandas as pd

LOG = logging.getLogger(__name__)

_PARTS = ("train_fit", "train_val", "eval_normal_holdout")


def split_normal_rows_temporal(
    df: pd.DataFrame,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
    seed: int = 42,
    fit_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """返回 {"train_fit", "train_val", "eval_normal_holdout"} 三个互斥 DataFrame。

    每个 case 独立按 time_col 排序，前 fit_frac 的时间窗 → train_fit，
    接着 val_frac → train_val，其余 → eval_normal_holdout。切分单位是"时间窗"
    （同一窗的所有 endpoint 行不拆散），保证 endpoint 在各份都出现、训练窗早于评估窗。

    窗数不足退化：某 case 时间窗数 < 3 时按可用窗数尽力分配（如 1 窗全进 train_fit，
    2 窗 train_fit+holdout），不足的份对该 case 贡献 0 行。空份仍保留 schema。
    """
    empty = df.iloc[0:0]
    buckets: dict[str, list[pd.DataFrame]] = {p: [] for p in _PARTS}

    for case_id, g in df.groupby(case_col, sort=False):
        windows = sorted(g[time_col].unique())
        n = len(windows)
        n_fit = int(n * fit_frac)
        n_val = int(n * val_frac)
        # 边界处理：至少各 1 窗给 fit；holdout 拿走剩余。窗数极少时 val/holdout 可能为空。
        fit_w = set(windows[:n_fit])
        val_w = set(windows[n_fit : n_fit + n_val])
        hold_w = set(windows[n_fit + n_val :])
        if not fit_w:  # n < 2：全部时间窗归 train_fit，另两份空
            fit_w, val_w, hold_w = set(windows), set(), set()
            LOG.warning("case=%s 只有 %d 个时间窗，全部归入 train_fit", case_id, n)
        buckets["train_fit"].append(g[g[time_col].isin(fit_w)])
        buckets["train_val"].append(g[g[time_col].isin(val_w)])
        buckets["eval_normal_holdout"].append(g[g[time_col].isin(hold_w)])

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [f for f in frames if len(f)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else empty.copy()

    return {p: _concat(buckets[p]) for p in _PARTS}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_contract_v1_split.py -v`
Expected: PASS（5 个测试全绿）

- [ ] **Step 5: 写 `configs/contract/v1.yaml`**

内容与 `v0.yaml` 完全一致，仅 `contract_version` 改为 `"v1"`。字段口径（modalities/features/normalization）不变——本轮契约变化只在"行集合划分方式"，不动特征定义。

```yaml
contract_version: "v1"
window_size_s: 15

modalities:
  # ... 与 configs/contract/v0.yaml 的 modalities 段逐字相同，复制过来即可 ...
```

（实现时直接 `cp configs/contract/v0.yaml configs/contract/v1.yaml` 再改第一行 `contract_version`，避免手抄 modalities 段出错。）

- [ ] **Step 6: 改 `scripts/build_contract.py` 按 `contract_version` 分支写出**

在 `main()` 末尾（当前 `scripts/build_contract.py:390-397`，`validate_contract_df(full, ...)` 之后）把写 parquet 的逻辑改为按版本分支。v0 路径逐行不变，仅新增 v1 分支：

```python
    validate_contract_df(full, args.config)

    if cfg.contract_version == "v1":
        _write_v1(out, full, normal_mask, args.seed)
    else:
        # v0：train=全部 Normal，eval_all=全部行（保持向后兼容，train ⊆ eval_all）
        full[normal_mask].reset_index(drop=True).to_parquet(out / "train.parquet", index=False)
        full.to_parquet(out / "eval_all.parquet", index=False)

    _write_schema(out, cfg)
    LOG.info("完成！版本=%s，总行数=%d", cfg.contract_version, len(full))
```

新增 helper（放在 `_write_schema` 附近），import 处加 `from src.contracts.split_v1 import split_normal_rows_temporal`：

```python
def _write_v1(out: Path, full: pd.DataFrame, normal_mask: pd.Series, seed: int) -> None:
    """v1：Normal 行按时间窗三路切分，eval_all 的 Normal 部分只取 holdout。"""
    normal_df = full[normal_mask].reset_index(drop=True)
    anomaly_df = full[~normal_mask].reset_index(drop=True)
    parts = split_normal_rows_temporal(normal_df, seed=seed)

    parts["train_fit"].to_parquet(out / "train_fit.parquet", index=False)
    parts["train_val"].to_parquet(out / "train_val.parquet", index=False)
    parts["eval_normal_holdout"].to_parquet(out / "eval_normal_holdout.parquet", index=False)

    # 冗余 train.parquet = train_fit，让 train_v1 stage 复用 Task 4 的训练脚本原样跑通
    # （训练脚本消费 train_val 早停是下一轮工作，spec 已排除）。
    parts["train_fit"].to_parquet(out / "train.parquet", index=False)

    # eval_all = 故障 case 全部行 + 仅 holdout 的 Normal 行（修复 train ⊆ eval_all 重叠）
    eval_all = pd.concat([anomaly_df, parts["eval_normal_holdout"]], ignore_index=True)
    eval_all.to_parquet(out / "eval_all.parquet", index=False)
    LOG.info(
        "v1 切分：train_fit=%d train_val=%d holdout=%d eval_all=%d",
        len(parts["train_fit"]), len(parts["train_val"]),
        len(parts["eval_normal_holdout"]), len(eval_all),
    )
```

- [ ] **Step 7: 写 v1 pipeline 的 e2e smoke 测试**

在 `tests/test_contract_v1_split.py` 末尾追加一个走完整 subprocess 的 smoke，验证时序切分在真实 `build_contract.py --config configs/contract/v1.yaml` 调用下不崩、产物齐全、三份 parquet 互斥。注意 mini fixture 的 Normal case 只有 3 个时间窗，按 60/20/20 会得 `train_fit`=1 窗 / `train_val`=0（`int(3*0.2)=0`）/ `holdout`=2 窗——`train_val` 为空是窗数太少的正常退化，`train_val.parquet` 仍必须写出、互斥断言仍成立：

```python
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_build_contract_v1_smoke(tmp_path):
    out_dir = tmp_path / "contract_v1"
    subprocess.run(
        [
            sys.executable, "scripts/build_contract.py",
            "--config", str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset", str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir", str(out_dir), "--seed", "42",
        ],
        check=True, cwd=str(REPO_ROOT),
    )
    for name in ("train_fit", "train_val", "eval_normal_holdout", "train", "eval_all"):
        assert (out_dir / f"{name}.parquet").exists(), f"{name}.parquet 缺失"

    import json
    schema = json.loads((out_dir / "schema.json").read_text())
    assert schema["contract_version"] == "v1"

    fit = pd.read_parquet(out_dir / "train_fit.parquet")
    val = pd.read_parquet(out_dir / "train_val.parquet")
    holdout = pd.read_parquet(out_dir / "eval_normal_holdout.parquet")
    ids = [set(d["sample_id"]) for d in (fit, val, holdout)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
```

Run: `pytest tests/test_contract_v1_split.py -v`
Expected: PASS（5 个测试全绿）

- [ ] **Step 8: 在 `dvc.yaml` 新增 v1 三个 stage**

`build_contract`/`train_v0`/`eval_v0` 原样不动，末尾追加：

```yaml
  build_contract_v1:
    cmd: python scripts/build_contract.py --config configs/contract/v1.yaml --dataset
      configs/data/merged_v2.yaml --out-dir artifacts/contract_v1 --seed 42
    deps:
      - scripts/build_contract.py
      - src/preprocessors
      - src/contracts/contract_v0.py
      - src/contracts/split_v1.py
      - src/data/normalization.py
      - src/data/dataset_config.py
      - configs/contract/v1.yaml
      - configs/contract/endpoint_to_service.yaml
      - configs/data/merged_v2.yaml
    outs:
      - artifacts/contract_v1/train_fit.parquet
      - artifacts/contract_v1/train_val.parquet
      - artifacts/contract_v1/eval_normal_holdout.parquet
      - artifacts/contract_v1/train.parquet
      - artifacts/contract_v1/eval_all.parquet
      - artifacts/contract_v1/normalization_stats.json
      - artifacts/contract_v1/schema.json

  train_v1:
    cmd: python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1
      out=artifacts/baseline_v1/scores.parquet seed=42 training.epochs=50
      fusion=concat model=deep_svdd
    deps:
      - scripts/train_baseline_v0.py
      - configs/base.yaml
      - configs/fusion/concat.yaml
      - configs/model/deep_svdd.yaml
      - configs/training/default.yaml
      - src/models/deep_svdd.py
      - src/fusion/early_concat.py
      - src/fusion/base.py
      - src/data/contract_dataloader.py
      - artifacts/contract_v1/train.parquet
      - artifacts/contract_v1/eval_all.parquet
      - artifacts/contract_v1/schema.json
    outs:
      - artifacts/baseline_v1/scores.parquet

  eval_v1:
    cmd: python scripts/eval_baseline_v0.py --scores artifacts/baseline_v1/scores.parquet
      --out artifacts/baseline_v1/metrics.json
    deps:
      - scripts/eval_baseline_v0.py
      - src/contracts
      - artifacts/baseline_v1/scores.parquet
    metrics:
      - artifacts/baseline_v1/metrics.json:
          cache: false
```

注：`train_v1`/`eval_v1` 复用 Task 4 改完的 `train_baseline_v0.py` 和现有 `eval_baseline_v0.py`（不新增脚本，spec 也没要求）。⚠️ 用真实 `merged_v2.yaml` 数据跑 `build_contract_v1` 时，Normal 来自 `normal_v2`（2 个 case，各 108/156 时间窗），走正常时序三路切分；窗数不足退化只在 fixture 那种 3 窗场景发生。

- [ ] **Step 9: 跑相关测试确认无回归**

Run: `pytest tests/test_contract_v1_split.py tests/test_build_contract_smoke.py tests/test_dvc_pipeline_v0.py -v`
Expected: PASS（v0 smoke 不受影响，v1 新测试全绿；`test_dvc_pipeline_v0.py` 只校验 `build_contract` stage 的 `--dataset`，不检查新 stage）

- [ ] **Step 10: Commit**

```bash
git add src/contracts/split_v1.py configs/contract/v1.yaml scripts/build_contract.py tests/test_contract_v1_split.py dvc.yaml
git commit -m "[Feature]: Contract v1 时序三路切分，修复 train/eval 重叠"
```

---

## Task 6: 参数量对齐工具 `solve_hidden_dim_for_param_budget`

**背景**：spec 组件 3。L0→L1（裸拼接零参数 → 独立 encoder + 拼接）的性能提升，混淆了"是否有门控设计"和"是否有非线性容量"两个变量。要把门控本身的贡献隔离出来，需要在参数量对齐的前提下比较——本任务提供这把"尺子"：给定一个 `build_fn(hidden_dim) -> nn.Module`，二分查找使参数量最接近目标的 `hidden_dim`。**本轮只提供工具，不接入任何训练脚本，不用它构造 L0-L3 中任何一个 baseline**（spec 明确排除）。

**Files:**
- Create: `src/utils/param_budget.py`
- Test: `tests/test_param_budget.py`

- [ ] **Step 1: 写失败的测试**

`tests/test_param_budget.py`：
```python
import pytest
import torch.nn as nn

from src.utils.param_budget import solve_hidden_dim_for_param_budget


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _two_layer(h: int) -> nn.Module:
    # 输入 20 → hidden h → 输出 10，参数量随 h 单调递增
    return nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 10))


def test_converges_close_to_target():
    target = _count(_two_layer(128))
    h = solve_hidden_dim_for_param_budget(_two_layer, target, lo=1, hi=1024)
    # 二分找到的 h 构造出的参数量应贴近 target（±该结构单步 hidden 的参数量增量）
    assert abs(_count(_two_layer(h)) - target) <= _count(_two_layer(2)) - _count(_two_layer(1)) + 1


def test_exact_hit_returns_that_dim():
    target = _count(_two_layer(64))
    h = solve_hidden_dim_for_param_budget(_two_layer, target, lo=1, hi=1024)
    assert _count(_two_layer(h)) == target


def test_target_below_range_raises():
    below = _count(_two_layer(1)) - 1
    with pytest.raises(ValueError):
        solve_hidden_dim_for_param_budget(_two_layer, below, lo=1, hi=1024)


def test_target_above_range_raises():
    above = _count(_two_layer(1024)) + 1
    with pytest.raises(ValueError):
        solve_hidden_dim_for_param_budget(_two_layer, above, lo=1, hi=1024)


def test_lo_equals_hi_returns_that_point():
    h = solve_hidden_dim_for_param_budget(_two_layer, _count(_two_layer(50)), lo=50, hi=50)
    assert h == 50
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_param_budget.py -v`
Expected: FAIL——`ModuleNotFoundError: No module named 'src.utils.param_budget'`

- [ ] **Step 3: 写 `src/utils/param_budget.py`**

```python
"""参数量对齐工具：二分查找使模型参数量最接近目标的 hidden_dim。

用于消融实验把不同容量的 baseline（如 L0/L1）参数量对齐到 L2/L3，
排除"参数量差异"这个混淆变量。纯函数，不依赖任何具体模型类。
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def solve_hidden_dim_for_param_budget(
    build_fn: Callable[[int], nn.Module],
    target_params: int,
    lo: int = 1,
    hi: int = 1024,
) -> int:
    """二分查找整数 hidden_dim ∈ [lo, hi]，使 build_fn(h) 的参数量最接近 target_params。

    前提：参数量随 hidden_dim 单调不减（纯 Linear 堆叠满足）。若 target_params 落在
    [count(build_fn(lo)), count(build_fn(hi))] 之外，说明 lo/hi 边界给错了，显式抛
    ValueError——而非静默返回边界值，那样会让调用方拿到一个"最接近但其实差很远"的错误结果。
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) 必须 <= hi ({hi})")

    lo_params = _count_params(build_fn(lo))
    hi_params = _count_params(build_fn(hi))
    if not (lo_params <= target_params <= hi_params):
        raise ValueError(
            f"target_params={target_params} 超出可达范围 [{lo_params}, {hi_params}]"
            f"（hidden_dim ∈ [{lo}, {hi}]），调整 lo/hi 边界"
        )

    # 标准整数二分：收敛到使 count(build_fn(h)) 不小于 target 的最小 h，
    # 再在 h 与 h-1 之间取参数量更接近 target 的那个。
    best = lo
    best_diff = abs(lo_params - target_params)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = _count_params(build_fn(mid))
        diff = abs(p - target_params)
        if diff < best_diff:
            best, best_diff = mid, diff
        if p == target_params:
            return mid
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return best
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_param_budget.py -v`
Expected: PASS（5 个测试全绿）

- [ ] **Step 5: Commit**

```bash
git add src/utils/param_budget.py tests/test_param_budget.py
git commit -m "[Feature]: 新增参数量对齐工具 solve_hidden_dim_for_param_budget"
```

---

## 最终验证

- [ ] **跑完整测试套件**

Run: `pytest tests/ -v`
Expected: PASS（Task 1-6 全部新增/修改测试 + 所有既有测试无回归）

- [ ] **v0 pipeline 冒烟（确认没被 v1 改动破坏）**

Run: `dvc repro build_contract train_v0 eval_v0`
Expected: 三个 v0 stage 正常跑通，`artifacts/baseline_v0/metrics.json` 产出

- [ ] **v1 pipeline 冒烟（确认新链路端到端可跑）**

Run: `dvc repro build_contract_v1 train_v1 eval_v1`
Expected: 三个 v1 stage 跑通，`artifacts/contract_v1/` 下五个 parquet + `artifacts/baseline_v1/metrics.json` 产出；日志里 v1 切分行数比例落在 ~60/20/20 附近（真实 108/156 窗时序切分，非退化）

- [ ] **写 history entry**

按 CLAUDE.md 规则，PR merge 前写一篇新 entry 到 `history/entries/`（下一个编号，当前最新是 010），记录本轮三块基础设施的决策与坑（Hydra 迁移带来的 batch_size 64→256 与 weight_decay 0→1e-4 副作用、v1 切分从 spec 的 service 分组改为时间窗时序切分的原因、v1 与 v0 指标不可比、窗数不足退化路径），并更新 `history/index.md` 的列表与影响域索引。
