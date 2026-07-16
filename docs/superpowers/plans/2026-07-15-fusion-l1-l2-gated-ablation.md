# L1/L2 融合模块（独立编码器 + 门控条件融合）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现两个新的 `FusionModule` 子类——L1 独立编码器无门控拼接、L2 门控条件融合——接入现有 Hydra 可插拔融合机制，跑通 Contract v1 训练/评估，产出 AUROC/AUPRC 与已有 L0 baseline（AUROC=0.617/AUPRC=0.327）对比。

**Architecture:** `endpoint_red`（10维）和 `concat(service_metric, service_log)`（8维）各过一个独立的单层 `Linear(bias=False)+ReLU` 编码器得到 `e_ep`/`e_svc`（同维 `branch_dim`，默认16）。L1 直接 `concat(e_ep, e_svc)`；L2 用 `g=sigmoid(Linear([e_ep;e_svc]))` 门控 `e_svc` 后与 `e_ep` 相加。两者输出的 `z` 都照常送入现有 `DeepSVDD` 三层 encoder，训练脚本 `scripts/train_baseline_v0.py` 不改动，只通过 `fusion=independent_concat`/`fusion=gated` CLI override 切换。

**Tech Stack:** PyTorch (`nn.Module`), Hydra (`hydra.utils.instantiate`), pytest, conda env `interface`

---

## 背景说明（供实施者理解代码库约定）

- 所有 fusion 模块继承 `src/fusion/base.py` 的 `FusionModule` 抽象类：必须实现 `forward(modality_dict: dict[str, torch.Tensor]) -> torch.Tensor` 和 `output_dim` property。
- `MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")` 是模块级常量，来自 `src/fusion/base.py`。所有 fusion 子类的 `__init__` 都接收 `modality_dims: dict[str, int]`，必须校验 `set(modality_dims) == set(MODALITY_ORDER)`，否则 `raise ValueError`（缺失或多余的 key 都要报错）——参照 `src/fusion/early_concat.py` 现有实现。
- Hydra 通过 `hydra.utils.instantiate(cfg.fusion, modality_dims=modality_dims)` 构造 fusion 实例（`scripts/train_baseline_v0.py:118`），`modality_dims` 是运行时传入的关键字参数，不在 config 文件里。config 文件里只放 `_target_` 和该类自己的其他超参数（如 `branch_dim`）。
- 测试环境：本机 conda base 环境跑 pytest 即可（`pyproject.toml` 无特殊 pytest 依赖锁定在 `interface` 环境），已有的 `tests/test_early_concat_fusion.py` 直接 `pytest tests/test_early_concat_fusion.py -v` 跑通，新测试遵照同样方式跑。
- 已有 `artifacts/contract_v1/` 由 entry 011 的 `dvc repro build_contract_v1` 产出，本轮假定它已经存在（若不存在，先跑 `dvc repro build_contract_v1`）。

---

## Task 1: IndependentConcatFusion（L1）

**Files:**
- Create: `src/fusion/independent_concat.py`
- Test: `tests/test_independent_concat_fusion.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_independent_concat_fusion.py`：

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_independent_concat_fusion.py -v`
Expected: FAIL，报错 `ModuleNotFoundError: No module named 'src.fusion.independent_concat'`

- [ ] **Step 3: 写最小实现**

创建 `src/fusion/independent_concat.py`：

```python
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import MODALITY_ORDER, FusionModule


class IndependentConcatFusion(FusionModule):
    """L1：endpoint/service 两路各过独立单层编码器后拼接，无门控交互。

    用于把 L0(裸拼接零参数)→L2(门控) 的性能提升，与"是否引入任意非线性容量"
    这一混淆变量分离——L1 提供有非线性容量但无门控设计的对照组。
    """

    def __init__(self, modality_dims: dict[str, int], branch_dim: int = 16):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._branch_dim = branch_dim
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        # bias=False：与 DeepSVDD.encoder 的约定一致，避免 bias 吸收偏移。
        self.ep_encoder = nn.Sequential(
            nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU()
        )
        self.svc_encoder = nn.Sequential(
            nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU()
        )

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat(
            [modality_dict["service_metric"], modality_dict["service_log"]], dim=-1
        )
        svc = self.svc_encoder(svc_in)
        return torch.cat([ep, svc], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self._branch_dim
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_independent_concat_fusion.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/fusion/independent_concat.py tests/test_independent_concat_fusion.py
git commit -m "[Feature]: 新增 L1 IndependentConcatFusion（独立编码器无门控）"
```

---

## Task 2: GatedFusion（L2）

**Files:**
- Create: `src/fusion/gated.py`
- Test: `tests/test_gated_fusion.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/test_gated_fusion.py`：

```python
import pytest
import torch

from src.fusion.gated import GatedFusion


def test_output_dim_equals_branch_dim():
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    assert fusion.output_dim == 16


def test_forward_shape():
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        branch_dim=16,
    )
    batch = {
        "endpoint_red": torch.randn(4, 10),
        "service_metric": torch.randn(4, 5),
        "service_log": torch.randn(4, 3),
    }
    out = fusion(batch)
    assert out.shape == (4, 16)


def test_missing_modality_key_raises():
    with pytest.raises(ValueError):
        GatedFusion(
            modality_dims={"endpoint_red": 10, "service_metric": 5}, branch_dim=16
        )


def test_extra_modality_key_raises():
    with pytest.raises(ValueError):
        GatedFusion(
            modality_dims={
                "endpoint_red": 10,
                "service_metric": 5,
                "service_log": 3,
                "extra": 2,
            },
            branch_dim=16,
        )


def test_gate_formula_matches_manual_computation():
    """构造已知权重，手算 g 和 z，验证 forward 输出与公式严格一致。

    公式：e_ep = ReLU(ep_encoder(endpoint_red))
          e_svc = ReLU(svc_encoder(concat(metric, log)))
          g = sigmoid(gate(concat(e_ep, e_svc)))
          z = e_ep + g * value(e_svc)
    """
    branch_dim = 2
    fusion = GatedFusion(
        modality_dims={"endpoint_red": 3, "service_metric": 2, "service_log": 1},
        branch_dim=branch_dim,
    )
    fusion.eval()

    # 固定所有可学习权重为已知值，便于手算期望输出。
    with torch.no_grad():
        fusion.ep_encoder[0].weight.fill_(0.1)
        fusion.svc_encoder[0].weight.fill_(0.2)
        fusion.gate.weight.fill_(0.05)
        fusion.gate.bias.fill_(0.0)
        fusion.value.weight.fill_(0.3)
        fusion.value.bias.fill_(0.0)

    batch = {
        "endpoint_red": torch.tensor([[1.0, 2.0, 3.0]]),
        "service_metric": torch.tensor([[1.0, 1.0]]),
        "service_log": torch.tensor([[1.0]]),
    }

    with torch.no_grad():
        e_ep = torch.relu(fusion.ep_encoder[0](batch["endpoint_red"]))
        svc_in = torch.cat([batch["service_metric"], batch["service_log"]], dim=-1)
        e_svc = torch.relu(fusion.svc_encoder[0](svc_in))
        g_expected = torch.sigmoid(fusion.gate(torch.cat([e_ep, e_svc], dim=-1)))
        z_expected = e_ep + g_expected * fusion.value(e_svc)

    out = fusion(batch)
    torch.testing.assert_close(out, z_expected)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/test_gated_fusion.py -v`
Expected: FAIL，报错 `ModuleNotFoundError: No module named 'src.fusion.gated'`

- [ ] **Step 3: 写最小实现**

创建 `src/fusion/gated.py`：

```python
from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import MODALITY_ORDER, FusionModule


class GatedFusion(FusionModule):
    """L2：门控条件融合。

    g = sigmoid(gate([e_ep; e_svc]))
    z = e_ep + g ⊙ value(e_svc)

    e_ep 用 endpoint 级信息动态调制被 left-join 广播复制的 service 级
    特征（e_svc）该贡献多少，而非无条件全量拼接（对比 L1）。gate/value
    是门控/调制层，不是表征编码器，bias 保留默认 True
    （区别于 ep_encoder/svc_encoder 的 bias=False 约定）。
    """

    def __init__(self, modality_dims: dict[str, int], branch_dim: int = 16):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._branch_dim = branch_dim
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        self.ep_encoder = nn.Sequential(
            nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU()
        )
        self.svc_encoder = nn.Sequential(
            nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU()
        )
        self.gate = nn.Linear(2 * branch_dim, branch_dim)
        self.value = nn.Linear(branch_dim, branch_dim)

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        e_ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat(
            [modality_dict["service_metric"], modality_dict["service_log"]], dim=-1
        )
        e_svc = self.svc_encoder(svc_in)
        g = torch.sigmoid(self.gate(torch.cat([e_ep, e_svc], dim=-1)))
        return e_ep + g * self.value(e_svc)

    @property
    def output_dim(self) -> int:
        return self._branch_dim
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pytest tests/test_gated_fusion.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/fusion/gated.py tests/test_gated_fusion.py
git commit -m "[Feature]: 新增 L2 GatedFusion（门控条件融合）"
```

---

## Task 3: Hydra config 接入

**Files:**
- Create: `configs/fusion/independent_concat.yaml`
- Create: `configs/fusion/gated.yaml`

- [ ] **Step 1: 创建 L1 config**

创建 `configs/fusion/independent_concat.yaml`：

```yaml
_target_: src.fusion.independent_concat.IndependentConcatFusion
branch_dim: 16
```

- [ ] **Step 2: 创建 L2 config**

创建 `configs/fusion/gated.yaml`：

```yaml
_target_: src.fusion.gated.GatedFusion
branch_dim: 16
```

- [ ] **Step 3: 验证 Hydra 能正确 instantiate（用已有 contract_v1 数据跑一个极小 smoke run）**

Run:
```bash
conda run -n interface python scripts/train_baseline_v0.py \
  contract_dir=artifacts/contract_v1 \
  out=/tmp/l1_smoke/scores.parquet \
  fusion=independent_concat \
  training.epochs=1
```
Expected: 无报错退出，`/tmp/l1_smoke/scores.parquet` 生成。日志里应看到 `epoch 1/1 loss=...`，不应出现 Hydra "Could not find" 或 shape mismatch 错误。

Run:
```bash
conda run -n interface python scripts/train_baseline_v0.py \
  contract_dir=artifacts/contract_v1 \
  out=/tmp/l2_smoke/scores.parquet \
  fusion=gated \
  training.epochs=1
```
Expected: 同上，无报错退出。

若报错，先检查 `artifacts/contract_v1/schema.json` 是否存在（若不存在需先跑 `dvc repro build_contract_v1`），再检查报错信息里的 shape 是否与 Task 1/2 的 `output_dim` 定义一致。

清理 smoke 产物：
```bash
rm -rf /tmp/l1_smoke /tmp/l2_smoke
```

- [ ] **Step 4: Commit**

```bash
git add configs/fusion/independent_concat.yaml configs/fusion/gated.yaml
git commit -m "[Feature]: 新增 L1/L2 融合机制的 Hydra config"
```

---

## Task 4: 全量测试 + 正式跑 L1/L2 实验，记录结果

**Files:**
- Modify: `history/index.md`
- Create: `history/entries/012-fusion-l1-l2-gated-ablation.md`

- [ ] **Step 1: 跑全量测试套件确认无回归**

Run: `pytest tests/ -v`
Expected: 全部 PASS（新增 12 个测试 + 之前的全部通过，无 FAIL）

- [ ] **Step 2: 正式跑 L1 实验（Contract v1，完整 epoch）**

```bash
conda run -n interface python scripts/train_baseline_v0.py \
  contract_dir=artifacts/contract_v1 \
  out=artifacts/l1/scores.parquet \
  fusion=independent_concat \
  training.epochs=50

conda run -n interface python scripts/eval_baseline_v0.py \
  --scores artifacts/l1/scores.parquet \
  --out artifacts/l1/metrics.json
```
Expected: `artifacts/l1/metrics.json` 生成，含 `auroc`/`auprc` 字段。

- [ ] **Step 3: 正式跑 L2 实验（Contract v1，完整 epoch）**

```bash
conda run -n interface python scripts/train_baseline_v0.py \
  contract_dir=artifacts/contract_v1 \
  out=artifacts/l2/scores.parquet \
  fusion=gated \
  training.epochs=50

conda run -n interface python scripts/eval_baseline_v0.py \
  --scores artifacts/l2/scores.parquet \
  --out artifacts/l2/metrics.json
```
Expected: `artifacts/l2/metrics.json` 生成。

- [ ] **Step 4: 读取三组指标并记录对比**

Run:
```bash
cat artifacts/l1/metrics.json
cat artifacts/l2/metrics.json
```

把读到的 `auroc`/`auprc` 数值，与已有 v1 L0 baseline（AUROC=0.617 / AUPRC=0.327，来自 entry 011）并列记入下一步的 history entry。

- [ ] **Step 5: 撰写 history entry**

创建 `history/entries/012-fusion-l1-l2-gated-ablation.md`，结构遵照 `history/entries/_template.md`。正文需包含：
- 做了什么：L1（IndependentConcatFusion）、L2（GatedFusion）两个 `FusionModule` 实现 + Hydra config，Contract v1 上的实测 AUROC/AUPRC 三组对比（L0 vs L1 vs L2，把 Step 4 读到的真实数字填入，不要用占位符）
- 关键决策：门控机制本身不是论文创新点（entry 010 结论），本轮定位是"收集门控相对 L0 的实际收益数据，为后续新融合模型设计提供依据"；L1/L2 之间不做参数量对齐的理由（gate/value 是架构本身而非混淆变量）
- 坑/已知问题：如果 Step 2/3 实测过程中遇到任何报错或数值异常，记录在这里；若一切顺利，写明"本轮未遇到新坑"
- 遗留 TODO：L3（全局标量门控）是否需要做，取决于本轮 L1/L2 的实际数字差距；若 L2 相对 L1 收益显著，需要在下一轮讨论"真正的新融合模型"该往哪个方向设计增量

- [ ] **Step 6: 更新 history/index.md**

在 `history/index.md` 的 Entry 列表表格里追加一行（编号 012，日期 2026-07-15，类型 Experiment，标题"L1/L2 融合消融：独立编码器 + 门控条件融合实测"，影响域 `src/fusion/`, `configs/fusion/`, `tests/`）。

在"影响域索引"表格的"多模态融合（`src/fusion/`）"行追加 `, 012`。

在"横切主题"部分，若 L2 相对 L0/L1 有值得记录的横切结论（例如"门控收益是否显著"这类会被后续工作反复查阅的判断），补充一条要点；若本轮数字差距不构成新的横切结论，跳过这一步不写。

- [ ] **Step 7: Commit**

```bash
git add history/index.md history/entries/012-fusion-l1-l2-gated-ablation.md artifacts/l1/metrics.json artifacts/l2/metrics.json
git commit -m "[Experiment]: L1/L2 融合消融实测结果记录"
```

注意：`artifacts/l1/scores.parquet`、`artifacts/l2/scores.parquet` 属于大文件产物，若 `.gitignore` 已排除 `artifacts/**/*.parquet`，不要强行加入 git（先跑 `git status` 确认哪些文件被跟踪）；只提交 `metrics.json`（小文件，是本轮结论的直接依据）。

---

## Self-Review Checklist（写完后核对，仅供实施前參考，不是任务步骤）

- Spec 的"架构设计"一节 → Task 1（L1）+ Task 2（L2）覆盖
- Spec 的"文件改动"一节 → Task 1/2/3 覆盖全部 6 个文件（2 个 fusion 类 + 2 个测试 + 2 个 config）
- Spec 的"不做的事" → 全部 Task 均未触碰 `DeepSVDD`、`train_baseline_v0.py`、`dvc.yaml`、L3、参数量对齐
- Spec 的"验证方式" → Task 4 完整覆盖（跑 L1/L2 + eval + 对比 L0 数字）
- Spec 的"遗留 TODO" → Task 4 Step 5 的 history entry 里承接
