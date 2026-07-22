# RG Coupling Containment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 PR#14 留在共享代码路径里的 3 处 RG 专属耦合（`train_baseline_v0.py` 字符串分支、`build_contract.py` 无条件 fit 统计量、`dvc.yaml` 默认 DAG 混入 RG stage）收回 RG 自己的模块/配置,不改动 RG 门控算法本身。

**Architecture:** 引入 `FusionModule.from_contract` classmethod 让融合机制自己声明如何从 contract 产物构造(RG 覆写、L0/L1/L2 走默认),基类 `forward` 统一接受 `endpoint_id`;用 `ContractConfig.fit_endpoint_baseline_stats` 开关(照抄 `expand_train_pool` 模式)取代 `contract_version=="v1"` 作为"是否产出 RG 需要的 endpoint_id 列与基线统计量"的判据;把 3 个 RG 专属 DVC stage 隔离到 `dvc_reliability_gate/dvc.yaml` 子目录(DVC 3.67.1 强制 pipeline 文件名为 `dvc.yaml`)。

**Tech Stack:** Python 3, PyTorch, Hydra(`hydra.utils.instantiate`/`get_class`), OmegaConf, DVC 3.67.1, pytest, pandas。

**Spec:** `docs/superpowers/specs/2026-07-21-rg-coupling-containment-design.md`

---

## Commit 边界说明(与 spec §6 的一处偏差,已确认必要)

spec §6 把"config 开关取值"放在 commit 3。但实测约束:`fit_endpoint_baseline_stats` 默认 `False`,若 commit 2 只改代码不改 config,则 `v1_expanded_pool.yaml` 走默认 `False` → 不产出 `endpoint_baseline_stats.json` → RG e2e smoke test(`tests/test_e2e_reliability_gate_smoke.py`,pytest 用例)当场变红。为保证**每个 commit 下 `pytest tests/` 全绿**,config 取值必须并入 commit 2。4 个 commit 的数量与类型(`[Refactor]`×3 + `[Docs]`×1)不变。commit 3 收窄为纯 `dvc.yaml` 层面改动(stage 拆分 + 过时 outs 清理 + `test_dvc_pipeline_v1.py` 更新)。

`dvc repro` 不在 pytest 覆盖内(测试只解析 dvc.yaml,不实际 `dvc repro`),因此 commit 2 后 `dvc repro build_contract_v1` 会因声明了不再产出的 `endpoint_baseline_stats.json` 而失败——这个 DVC 层面的不一致由 commit 3 修复。commit 2 的绿是 pytest 绿;端到端 `dvc repro` 的绿在 commit 3 之后。

---

## File Structure

**Commit 1(fusion 解耦):**
- Modify: `src/fusion/base.py` — 新增 `from_contract` classmethod + `forward` 加 `endpoint_id` 参数
- Modify: `src/fusion/reliability_gate.py` — 覆写 `from_contract`
- Modify: `src/fusion/early_concat.py` / `independent_concat.py` / `gated.py` — `forward` 加 `endpoint_id=None`(接受即忽略)
- Modify: `scripts/train_baseline_v0.py` — 删除 `is_reliability_gate` 全部分支,改用 `from_contract`
- Create: `tests/test_fusion_from_contract.py`

**Commit 2(contract 开关):**
- Modify: `src/contracts/contract_config.py` — 新增 `fit_endpoint_baseline_stats` 字段 + loader
- Modify: `scripts/build_contract.py` — `endpoint_id_map` 派生与 `EndpointBaselineStats` fit/save 改由开关门控
- Modify: `configs/contract/v1.yaml` — 加 `fit_endpoint_baseline_stats: false`
- Modify: `configs/contract/v1_expanded_pool.yaml` — 加 `fit_endpoint_baseline_stats: true`
- Modify: `tests/test_contract_config.py` — 新增开关字段测试
- Modify: `tests/test_build_contract_v1_endpoint_id.py` — `_run_v1_build` 重指向 flag-on config + 新增 flag-off 不产出 stats 的测试

**Commit 3(DVC stage 隔离):**
- Create: `dvc_reliability_gate/dvc.yaml` — 3 个 RG 专属 stage(带 `wdir: ..`)
- Modify: `dvc.yaml` — 删除 3 个 RG stage + 删除 `build_contract_v1.outs` 里的 `endpoint_baseline_stats.json`
- Modify: `tests/test_dvc_pipeline_v1.py` — RG stage 断言改读新文件位置

**Commit 4(文档):**
- Modify: `CLAUDE.md` — Commands 小节 RG 命令 + Known Gotchas 的 RG 耦合技术债条目
- Create: `history/entries/015-*.md` — 占位(内容待落地后补)
- Modify: `history/index.md` — 加 015 索引项

---

## Task 1: Fusion 层解耦(`from_contract` 钩子 + `endpoint_id` 统一)

**Files:**
- Modify: `src/fusion/base.py`
- Modify: `src/fusion/reliability_gate.py`
- Modify: `src/fusion/early_concat.py`
- Modify: `src/fusion/independent_concat.py`
- Modify: `src/fusion/gated.py`
- Modify: `scripts/train_baseline_v0.py`
- Test: `tests/test_fusion_from_contract.py`

- [ ] **Step 1: 写失败测试 `tests/test_fusion_from_contract.py`**

```python
"""FusionModule.from_contract 钩子:L0/L1/L2 走基类默认(等价 hydra.instantiate),
RG 覆写自行加载 endpoint_baseline_stats + 派生 id_to_endpoint_key。同时验证
基类 forward 新增的 endpoint_id 参数对 L0/L1/L2 是接受即忽略(不改变输出)。"""

from pathlib import Path

import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.early_concat import EarlyConcatFusion
from src.fusion.gated import GatedFusion
from src.fusion.independent_concat import IndependentConcatFusion
from src.fusion.reliability_gate import ReliabilityGatedFusion

REPO_ROOT = Path(__file__).parents[1]
_DIMS = {"endpoint_red": 10, "service_metric": 5, "service_log": 3}


def _sample_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {m: torch.randn(batch_size, d) for m, d in _DIMS.items()}


def test_default_from_contract_matches_direct_instantiate(tmp_path):
    """L0 未覆写 from_contract,应走基类默认路径,构造结果与 hydra.instantiate 等价。
    contract_dir 在默认实现里被忽略(传一个不含任何 sidecar 的 tmp 目录也不报错)。"""
    cfg = OmegaConf.create({"_target_": "src.fusion.early_concat.EarlyConcatFusion"})
    fusion = EarlyConcatFusion.from_contract(cfg, contract_dir=tmp_path, modality_dims=_DIMS)
    assert isinstance(fusion, EarlyConcatFusion)
    assert fusion.output_dim == sum(_DIMS.values())


def test_l0_l1_l2_forward_accepts_and_ignores_endpoint_id():
    """基类 forward 统一新增 endpoint_id 参数后,L0/L1/L2 必须接受它但输出不受其影响
    (它们没有 per-endpoint 路由概念)。这样 train 脚本才能无条件传 endpoint_id。"""
    batch = _sample_batch()
    eid = torch.arange(batch["endpoint_red"].shape[0])
    for fusion in (
        EarlyConcatFusion(modality_dims=_DIMS),
        IndependentConcatFusion(modality_dims=_DIMS),
        GatedFusion(modality_dims=_DIMS),
    ):
        fusion.eval()
        with torch.no_grad():
            out_without = fusion(batch)
            out_with = fusion(batch, endpoint_id=eid)
        assert torch.equal(out_without, out_with)


def test_reliability_gate_from_contract_loads_baseline_stats(tmp_path):
    """RG 覆写 from_contract:从 contract_dir 读 endpoint_baseline_stats.json,
    并用 configs/contract/endpoint_to_service.yaml 派生 id_to_endpoint_key。
    验证加载的映射与权威派生函数一致、统计量覆盖到 fit 过的 endpoint。"""
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    key = sorted(ep_to_svc)[0]
    df = pd.DataFrame(
        {
            "endpoint_key": [key, key],
            "endpoint_red__a": [0.1, 0.2],
            "endpoint_red__b": [0.3, 0.5],
            "service_metric__c": [0.4, 0.6],
        }
    )
    stats = EndpointBaselineStats(
        red_cols=["endpoint_red__a", "endpoint_red__b"], svc_cols=["service_metric__c"]
    )
    stats.fit(df)
    stats.save(tmp_path / "endpoint_baseline_stats.json")

    cfg = OmegaConf.create(
        {
            "_target_": "src.fusion.reliability_gate.ReliabilityGatedFusion",
            "branch_dim": 16,
            "dropout": 0.1,
        }
    )
    fusion = ReliabilityGatedFusion.from_contract(
        cfg, contract_dir=tmp_path, modality_dims=_DIMS
    )
    assert isinstance(fusion, ReliabilityGatedFusion)
    assert fusion.output_dim == 16
    assert fusion._id_to_key == _derive_id_to_key(ep_to_svc)
    assert key in fusion._baseline.fitted_endpoints()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_fusion_from_contract.py -v`
Expected: FAIL — `AttributeError: type object 'EarlyConcatFusion' has no attribute 'from_contract'`(基类尚无 `from_contract`);L0/L1/L2 的 `forward` 尚不接受 `endpoint_id` → `TypeError`。

- [ ] **Step 3: 改 `src/fusion/base.py`**

整份文件替换为:

```python
from __future__ import annotations

from pathlib import Path

import hydra
import torch
import torch.nn as nn

# 三个模态的固定顺序,所有融合机制共享(不绑定任何具体子类)。
# 训练脚本据此拼 batch/tensor,顺序错位会导致特征维度错位但不报错——务必保持全局唯一来源。
MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @classmethod
    def from_contract(
        cls, cfg, *, contract_dir: Path, modality_dims: dict[str, int]
    ) -> "FusionModule":
        """从 contract 产物构造融合机制。默认实现等价于
        hydra.utils.instantiate(cfg, modality_dims=...),即 L0/L1/L2 的现有行为——
        contract_dir 在默认实现里被忽略。需要读取 contract 侧产物(如 per-endpoint
        基线统计量)的融合机制覆写此方法,把加载逻辑收进自己模块内,训练脚本因此
        无需再按 _target_ 字符串分支为特定机制注入运行时 kwargs。"""
        return hydra.utils.instantiate(cfg, modality_dims=modality_dims)
```

注意:`ABC`/`abstractmethod` 的 import 必须保留(上面替换体省略了原第 3 行 `from abc import ABC, abstractmethod`——实际编辑时保留它)。完整 import 块应为:

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import torch
import torch.nn as nn
```

- [ ] **Step 4: 改 `src/fusion/early_concat.py` 的 forward 签名**

```python
    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 被接受但忽略:L0 无 per-endpoint 路由概念,仅为统一 FusionModule
        # 接口以便训练脚本无条件传参。
        return torch.cat([modality_dict[m] for m in MODALITY_ORDER], dim=-1)
```

- [ ] **Step 5: 改 `src/fusion/independent_concat.py` 的 forward 签名**

```python
    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 接受即忽略(见 EarlyConcatFusion.forward 注释)。
        ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        svc = self.svc_encoder(svc_in)
        return torch.cat([ep, svc], dim=-1)
```

- [ ] **Step 6: 改 `src/fusion/gated.py` 的 forward 签名**

```python
    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 接受即忽略(见 EarlyConcatFusion.forward 注释)。
        e_ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        e_svc = self.svc_encoder(svc_in)
        g = torch.sigmoid(self.gate(torch.cat([e_ep, e_svc], dim=-1)))
        return e_ep + g * self.value(e_svc)
```

- [ ] **Step 7: 给 `src/fusion/reliability_gate.py` 加 `from_contract` 覆写**

在文件顶部 import 区补充(现有 import 保留):

```python
from pathlib import Path

import hydra
import yaml

from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
```

在模块级常量区(`_DEVIATION_THRESHOLD` 附近)加:

```python
# endpoint_to_service.yaml 是 id_to_endpoint_key 反查表的唯一权威来源,必须与
# build_contract.py 派生 endpoint_id 列时用的同一份 sorted() 逻辑一致
# (src/contracts/endpoint_id_mapping.py),否则整数 id 在两处指向不同 endpoint_key,
# 门控静默查错基线统计量。reliability_gate.py 在 src/fusion/ 下,parents[2] 为 repo 根。
_EP_TO_SVC_PATH = Path(__file__).resolve().parents[2] / "configs/contract/endpoint_to_service.yaml"
```

在 `ReliabilityGatedFusion` 类体内(`__init__` 之前或 `output_dim` 之后均可)加 classmethod:

```python
    @classmethod
    def from_contract(cls, cfg, *, contract_dir, modality_dims):
        """RG 专属构造:从 contract_dir 加载 per-endpoint 基线统计量,并派生
        id_to_endpoint_key 反查表,再交给 hydra.utils.instantiate 注入这两个运行时对象。
        这段逻辑原先散落在 train_baseline_v0.py 的 is_reliability_gate 分支里,现收进
        RG 自己模块——训练脚本对"RG 需要什么"一无所知即可正确构造。"""
        baseline_stats = EndpointBaselineStats.load(
            Path(contract_dir) / "endpoint_baseline_stats.json"
        )
        ep_to_svc = yaml.safe_load(_EP_TO_SVC_PATH.read_text())
        id_to_key = _derive_id_to_endpoint_key(ep_to_svc)
        return hydra.utils.instantiate(
            cfg,
            modality_dims=modality_dims,
            endpoint_baseline_stats=baseline_stats,
            id_to_endpoint_key=id_to_key,
        )
```

(RG 的 `forward` 已经带 `endpoint_id: torch.Tensor | None = None`,无需改。)

- [ ] **Step 8: 运行新测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_fusion_from_contract.py -v`
Expected: PASS(3 个测试全绿)。

- [ ] **Step 9: 改 `scripts/train_baseline_v0.py` — 删 `is_reliability_gate`,改用 `from_contract`**

9a. 删除不再使用的 import(第 33、35 行):
```python
from src.contracts.endpoint_id_mapping import id_to_endpoint_key as _derive_id_to_endpoint_key
from src.data.endpoint_baseline_stats import EndpointBaselineStats
```
以及 `import yaml`(第 28 行)——若文件内 `yaml` 无其他用途则删除。保留 `import hydra`、`from src.fusion.base import MODALITY_ORDER, FusionModule`。

9b. `_train` 去掉 `is_reliability_gate` 参数,forward 无条件传 `endpoint_id`:
```python
def _train(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    fusion.train()
    svdd.train()
    for epoch in range(epochs):
        total = 0.0
        n_batches = 0
        for batch in loader:
            # 所有融合机制的 forward 现在统一接受 endpoint_id(L0/L1/L2 忽略,
            # RG 使用);v0/L0 的 batch 无该 key 时 .get() 返回 None,走默认路径。
            x = fusion(
                {m: batch[m] for m in MODALITY_ORDER}, endpoint_id=batch.get("endpoint_id")
            )
            loss = svdd.svdd_loss(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            n_batches += 1
        LOG.info("epoch %d/%d loss=%.6f", epoch + 1, epochs, total / max(n_batches, 1))
```

9c. `_infer` 同样去掉 `is_reliability_gate` 参数:
```python
@torch.no_grad()
def _infer(
    fusion: FusionModule,
    svdd: DeepSVDD,
    loader: DataLoader,
) -> dict[str, float]:
    fusion.eval()
    svdd.eval()
    scores: dict[str, float] = {}
    for batch in loader:
        x = fusion(
            {m: batch[m] for m in MODALITY_ORDER}, endpoint_id=batch.get("endpoint_id")
        )
        s = svdd.score(x)
        for sid, val in zip(batch["sample_id"], s.tolist()):
            scores[sid] = val
    return scores
```

9d. `main()` 里替换 fusion 构造块(现第 136-158 行整段):
```python
    modality_dims = _modality_dims(schema)
    # 融合机制自己声明如何从 contract 产物构造:L0/L1/L2 走 FusionModule.from_contract
    # 的默认实现(等价 hydra.instantiate),RG 覆写它自行加载 endpoint_baseline_stats
    # 并派生 id_to_endpoint_key。训练脚本不再按 _target_ 字符串分支注入 RG 专属 kwargs。
    fusion = hydra.utils.get_class(cfg.fusion._target_).from_contract(
        cfg.fusion, contract_dir=contract_dir, modality_dims=modality_dims
    )
    svdd = hydra.utils.instantiate(cfg.model, input_dim=fusion.output_dim)
```

9e. `main()` 里 SVDD center 初始化(现第 169-170 行)简化:
```python
    init_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=_collate)
    init_batch = next(iter(init_loader))
    svdd.init_center(
        fusion({m: init_batch[m] for m in MODALITY_ORDER}, endpoint_id=init_batch.get("endpoint_id"))
    )
```

9f. `_train`/`_infer` 调用点(现第 188-195、204 行)去掉 `is_reliability_gate=...`:
```python
    _train(fusion, svdd, train_loader, epochs=cfg.training.epochs, optimizer=optimizer)
```
```python
    score_map = _infer(fusion, svdd, eval_loader)
```

optimizer 构造(第 177-179 行,`params=list(svdd.parameters()) + list(fusion.parameters())`)与 `cfg.fusion_checkpoint` 保存块(第 229-233 行)**保持不变**。

- [ ] **Step 10: 运行相关单测确认训练脚本改动无回归**

Run: `conda run -n interface python -m pytest tests/test_train_baseline_v0.py tests/test_early_concat_fusion.py tests/test_independent_concat_fusion.py tests/test_gated_fusion.py tests/test_reliability_gated_fusion.py -v`
Expected: PASS(现有测试调用 `forward(modality_dict)` 因 `endpoint_id` 有默认值仍合法;`_collate` 行为未变)。

- [ ] **Step 11: 跑 RG e2e smoke,验证 from_contract 路径端到端可用**

Run: `conda run -n interface python -m pytest tests/test_e2e_reliability_gate_smoke.py -v`
Expected: PASS。该测试用 `v1_expanded_pool.yaml` 构建(commit 1 未动 build_contract,`endpoint_baseline_stats.json` 仍照旧产出),再走新的 `from_contract` 构造 RG 训练,scores 行数与 `eval_all` 一致。

- [ ] **Step 12: Commit**

```bash
git add src/fusion/base.py src/fusion/early_concat.py src/fusion/independent_concat.py \
        src/fusion/gated.py src/fusion/reliability_gate.py scripts/train_baseline_v0.py \
        tests/test_fusion_from_contract.py
git commit -m "[Refactor]: FusionModule.from_contract 钩子解耦 RG 专属构造

基类新增 from_contract classmethod(默认=hydra.instantiate),forward 统一接受
endpoint_id;RG 覆写 from_contract 自行加载 endpoint_baseline_stats 并派生
id_to_endpoint_key;train_baseline_v0.py 删除 is_reliability_gate 全部字符串分支。"
```

---

## Task 2: Contract 开关(`fit_endpoint_baseline_stats`)

**Files:**
- Modify: `src/contracts/contract_config.py`
- Modify: `scripts/build_contract.py`
- Modify: `configs/contract/v1.yaml`
- Modify: `configs/contract/v1_expanded_pool.yaml`
- Test: `tests/test_contract_config.py`
- Test: `tests/test_build_contract_v1_endpoint_id.py`

- [ ] **Step 1: 写失败测试 — `tests/test_contract_config.py` 新增开关字段测试**

在文件末尾追加:
```python
def test_fit_endpoint_baseline_stats_defaults_false(tmp_path):
    """新增开关默认 False(与 expand_train_pool 同款模式):config 不写该字段时,
    build 不产出 endpoint_id 列与 endpoint_baseline_stats.json。"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fit_endpoint_baseline_stats is False


def test_fit_endpoint_baseline_stats_override_true(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fit_endpoint_baseline_stats: true\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fit_endpoint_baseline_stats is True
```
(确认文件顶部已 `from src.contracts.contract_config import load_contract_config`;若无则加。)

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n interface python -m pytest tests/test_contract_config.py -k fit_endpoint_baseline_stats -v`
Expected: FAIL — `AttributeError: 'ContractConfig' object has no attribute 'fit_endpoint_baseline_stats'`。

- [ ] **Step 3: 改 `src/contracts/contract_config.py` — 加字段 + loader**

`ContractConfig` dataclass 里在 `expand_train_pool` 之后加:
```python
    # v1 专属(RG 消费):是否 fit 并落盘 per-endpoint 双分支(ep/svc)normal-only
    # 基线统计量(endpoint_baseline_stats.json),并同步产出 endpoint_id 列。默认
    # False——L0/L1/L2 对比基线不需要这份统计量,不产出可避免 build_contract_v1
    # 无谓多算一遍并少一个产物依赖。ReliabilityGatedFusion 用 v1_expanded_pool.yaml
    # 显式打开。v0 不消费此字段(走默认 False)。取值与 expand_train_pool 相互独立。
    fit_endpoint_baseline_stats: bool = False
```
`load_contract_config` 的 `return ContractConfig(...)` 里加:
```python
        fit_endpoint_baseline_stats=raw.get("fit_endpoint_baseline_stats", False),
```

- [ ] **Step 4: 运行确认 config 测试通过**

Run: `conda run -n interface python -m pytest tests/test_contract_config.py -v`
Expected: PASS。

- [ ] **Step 5: 改 `scripts/build_contract.py` — 门控 endpoint_id_map 派生**

`main()` 第 344 行:
```python
    endpoint_id_map = _derive_endpoint_id_map(ep_to_svc) if cfg.contract_version == "v1" else None
```
改为:
```python
    # endpoint_id 列只在需要 fit per-endpoint 基线统计量时才产出(RG 专属)。用契约
    # 版本号(v0/v1)当代理判据语义不准——v1 不等于"需要 RG 统计量"。改由 cfg 显式声明。
    endpoint_id_map = _derive_endpoint_id_map(ep_to_svc) if cfg.fit_endpoint_baseline_stats else None
```

- [ ] **Step 6: 改 `scripts/build_contract.py` — `_write_v1` 门控 stats fit/save**

6a. 第 460-461 行调用点传入开关:
```python
    if cfg.contract_version == "v1":
        _write_v1(out, full, normal_mask, args.seed, cfg.expand_train_pool)
```
改为:
```python
    if cfg.contract_version == "v1":
        _write_v1(
            out,
            full,
            normal_mask,
            args.seed,
            cfg.expand_train_pool,
            cfg.fit_endpoint_baseline_stats,
        )
```

6b. `_write_v1` 签名(第 472-474 行)加参数:
```python
def _write_v1(
    out: Path,
    full: pd.DataFrame,
    normal_mask: pd.Series,
    seed: int,
    expand_train_pool: bool,
    fit_endpoint_baseline_stats: bool,
) -> None:
```

6c. `_write_v1` 末尾的 `EndpointBaselineStats` 计算/fit/告警/save 整块(现第 543-562 行)用开关包裹:
```python
    if not fit_endpoint_baseline_stats:
        return

    train_fit_normalized = parts["train_fit"]
    red_cols = [c for c in train_fit_normalized.columns if c.startswith("endpoint_red__")]
    svc_cols = [
        c
        for c in train_fit_normalized.columns
        if c.startswith(("service_metric__", "service_log__"))
    ]
    baseline_stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    baseline_stats.fit(train_fit_normalized)
    for branch in ("ep", "svc"):
        degenerate = baseline_stats.degenerate_columns(branch)
        if degenerate:
            LOG.warning(
                "EndpointBaselineStats %s 分支在整个 fit 集合上退化(全 NaN/零方差),"
                "std 兜底为 epsilon,偏离量 z-score 在该列上会被放大,"
                "可能扭曲 ReliabilityGatedFusion 门控输入信号: %s",
                branch,
                degenerate,
            )
    baseline_stats.save(out / "endpoint_baseline_stats.json")
```
(`_write_v1` 内其余逻辑——train_fit/train_val/holdout 落盘、expand_train_pool 分支——保持不变,`fit_endpoint_baseline_stats` 与 `expand_train_pool` 相互独立。)

- [ ] **Step 7: 改配置 — 两份 v1 config 加开关**

`configs/contract/v1.yaml` 在 `expand_train_pool: false` 之后加:
```yaml
# per-endpoint 基线统计量(endpoint_baseline_stats.json)+ endpoint_id 列:仅 RG 需要。
# L0/L1/L2 对比基线走本配置,不产出这些,避免多算与多余产物依赖。
fit_endpoint_baseline_stats: false
```
`configs/contract/v1_expanded_pool.yaml` 在 `expand_train_pool: true` 之后加:
```yaml
# RG(ReliabilityGatedFusion)消费 per-endpoint 基线统计量做偏离量门控,必须产出。
fit_endpoint_baseline_stats: true
```

- [ ] **Step 8: 改 `tests/test_build_contract_v1_endpoint_id.py` — 重指向 + 新增 flag-off 测试**

8a. `_run_v1_build`(第 21-37 行)的 config 从 `v1.yaml` 改为 `v1_expanded_pool.yaml`(该文件所有"应产出 endpoint_id/sidecar"的测试现在依赖 flag-on 的配置;`v1_expanded_pool.yaml` 的 `train_fit` 与 `v1.yaml` 一致,fit 范围不受 `expand_train_pool` 影响,故 sidecar 数值重算测试不变):
```python
def _run_v1_build(out_dir: Path) -> None:
    # 这些测试验证"开启 fit_endpoint_baseline_stats 时产出 endpoint_id 列与
    # endpoint_baseline_stats.json"。该开关现由 config 声明,v1_expanded_pool.yaml
    # 打开它;v1.yaml 关闭(见 test_v1_without_flag_omits_endpoint_stats)。train_fit
    # 的构成不受 expand_train_pool 影响,故 sidecar 数值重算断言仍成立。
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1_expanded_pool.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
```

8b. 新增测试锁定 flag-off 行为(v1.yaml 不产出 endpoint_id/sidecar),放在 `test_endpoint_id_not_added_in_v0` 附近:
```python
def test_v1_without_flag_omits_endpoint_stats(tmp_path):
    """v1.yaml 现在 fit_endpoint_baseline_stats=false:不产出 endpoint_id 列,
    也不产出 endpoint_baseline_stats.json。这是把"是否产出 RG 统计量"从
    contract_version=="v1" 解耦成显式开关后的核心不变量——L0/L1/L2 基线
    (走 v1.yaml)不再无谓 fit 一份没人消费的统计量。"""
    out_dir = tmp_path / "contract_v1_noflag"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    for name in ("train_fit", "eval_all"):
        df = pd.read_parquet(out_dir / f"{name}.parquet")
        assert "endpoint_id" not in df.columns
    assert not (out_dir / "endpoint_baseline_stats.json").exists()
```

- [ ] **Step 9: 运行 contract build 相关测试确认全绿**

Run: `conda run -n interface python -m pytest tests/test_build_contract_v1_endpoint_id.py tests/test_contract_v1_train_pool.py tests/test_contract_config.py -v`
Expected: PASS。`test_endpoint_id_not_added_in_v0`(v0 flag 默认 False)、新增的 `test_v1_without_flag_omits_endpoint_stats`(v1 flag=false)、以及重指向后的 endpoint_id/sidecar 测试(v1_expanded flag=true)全绿。

- [ ] **Step 10: 回跑 RG e2e smoke,确认 flag 打开后 sidecar 仍产出**

Run: `conda run -n interface python -m pytest tests/test_e2e_reliability_gate_smoke.py -v`
Expected: PASS(`v1_expanded_pool.yaml` 现 flag=true → 产出 sidecar → RG 训练正常)。

- [ ] **Step 11: Commit**

```bash
git add src/contracts/contract_config.py scripts/build_contract.py \
        configs/contract/v1.yaml configs/contract/v1_expanded_pool.yaml \
        tests/test_contract_config.py tests/test_build_contract_v1_endpoint_id.py
git commit -m "[Refactor]: fit_endpoint_baseline_stats 开关取代 contract_version 判据

新增 ContractConfig.fit_endpoint_baseline_stats(默认 false,照抄 expand_train_pool
模式)门控 endpoint_id 列派生与 EndpointBaselineStats fit/save;v1.yaml=false、
v1_expanded_pool.yaml=true。L0/L1/L2 基线不再无谓 fit 无人消费的统计量。"
```

---

## Task 3: DVC stage 隔离

**Files:**
- Create: `dvc_reliability_gate/dvc.yaml`
- Modify: `dvc.yaml`
- Test: `tests/test_dvc_pipeline_v1.py`

- [ ] **Step 1: 写失败测试 — 更新 `tests/test_dvc_pipeline_v1.py` 读新位置**

`test_build_contract_v1_expanded_uses_expanded_config` 改为从子目录 dvc.yaml 读:
```python
def test_build_contract_v1_expanded_uses_expanded_config():
    """build_contract_v1_expanded(RG 专属)已隔离到 dvc_reliability_gate/dvc.yaml,
    必须指向 v1_expanded_pool.yaml 且 expand_train_pool=true。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc_reliability_gate/dvc.yaml").read_text())
    stage = pipeline["stages"]["build_contract_v1_expanded"]

    config_path = _stage_config_path(stage)
    assert config_path == "configs/contract/v1_expanded_pool.yaml"
    assert config_path in stage.get("deps", [])

    cfg = yaml.safe_load((REPO_ROOT / config_path).read_text())
    assert cfg.get("expand_train_pool", False) is True
    assert cfg.get("fit_endpoint_baseline_stats", False) is True
```
`test_train_v1_reliability_gate_reads_expanded_contract_dir` 同样改读新文件:
```python
def test_train_v1_reliability_gate_reads_expanded_contract_dir():
    pipeline = yaml.safe_load((REPO_ROOT / "dvc_reliability_gate/dvc.yaml").read_text())
    stage = pipeline["stages"]["train_v1_reliability_gate"]

    cmd = stage["cmd"]
    assert "contract_dir=artifacts/contract_v1_expanded" in cmd

    deps = stage.get("deps", [])
    assert any(d.startswith("artifacts/contract_v1_expanded/") for d in deps)
    assert not any(
        d.startswith("artifacts/contract_v1/") for d in deps
    ), "train_v1_reliability_gate 的 deps 不应指向未扩容的 contract_v1/"
```
新增一个断言 RG stage 已从根 dvc.yaml 移除、且 build_contract_v1 不再声明 sidecar 输出:
```python
def test_root_dvc_excludes_rg_stages_and_stale_sidecar_output():
    """3 个 RG 专属 stage 必须从根 dvc.yaml 移除(裸 dvc repro 不再触发它们);
    build_contract_v1 的 outs 不再声明 endpoint_baseline_stats.json
    (v1.yaml flag=false 后不再产出该文件)。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stages = pipeline["stages"]
    for rg_stage in (
        "build_contract_v1_expanded",
        "train_v1_reliability_gate",
        "eval_v1_reliability_gate",
    ):
        assert rg_stage not in stages
    outs = stages["build_contract_v1"].get("outs", [])
    assert "artifacts/contract_v1/endpoint_baseline_stats.json" not in outs
```
`test_build_contract_v1_uses_non_expanded_config` 与 `test_v1_and_v1_expanded_configs_share_identical_modalities` 不改(仍读根 dvc.yaml / configs)。

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n interface python -m pytest tests/test_dvc_pipeline_v1.py -v`
Expected: FAIL — `dvc_reliability_gate/dvc.yaml` 不存在(`FileNotFoundError`);根 dvc.yaml 仍含 RG stage 与 sidecar output。

- [ ] **Step 3: 创建 `dvc_reliability_gate/dvc.yaml`**

把根 dvc.yaml 现第 123-184 行的 3 个 stage 整段搬入,每个 stage 加 `wdir: ..`(使 cmd/outs/deps/metrics 的相对路径继续相对 repo 根解析,路径字符串本身不变):

```yaml
# RG(ReliabilityGatedFusion)专属 pipeline,从根 dvc.yaml 隔离出来:裸 `dvc repro`
# 不再触发这 3 个 stage(约 20min rebuild)。触发方式:`dvc repro dvc_reliability_gate/dvc.yaml`
# (单跑某 stage 用 `dvc_reliability_gate/dvc.yaml:stage_name`)。DVC 3.67.1 强制 pipeline
# 文件名字面量为 dvc.yaml,故放子目录而非根级 dvc_reliability_gate.yaml。每个 stage 的
# wdir: .. 让所有相对路径继续相对 repo 根解析。
stages:
  build_contract_v1_expanded:
    wdir: ..
    cmd: python scripts/build_contract.py --config configs/contract/v1_expanded_pool.yaml
      --dataset configs/data/merged_v2.yaml --out-dir artifacts/contract_v1_expanded --seed 42
    deps:
      - scripts/build_contract.py
      - src/preprocessors
      - src/contracts/contract_v0.py
      - src/contracts/split_v1.py
      - src/data/normalization.py
      - src/data/dataset_config.py
      - src/data/endpoint_baseline_stats.py
      - configs/contract/v1_expanded_pool.yaml
      - configs/contract/endpoint_to_service.yaml
      - configs/data/merged_v2.yaml
    outs:
      - artifacts/contract_v1_expanded/train_fit.parquet
      - artifacts/contract_v1_expanded/train_val.parquet
      - artifacts/contract_v1_expanded/eval_normal_holdout.parquet
      - artifacts/contract_v1_expanded/train.parquet
      - artifacts/contract_v1_expanded/eval_all.parquet
      - artifacts/contract_v1_expanded/normalization_stats.json
      - artifacts/contract_v1_expanded/endpoint_baseline_stats.json
      - artifacts/contract_v1_expanded/schema.json

  train_v1_reliability_gate:
    wdir: ..
    cmd: python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1_expanded
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
      - artifacts/contract_v1_expanded/train.parquet
      - artifacts/contract_v1_expanded/eval_all.parquet
      - artifacts/contract_v1_expanded/schema.json
      - artifacts/contract_v1_expanded/endpoint_baseline_stats.json
    outs:
      - artifacts/baseline_v1_reliability_gate/scores.parquet

  eval_v1_reliability_gate:
    wdir: ..
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

- [ ] **Step 4: 改根 `dvc.yaml` — 删 3 个 RG stage + 清理 build_contract_v1 outs**

4a. 删除根 dvc.yaml 第 123-184 行(`build_contract_v1_expanded`/`train_v1_reliability_gate`/`eval_v1_reliability_gate` 三段)。

4b. 删除 `build_contract_v1` stage 的 outs 里(现第 89 行):
```yaml
      - artifacts/contract_v1/endpoint_baseline_stats.json
```
(该文件在 v1.yaml flag=false 后不再产出;`src/data/endpoint_baseline_stats.py` 作为 build_contract.py 的模块级 import 依赖保留在 deps 中不动。)

- [ ] **Step 5: 运行 dvc pipeline 测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_dvc_pipeline_v1.py -v`
Expected: PASS(5 个测试:2 个读新子目录文件、1 个新增排除断言、2 个读根 dvc.yaml/config 不变)。

- [ ] **Step 6: 校验两份 dvc.yaml 语法合法(dvc dag 解析,不实际重跑)**

Run: `conda run -n interface dvc dag dvc_reliability_gate/dvc.yaml 2>&1 | head -20`
Expected: 打印 RG 三段 stage 的 DAG,无 `bad DVC file name` 或 parse error。

Run: `conda run -n interface dvc status build_contract_v1 2>&1 | head -20`
Expected: 正常解析(可能报 stage 需重跑,但不应有 YAML/output 声明错误)。

- [ ] **Step 7: Commit**

```bash
git add dvc.yaml dvc_reliability_gate/dvc.yaml tests/test_dvc_pipeline_v1.py
git commit -m "[Refactor]: RG 专属 DVC stage 隔离到 dvc_reliability_gate/dvc.yaml

3 个 RG stage 移出根 dvc.yaml(裸 dvc repro 不再触发 ~20min 无谓 rebuild),
每 stage 带 wdir: ..;删除 build_contract_v1 outs 里不再产出的
endpoint_baseline_stats.json;test_dvc_pipeline_v1.py 断言改读新位置。"
```

---

## Task 4: 文档同步

**Files:**
- Modify: `CLAUDE.md`
- Create: `history/entries/015-rg-coupling-containment.md`
- Modify: `history/index.md`

- [ ] **Step 1: 改 `CLAUDE.md` Commands 小节 RG 触发命令**

把 `# === Contract v1 训练池扩容开关 ===` 小节里的:
```bash
dvc repro build_contract_v1_expanded train_v1_reliability_gate eval_v1_reliability_gate
```
改为:
```bash
dvc repro dvc_reliability_gate/dvc.yaml
```
并在该行附近补一句说明:RG 专属 pipeline 已隔离到 `dvc_reliability_gate/dvc.yaml`,裸 `dvc repro` 不再触发。

- [ ] **Step 2: 改 `CLAUDE.md` Known Gotchas 的 RG 耦合技术债条目**

把 `- **RG coupling tech debt(3 处,待下一 PR 清理)**：...` 整条替换为已解决的表述:
```markdown
- **RG coupling 已收束(见 history/entries/015)**：① fusion 构造改为 `FusionModule.from_contract` 钩子,RG 覆写自行加载 `EndpointBaselineStats`+派生 `id_to_endpoint_key`,`train_baseline_v0.py` 不再按 `_target_` 字符串分支;② `EndpointBaselineStats` fit/save 与 `endpoint_id` 列派生改由 `ContractConfig.fit_endpoint_baseline_stats` 开关门控(v1.yaml=false / v1_expanded_pool.yaml=true),不再无条件 fit;③ 3 个 RG 专属 stage 隔离到 `dvc_reliability_gate/dvc.yaml`,裸 `dvc repro` 不再触发。`scripts/analyze_gate_weights.py`(一次性分析脚本)维持直接构造 RG 的用法,不在收束范围内。
```

- [ ] **Step 3: 创建 `history/entries/015-rg-coupling-containment.md` 占位**

按 `history/entries/` 既有格式写一篇 entry。内容以本 spec/plan 为骨架,记录:动机(3 处耦合技术债)、方案(from_contract 钩子 / fit_endpoint_baseline_stats 开关 / dvc 子目录隔离)、关键决策(DVC 文件名约束导致子目录+wdir、commit 边界为保 pytest 绿把 config 值并入 commit 2)、遗留(issue #16 eval 类别失衡另行处理)。若实现阶段发现新坑,补进本 entry。

- [ ] **Step 4: 更新 `history/index.md`**

在 entry 列表加 015 一行,并在影响域倒排索引里把 015 挂到相关影响域(fusion 构造 / contract build / dvc pipeline)。

- [ ] **Step 5: 全量测试 + 提交前最终校验**

Run: `conda run -n interface python -m pytest tests/ -q`
Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md history/entries/015-rg-coupling-containment.md history/index.md
git commit -m "[Docs]: 同步 RG 耦合收束至 CLAUDE.md 与 history/015

Commands 小节 RG 触发改为 dvc repro dvc_reliability_gate/dvc.yaml;
Known Gotchas 的 RG coupling tech debt 条目更新为已收束;新增 entry 015。"
```

---

## 最终端到端回归(全部 commit 完成后,人工验证,非自动化测试)

确认重构未改变任何模型的实际行为——这是"收束不改算法"的核心验收。

- [ ] **L0/L1/L2 回归**:seed=42 重跑三条基线,与现有 `artifacts/{l0,l1,l2}/scores.parquet` 对比 scores 数值一致(合理浮点误差内)。
  ```bash
  conda run -n interface dvc repro build_contract_v1 train_v1 eval_v1
  ```
  (L1/L2 若不在默认 DAG,按 CLAUDE.md 现有命令单独跑;逐一与对应 artifacts 对比。)

- [ ] **RG 回归**:seed=42 重跑 RG,与 PR#14 记录的 softmax 配置数字(AUROC≈0.6024)对比一致。
  ```bash
  conda run -n interface dvc repro dvc_reliability_gate/dvc.yaml
  conda run -n interface dvc metrics show artifacts/baseline_v1_reliability_gate/metrics.json
  ```

- [ ] **裸 `dvc repro` 不再触发 RG stage**:确认无 `train_v1_reliability_gate` 等 RG stage 出现在待跑列表。
  ```bash
  conda run -n interface dvc status 2>&1 | grep -i reliability_gate
  ```
  Expected: 无输出(RG stage 已不在默认 pipeline)。

若任一回归数值漂移超出浮点误差,视为重构引入了行为改变,须定位根因(优先怀疑 `from_contract` 构造路径或 `endpoint_id` 传参与旧 `is_reliability_gate` 分支不等价),不得直接接受新数字。

---

## Self-Review

**Spec 覆盖**:spec §2.1(耦合点1)→ Task 1;§2.2(耦合点2)→ Task 2;§2.3(耦合点3)→ Task 3;§4 测试策略 → 各 Task 的 TDD 步骤 + 末尾端到端回归;§5 out-of-scope(`analyze_gate_weights.py`、issue #16)→ 未列入任何 Task,并在 Task 4 Known Gotchas 文案中显式声明;§6 四 commit → Task 1-4,偏差(config 值移入 commit 2)已在开头"Commit 边界说明"记录并说明必要性。

**占位符扫描**:无 TBD/TODO;所有代码步骤含完整代码块;所有命令含 expected output。Task 4 Step 3 的 history entry 内容为"占位待补"——这是 spec §6 明确的 commit 4 约定(内容落地后补),非计划缺陷。

**类型/签名一致性**:`from_contract(cls, cfg, *, contract_dir, modality_dims)` 在 base(默认)、RG(覆写)、train 调用点(`hydra.utils.get_class(...).from_contract(...)`)三处签名一致;`forward(self, modality_dict, endpoint_id=None)` 在 base(抽象)、L0/L1/L2(实现)、RG(既有)四处一致;`_write_v1` 新增参数 `fit_endpoint_baseline_stats: bool` 在定义(Step 6b)与调用点(Step 6a)一致;`ContractConfig.fit_endpoint_baseline_stats` 字段名在 dataclass、loader、config yaml、测试四处一致。
