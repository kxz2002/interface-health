# RG Coupling Containment — Design Spec

**Date:** 2026-07-21
**Branch:** `bugfix/rg-coupling-containment`
**Status:** Design approved, pending spec review before implementation-plan.

## 1. 背景与动机

PR#14（Reliability Gate Fusion 实验，commit `f6c0423`）在打通 RG 这条实验链路时，为了走通端到端，在三处**共享/通用代码路径**里插入了 RG 专属的耦合逻辑。这些耦合点让本该与 fusion 机制无关的通用代码（训练脚本主循环、contract 构建脚本、默认 DVC DAG）都必须知道"RG 是什么、RG 需要什么"，违反了现有 L0/L1/L2/RG 平级可插拔的架构约定，并已被记录为 `CLAUDE.md` Known Gotchas 里的技术债待清理项。

本次改动的目标是**收束这 3 处耦合的影响半径**，把 RG 专属的知识收回 RG 自己的模块/配置里，不改动 RG 门控算法本身（softmax collapse 等既有问题不在本次范围内）。

## 2. 三处耦合点与收束方案

### 2.1 耦合点 1：`train_baseline_v0.py` 的字符串匹配分支

**现状**：`main()` 里 `is_reliability_gate = (cfg.fusion._target_ == "src.fusion.reliability_gate.ReliabilityGatedFusion")`，之后在构造 fusion、SVDD center 初始化、`_train`/`_infer` 调用共 4 处依据这个布尔值分支处理，其中构造分支里硬编码读取 `configs/contract/endpoint_to_service.yaml` 相对路径、加载 `EndpointBaselineStats`、派生 `id_to_endpoint_key`，再手工拼进 `fusion_kwargs` 传给 `hydra.utils.instantiate`。

**方案**：给 `FusionModule` 基类（`src/fusion/base.py`）新增 `from_contract` classmethod 钩子：

```python
@classmethod
def from_contract(cls, cfg, *, contract_dir, modality_dims):
    return hydra.utils.instantiate(cfg, modality_dims=modality_dims)
```

默认实现等价于今天的行为，L0/L1/L2 不需要覆写。`ReliabilityGatedFusion` 覆写该方法，把"从 `contract_dir` 读 `endpoint_baseline_stats.json`、派生 `id_to_endpoint_key`、拼 kwargs"这部分逻辑整段移进自己的覆写体内。

基类抽象 `forward` 签名统一新增 `endpoint_id: torch.Tensor | None = None`（默认值保证向后兼容）；L0/L1/L2 的 `forward()` 相应接受但忽略该参数。

`train_baseline_v0.py` 的构造点改为：

```python
fusion_cls = hydra.utils.get_class(cfg.fusion._target_)
fusion = fusion_cls.from_contract(cfg.fusion, contract_dir=contract_dir, modality_dims=modality_dims)
```

`_train`/`_infer` 统一无条件传 `endpoint_id=batch.get("endpoint_id")`（L0/L1/L2 忽略，RG 使用）。SVDD center 初始化分支同理消除——具体初始化差异（如果有）改为查询 `fusion` 实例本身是否需要特殊初始化路径，而不是查询"是不是 RG"。

**结果**：`is_reliability_gate` 这个变量名及其所有分支从 `train_baseline_v0.py` 中完全消失；新增 fusion 类型今后想要感知 contract 侧信息，只需覆写 `from_contract`，无需再修改训练脚本。

### 2.2 耦合点 2：`build_contract.py` 无条件派生/落盘

**现状**：`main()` 里 `endpoint_id_map = _derive_endpoint_id_map(ep_to_svc) if cfg.contract_version == "v1" else None`——用契约版本号（v0/v1）作为"要不要产出 RG 需要的东西"的代理判断，语义不准确（v1 不等于"需要 RG 统计量"，只是历史上 v1 恰好是唯一用到 RG 的版本）。`_write_v1()` 内部（约 543-562 行）无条件计算、fit、保存 `EndpointBaselineStats`，不管调用方（L0/L1/L2 契约）是否会消费它。

**方案**：给 `ContractConfig`（`src/contracts/contract_config.py`）新增字段 `fit_endpoint_baseline_stats: bool = False`，完全照抄现有 `expand_train_pool` 的模式（dataclass 字段 + `load_contract_config()` 里 `raw.get("fit_endpoint_baseline_stats", False)`）。

`build_contract.py` 的两处改动：
- `endpoint_id_map` 的派生条件从 `cfg.contract_version == "v1"` 改为 `cfg.fit_endpoint_baseline_stats`。
- `_write_v1()` 里 `EndpointBaselineStats` 的 fit/save 整块（含退化列告警日志）用 `if cfg.fit_endpoint_baseline_stats:` 包裹。`_write_v1()` 当前签名只接收 `expand_train_pool: bool`，需要扩展为接收 `fit_endpoint_baseline_stats: bool`（或直接传 `cfg`，实现时二选一，优先选参数更少、职责更清晰的一种）。

配置取值：`configs/contract/v1.yaml` 设 `fit_endpoint_baseline_stats: false`（默认路径不再产出这份统计量和 `endpoint_id` 列）；`configs/contract/v1_expanded_pool.yaml` 设 `fit_endpoint_baseline_stats: true`（RG 专属路径维持现状）。

**必须保留的不变量**（已有测试锁定，见 `tests/test_build_contract_v1_endpoint_id.py::test_endpoint_id_not_added_in_v0`）：v0 契约永远不产出 `endpoint_id` 列，也不产出 `endpoint_baseline_stats.json`——v0.yaml 不设置该字段，走 dataclass 默认值 `False`，天然满足。

### 2.3 耦合点 3：`dvc.yaml` 默认 DAG 里的 RG 专属 stage

**现状**：`build_contract_v1_expanded`/`train_v1_reliability_gate`/`eval_v1_reliability_gate` 三个 RG 专属 stage 定义在根 `dvc.yaml` 里，与 L0/L1/L2 的默认 stage 混在同一个文件、同一条默认 DAG 中，裸 `dvc repro`（不指定 target）会把这三个 stage 也一起触发，造成约 20 分钟不必要的 rebuild。另外 `build_contract_v1` 的 `outs` 里包含 `artifacts/contract_v1/endpoint_baseline_stats.json`——这个文件今后（耦合点 2 修复后）默认不再产出，这一行本身就是过时声明。

**方案**：把三个 RG 专属 stage 整段移到新文件 `dvc_reliability_gate/dvc.yaml`（**必须是子目录下的 `dvc.yaml`**——实测 DVC 3.67.1 在 repo 任意位置都强制要求 pipeline 文件字面量命名为 `dvc.yaml` 或 `*.dvc` 后缀，`dvc_reliability_gate.yaml` 这种命名会被直接拒绝：`ERROR: bad DVC file name`）。

每个迁移的 stage 增加 `wdir: ..`，使其 `cmd`/`outs` 里现有的路径字符串（相对 repo root 书写）在新位置下继续按原样解析，不需要改写任何路径。

触发方式从原来的 `dvc repro build_contract_v1_expanded train_v1_reliability_gate eval_v1_reliability_gate` 变为 `dvc repro dvc_reliability_gate/dvc.yaml`（作为完整 DAG 一次性跑三个 stage；如需单跑某个 stage 用 `dvc_reliability_gate/dvc.yaml:stage_name` 语法）。

根 `dvc.yaml` 中 `build_contract_v1` 的 `outs` 删除 `artifacts/contract_v1/endpoint_baseline_stats.json` 这一行。

**结果**：裸 `dvc repro` 不再触碰 RG 相关 stage；`dvc metrics show` 的输出范围需要确认是否受影响（DVC 默认会遍历所有找到的 `dvc.yaml`，需在实现阶段验证子目录 stage 的 metrics 仍能被发现，如不能则在文档里注明改用 `dvc metrics show -R` 或显式路径）。

## 3. 组件依赖关系图

```
train_baseline_v0.py
  └─ 依赖 FusionModule.from_contract()（新钩子，耦合点1）
       └─ RG 覆写版依赖 build_contract.py 产出的
          endpoint_baseline_stats.json / endpoint_id 列
             └─ 是否产出由 ContractConfig.fit_endpoint_baseline_stats 控制（耦合点2）
                  └─ 该字段的取值由 v1.yaml / v1_expanded_pool.yaml 声明
                       └─ 两份 contract 分别被 dvc.yaml 主 DAG / dvc_reliability_gate/dvc.yaml 消费（耦合点3）
```

三处修复是单向依赖链，不是三个独立并列改动——耦合点1的 `from_contract` 钩子设计必须先确定"RG 需要从 contract_dir 拿什么"，才能反过来确认耦合点2的开关要控制哪些产出；耦合点3只是把已经通过开关分叉开的两条 contract 产出路径，在 DVC 层面也物理分开。实现顺序上按耦合点 1→2→3 推进是自然的，不需要并行开发再合并。

## 4. 测试策略

- **`from_contract` 单测**：至少覆盖 RG（验证覆写路径正确从 `contract_dir` 读取统计量并构造出等价于今天手工拼装的实例）和一个 baseline（如 L0，验证走的是默认实现、行为与改动前 `hydra.utils.instantiate` 完全一致）。
- **`ContractConfig`/`build_contract.py` 门控测试**：新增测试确认 `fit_endpoint_baseline_stats=False`（默认）时不产出 `endpoint_id` 列和 sidecar 文件，`=True` 时两者都产出——沿用 `test_build_contract_v1_endpoint_id.py` 已有的断言模式。
- **既有测试更新**（非新增，是维护现有覆盖不失真）：`tests/test_dvc_pipeline_v1.py` 里读取 `build_contract_v1_expanded`/`train_v1_reliability_gate` stage 定义的断言，改为从 `dvc_reliability_gate/dvc.yaml` 读取；`test_build_contract_v1_uses_non_expanded_config` 等仍读根 `dvc.yaml` 的断言不变。
- **回归验证（端到端，非自动化测试，人工跑一遍确认数字不漂移）**：seed=42 重跑 L0/L1/L2 和 RG，对比现有 `artifacts/{l0,l1,l2}/scores.parquet` 及 PR#14 记录的 RG 数字（AUROC≈0.6024，softmax 配置），确认重构后指标一致（在合理数值误差范围内）。这一步验证的是"重构没有意外改变任何模型的实际行为"，而不是测试新功能。

## 5. Out of Scope

- RG 门控算法本身（softmax gate collapse、`independent_sigmoid` 消融等既有问题）——本次改动只搬运/收束代码位置，不碰算法逻辑。
- `eval_all` 正负样本比例失衡问题——已拆分为独立 issue #16（`ready-for-agent`），与本次耦合收束无关，不在本分支处理。
- `scripts/analyze_gate_weights.py`——一次性分析脚本，明确不是训练/评估管线的一部分，直接构造 `ReliabilityGatedFusion` 绕过 Hydra 属于预期用法，保持不变。

## 6. 实现拆分（4 个 commit）

1. `[Refactor]`：`FusionModule.from_contract` 钩子 + 基类 `forward` 新增 `endpoint_id` 参数（L0/L1/L2 同步接受-忽略）+ `train_baseline_v0.py` 移除 `is_reliability_gate` 全部分支。
2. `[Refactor]`：`ContractConfig.fit_endpoint_baseline_stats` 字段 + `build_contract.py` 用该开关门控 `endpoint_id_map` 派生与 `EndpointBaselineStats` fit/save。
3. `[Refactor]`：`v1.yaml`/`v1_expanded_pool.yaml` 补齐开关取值 + `dvc_reliability_gate/dvc.yaml` 拆分（`wdir: ..`）+ 根 `dvc.yaml` 的 `build_contract_v1.outs` 清理 + `tests/test_dvc_pipeline_v1.py` 更新为读取新文件位置。
4. `[Docs]`：`CLAUDE.md` 的 Commands 小节（RG 命令改为 `dvc repro dvc_reliability_gate/dvc.yaml`）与 Known Gotchas 的"RG coupling tech debt（3 处，待下一 PR 清理）"条目同步为已解决；新增 `history/entries/015`（内容待实现落地后补，先占位/事后补写，不阻塞本 spec）。
