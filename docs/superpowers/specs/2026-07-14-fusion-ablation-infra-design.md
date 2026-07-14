# Fusion Ablation Infrastructure — Design

- **日期**: 2026-07-14
- **分支**: `feature/fusion-ablation-infra`
- **前置背景**: `history/entries/010-gated-fusion-novelty-check.md` — 门控融合机制本身不构成独立创新点，本轮工作定位为"补齐消融实验基础设施"，不实现门控融合本身。

## 背景与目标

计划中的 L0-L3 融合机制消融实验（裸拼接 / 独立编码器无门控 / 门控 / 全局标量门控）需要三块目前缺失的基础设施支撑，否则实验结果站不住：

1. **融合与模型机制目前硬编码**：`scripts/train_baseline_v0.py` 直接 `import EarlyConcatFusion` 并实例化，切换融合机制等于改代码。`CLAUDE.md` 声称模型走 `hydra.utils.instantiate(cfg.model)`，但 `configs/model/` 目录不存在、`configs/base.yaml` 的 `model: ???` 从未填充——文档与代码脱节。
2. **train/eval 存在数据重叠**：现状 `build_contract.py` 产出 `train.parquet`（全部 Normal case 行）和 `eval_all.parquet`（`= full`，即全部行），后者是前者的超集——`eval_all` 里用作负样本的 Normal 行，与训练时见过的行**逐行相同**。这违反了 One-Class/异常检测评估的基本要求（测试集里的正常样本必须是训练时未见过的），会导致当前 baseline 指标虚高。同时完全没有 held-out validation split 用于早停/模型选择。
3. **无参数量对齐能力**：L0（裸拼接零参数）→ L1（独立 encoder + 拼接）→ L2（+ 门控）→ L3（门控简化）的性能对比中，L0→L1 的提升混淆了"是否有门控式设计"和"是否有任何非线性容量"两个变量，无法把门控本身的贡献隔离出来。

**本轮目标**：补齐上述三块基础设施；**不实现**门控融合本身、不跑 L0-L3 实际对比实验——那是下一轮工作。

## 架构与组件

### 组件 1：Hydra 可插拔 fusion + model

- 新增 `configs/fusion/concat.yaml`，包裹现有 `EarlyConcatFusion`，含 `_target_: src.fusion.early_concat.EarlyConcatFusion`。为未来的 `gated.yaml` 等预留 config-group 位置，本轮不新增其他融合实现。
- 新增 `configs/model/deep_svdd.yaml`，包裹现有 `DeepSVDD`，同时把 `configs/base.yaml` 的 `model: ???` 正式填上默认值——修复 CLAUDE.md 与代码脱节。
- `scripts/train_baseline_v0.py` 改为 `@hydra.main(config_path="../configs", config_name="base")` 装饰的入口，内部：
  ```python
  fusion = hydra.utils.instantiate(cfg.fusion, modality_dims=modality_dims)
  svdd = hydra.utils.instantiate(cfg.model, input_dim=fusion.output_dim)
  ```
  不再有任何硬编码的 `from src.fusion.early_concat import EarlyConcatFusion`。
- `MODALITY_ORDER` 常量从 `EarlyConcatFusion` 类属性（`src/fusion/early_concat.py`）移到 `src/fusion/base.py`（与 `FusionModule` 抽象基类同一模块，融合机制间共享，不绑定任何具体子类）。
- `configs/base.yaml` 新增两个必填字段 `contract_dir: ???`、`out: ???`（沿用 `model: ???` 的"未填充则报错"约定），替代现有的 `--contract-dir`/`--out` argparse 参数。
- `dvc.yaml` 的 `train_v0` stage 命令改为 Hydra overrides 语法：
  ```
  python scripts/train_baseline_v0.py contract_dir=... out=... seed=42 training.epochs=50 fusion=concat model=deep_svdd
  ```
  本轮只保证单 stage 可通过 `fusion=xxx` 切换，不预先在 `dvc.yaml` 里搭 L0-L3 矩阵。

### 组件 2：Contract v1 — 三路 group-aware 切分

新增 `configs/contract/v1.yaml`（与现有 `v0.yaml` 并存，不替换）。`build_contract.py` 新增 v1 产出路径：

- 对 Normal case 的行，按 `service_name` 分两步调用 `GroupShuffleSplit`（均固定 `seed=42`）做三路切分，按**行数比例**目标 ~60/20/20：
  1. 第一次切分：从全部 Normal 行中切出 `eval_normal_holdout`（`test_size≈0.2`），剩余记为 `remainder`
  2. 第二次切分：从 `remainder` 中切出 `train_val`（`test_size≈0.25`，即占全量 ~20%），剩余为 `train_fit`（~60%）
  - `train_fit.parquet`：参与梯度下降
  - `train_val.parquet`：早停/模型选择用
  - `eval_normal_holdout.parquet`：**不参与任何训练环节**，专门作为最终评估的负样本来源
  - 8 个 service 行数分布不均（26~224 行），两次 `GroupShuffleSplit` 都是按 service 整体分配、不能拆散 service，实际比例会偏离 60/20/20（允许 ±15 个百分点的容差，测试按此容差校验，不要求精确命中）
- `eval_all.parquet` 的 Normal 部分从"全部 Normal 行"改为"仅 `eval_normal_holdout`"——修复训练/评估重叠问题。故障注入 case 的行不受影响（本来就不参与训练）。
- **Contract 版本升级到 `v1`**：按 CLAUDE.md"破坏字段口径必须升版"的规则，行集合划分方式变化属于接口契约的实质性变化。`configs/contract/v0.yaml`/`artifacts/contract_v0/`/`dvc.yaml` 里的 `build_contract`/`train_v0`/`eval_v0` stage **原样保留、保持可运行**；新增 `build_contract_v1`/`train_v1`/`eval_v1` stage，产出 `artifacts/contract_v1/`、`artifacts/baseline_v1/`。v0/v1 两条 pipeline 并存。
- **已知代价**：`history/entries/008` 记录的 baseline 指标基于 v0（train⊆eval_all），与 v1 的新指标不可比——这是修复重叠问题必须接受的成本，不是缺陷。
- Contract 层新增显式断言测试：`train_fit`/`train_val`/`eval_normal_holdout` 三者之间 `sample_id` 两两不重叠（`GroupShuffleSplit` 保证同一 service 的行不会跨切分出现，测试把这个保证锁定下来，防止未来数据源变化时静默失效）。

### 组件 3：参数量对齐工具

新增 `src/utils/param_budget.py`：

```python
def solve_hidden_dim_for_param_budget(
    build_fn: Callable[[int], nn.Module],
    target_params: int,
    lo: int = 1,
    hi: int = 1024,
) -> int:
    """二分查找 hidden_dim，使 build_fn(hidden_dim) 的参数量最接近 target_params。
    依赖参数量随 hidden_dim 单调递增这一前提（纯 Linear 堆叠结构满足）。
    """
```

纯函数，不依赖任何具体融合类，只依赖"给定 hidden_dim 能构造出模型"这一约定。本轮**不接入**任何训练脚本，也不用它构造 L0-L3 中的任何一个——它是下一轮消融实验用来把 L0/L1 的 baseline 参数量对齐到 L2/L3 的工具，本轮只提供这把"尺子"。

## 测试

- `tests/test_hydra_instantiate.py`：`configs/fusion/concat.yaml`、`configs/model/deep_svdd.yaml` 能正确 instantiate 出预期类型，`fusion.output_dim` 正确传递给 `svdd.input_dim`。
- `tests/test_contract_v1_split.py`：三路切分的 `sample_id` 两两不重叠；同一 `service_name` 的所有行落在同一个切分里（不会被 `GroupShuffleSplit` 拆散跨切分）；切分行数比例落在 60/20/20 ± 15 个百分点范围内（用 fixture 数据）。
- `tests/test_param_budget.py`：二分查找在若干 toy `build_fn` 上收敛到容差范围内；当 `target_params` 超出 `[build_fn(lo), build_fn(hi)]` 的参数量范围时，明确抛出异常而不是静默返回错误结果。

## 明确不在本轮范围内

- 不实现门控融合机制本身（`GatedFusion`/`ScalarGatedFusion` 等），门控相关的 `configs/fusion/gated.yaml` 等留空
- 不跑 L0-L3 四个变体的实际对比实验
- 不在 `dvc.yaml` 里预先搭建 L0-L3 的 stage 矩阵
- 不用参数量对齐工具去构造任何具体 baseline，只提供工具本身
