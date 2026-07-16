# L1/L2 融合模块：独立编码器 + 门控条件融合消融

- **日期**: 2026-07-15
- **分支**: `exp/fusion-l1-l2-gated-ablation`
- **类型**: Experiment（消融）
- **前置依赖**: entry 010（门控机制 novelty-check 结论）、entry 011（Hydra 可插拔 fusion/model 基础设施、Contract v1 时序切分）

## 背景与定位

`docs/NOVELTY_DOSSIER.md` 记录的门控条件融合方案（`g=sigmoid(W[e_ep;e_svc])`, `z=e_ep+g⊙(W_v·e_svc)`）在 entry 010 的 novelty-check 中被判定为不能单独构成论文创新点（与 FiLM/GS-Fuse 属于同一机制类别）。

**本轮定位的澄清**：门控机制本身仍然值得实现——不是作为论文最终要主张的创新点，而是作为后续提出真正的新融合模型时的**强 baseline / 组件候选**。是否值得继续投入取决于它相对 L0（现有 `EarlyConcatFusion`）能带来多大实际收益；收益越大，越说明"内容自适应地调制被广播复制的 service 级特征"这个方向本身有增量价值，为后续新模型设计提供依据。

本轮只做 **L1**（独立编码器，无门控）和 **L2**（完整门控）。entry 010 记录的 confounding variable 问题（L0→L1 混淆了"是否有门控"和"是否有非线性容量"两个变量）要求 L1 必须存在才能把 L2 的门控贡献从"任意非线性变换的贡献"中分离出来。L3（门控退化为无内容自适应的全局标量）本轮不做，留到看到 L1/L2 结果后再决定是否需要。

## 架构设计

两个模态分支，各自过一个独立的单层 `Linear(bias=False) + ReLU` 编码器，输出维度相同（`branch_dim`，新增超参数，默认 16）：

- `e_ep = ReLU(Linear(endpoint_red, branch_dim, bias=False))`，输入 10 维（trace+api 融合特征）
- `e_svc = ReLU(Linear(concat(service_metric, service_log), branch_dim, bias=False))`，输入 8 维（metric 5 维 + log 3 维）

`bias=False` 与现有 `DeepSVDD.encoder` 的约定一致（避免 bias 吸收偏移导致超球退化，参见 `src/models/deep_svdd.py`）。单层结构是有意选择：分支编码器只做维度对齐和非线性激活，不与下游 `DeepSVDD` 自带的三层 encoder 叠加过深。

**L1 — `IndependentConcatFusion`**：
```
z = concat(e_ep, e_svc)
output_dim = 2 * branch_dim
```

**L2 — `GatedFusion`**：
```
g = sigmoid(Linear(concat(e_ep, e_svc), branch_dim, bias=True))   # W_g
z = e_ep + g ⊙ Linear(e_svc, branch_dim, bias=True)               # W_v
output_dim = branch_dim
```
`W_g`/`W_v` 的 bias 保留默认（`True`）——它们是门控/调制层，不是表征编码器，不适用"避免超球退化"的约束。

L1 与 L2 之间**不做参数量对齐**：`W_g`/`W_v` 是门控架构本身的组成部分，不是无关的容量差异（混淆变量特指"和架构设计无关的额外非线性容量"，L0→L1 才是这种情况）。

**与下游衔接**：`z` 照常送入现有 `DeepSVDD` 的三层 encoder（`input_dim=fusion.output_dim → hidden_dim → rep_dim`），不修改 `DeepSVDD` 或 `train_baseline_v0.py` 的训练循环——完全复用 entry 011 的 Hydra `hydra.utils.instantiate(cfg.fusion, modality_dims=...)` 机制，只新增两个 `FusionModule` 子类和对应 config。

## 文件改动

- `src/fusion/independent_concat.py`：`IndependentConcatFusion(FusionModule)`
- `src/fusion/gated.py`：`GatedFusion(FusionModule)`
- 两者构造签名均为 `__init__(self, modality_dims: dict[str, int], branch_dim: int = 16)`，与 `EarlyConcatFusion` 一样在 `__init__` 里校验 `modality_dims` 的 key 集合等于 `MODALITY_ORDER`（缺失/多余均 `raise ValueError`）
- `configs/fusion/independent_concat.yaml`、`configs/fusion/gated.yaml`：`_target_` 指向上述类，`branch_dim: 16`
- `tests/test_independent_concat_fusion.py`、`tests/test_gated_fusion.py`：覆盖模式照抄 `tests/test_early_concat_fusion.py`（output_dim 正确性、forward shape、确定性、缺失/多余 modality key 报错），额外为 `GatedFusion` 加一组手算数值的门控公式正确性测试（构造一组已知权重，验证 `g`、`z` 的具体数值）

## 不做的事（本轮范围排除）

- 不做 L3（门控退化为全局标量）——留到看 L1/L2 结果后再决定
- 不修改 `DeepSVDD`、不引入"轻量 SVDD"变体——`z` 统一走现有三层 encoder
- 不做 L1/L2 参数量对齐（理由见架构设计一节）
- **不接入 `dvc.yaml`**：本轮手工用 CLI 跑 `train_baseline_v0.py fusion=independent_concat/gated`，产物落在 `artifacts/l1/`、`artifacts/l2/`（不进 DVC pipeline，不影响现有 v0/v1 stage 图）
- 不修改 `train_baseline_v0.py` 训练/推理循环本身

## 验证方式

分别对 Contract v1 跑 L1、L2，与已有 v1 L0 结果（AUROC=0.617 / AUPRC=0.327）对比：

```bash
python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1 out=artifacts/l1/scores.parquet fusion=independent_concat
python scripts/eval_baseline_v0.py --scores artifacts/l1/scores.parquet --out artifacts/l1/metrics.json

python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1 out=artifacts/l2/scores.parquet fusion=gated
python scripts/eval_baseline_v0.py --scores artifacts/l2/scores.parquet --out artifacts/l2/metrics.json
```

`pytest tests/` 全量跑通作为单元测试门槛。三组 AUROC/AUPRC 数字（L0/L1/L2）记入新的 history entry，作为门控收益判断的依据。

## 遗留 TODO（不在本轮做，供后续参考）

- L3（全局标量门控）视 L1/L2 结果决定是否需要
- 若门控收益显著，需要回头讨论"真正的新融合模型"该在此基础上做什么增量设计（这是本轮之后的下一个问题，不在本 spec 范围内）
- `solve_hidden_dim_for_param_budget` 目前仍是孤立工具，本轮未使用（无对齐需求）
