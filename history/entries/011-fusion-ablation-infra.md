# 011 · Fusion 消融基础设施：Hydra 可插拔化 + Contract v1 时序切分 + 参数量对齐工具

- **日期**: 2026-07-15
- **PR**: N/A（同分支多 commit，未开 PR）· **Commit**: bebd527..aaafdd2（7 个 commit）
- **类型**: Feature / Refactor
- **影响域**: `src/fusion/`, `src/contracts/`, `src/utils/`, `configs/`, `scripts/train_baseline_v0.py`, `scripts/build_contract.py`, `dvc.yaml`, `CLAUDE.md`, `tests/`

## 做了什么

补齐 L0-L3 融合机制消融实验所需的三块基础设施，均为工具/管线层，不涉及门控融合本身的实现（立项依据见 entry 010）：

1. **Hydra 可插拔 fusion/model**（Task 1-4）：`MODALITY_ORDER` 从 `EarlyConcatFusion` 类属性挪到 `src/fusion/base.py` 模块常量；新增 `configs/fusion/concat.yaml`、`configs/model/deep_svdd.yaml`；`configs/base.yaml` 填上 `model`/`fusion` 默认值 + `contract_dir`/`out` 必填字段；`scripts/train_baseline_v0.py` 从纯 argparse 整体重写为 `@hydra.main` 入口，融合/模型通过 `hydra.utils.instantiate(cfg.fusion/cfg.model)` 构造。
2. **Contract v1 三路时序切分**（Task 5）：新增 `src/contracts/split_v1.py::split_normal_rows_temporal` 纯函数 + `configs/contract/v1.yaml`，把每个 Normal case 按时间窗切成 `train_fit`/`train_val`/`eval_normal_holdout`（60/20/20），修复 v0 `train ⊆ eval_all` 的评估重叠问题。`build_contract.py` 按 `contract_version` 分支产出，v0 路径原样不动。
3. **参数量对齐工具**（Task 6）：`src/utils/param_budget.py::solve_hidden_dim_for_param_budget`，二分查找使模型参数量最接近目标的 `hidden_dim`。本轮只提供工具，未接入任何训练脚本或 baseline 构造。

全部通过 subagent-driven-development 流程实现：每个 Task 独立 implementer → spec compliance review → code quality review 两阶段验证后才进入下一 Task。全量测试 179 passed / 2 skipped（跑完 Task 6 后的最终状态）。

## 关键决策（不在 commit 里）

- **Contract v1 切分维度改为时间窗时序切分，而不是原 spec 的 `service_name` GroupShuffleSplit，也不是 entry 010 遗留 TODO 里主张的 leave-service-out**：这是有意识翻案，已与用户确认，理由分两层。
  - **为什么不用 leave-service-out**：本数据集 endpoint→service 接近 1:1（8 对 8），leave-service-out 会让 `eval_normal_holdout` 里的 endpoint 训练时完全没见过——对 per-endpoint One-Class 检测变成"对未见 endpoint 的泛化能力"测试，违背"对已监控 endpoint 检测异常"的研究设想，holdout 正常样本会仅因"没见过"被打高分，指标失真。
  - **时序切分为什么仍能回应 010 的泄漏担忧**：010 担忧的本质是"train 和 eval 出现逐行相同的 service 级广播特征行"。时序切分下 train 取早期窗、eval 取晚期窗，service 级广播特征随时间变化（不是常量），eval 的 holdout 行是训练时未见过的时间窗，FP 估计诚实。"同 service 下 endpoint 无法区分"在 One-Class（无监督分类目标）设定下不构成监督式泄漏，且这正是门控融合方法本身要解决的建模问题，不该靠切分回避。
  - 真实数据验证：两个 Normal case 各 108/156 时间窗，实测切分比例 63.4%/18.7%/17.9%，8 个 endpoint 三份全覆盖，每 case 内 fit<val<hold 时序分离成立。
- **v1 也写一份冗余 `train.parquet`（= `train_fit` 内容）**：目的是让 `train_v1` dvc stage 今天就能复用 Task 4 改完的同一份 `train_baseline_v0.py`，不需要为 v1 另写训练脚本。`train_val` 的早停消费明确留给下一轮，本轮训练脚本完全不读 `train_val.parquet`。
- **`training/default.yaml` 首次被真正消费，引入两处有意的行为副作用**：
  - `batch_size`：旧 argparse 默认 64 → 现在 config 默认 256（dvc override 只改 `training.epochs`，不改 `batch_size`）。
  - `weight_decay`：旧脚本手搓 `Adam(params, lr=lr)` 隐式 `weight_decay=0` → 现在 `hydra.utils.instantiate(cfg.training.optimizer, params=...)` 消费整份 optimizer config，`weight_decay` 变为 1e-4。
  - 两处都不是 bug，也**没有**加 override 去掩盖——这是迁移到统一 Hydra config 后的必然结果，写在这里是为了让后续看 v0/v1 历史指标对比的人知道超参数基线变了，不要误判为回归。
- **`training.scheduler`/`training.early_stopping` 划清边界，本轮不消费**：config 里已有这两块（lr 衰减、早停 patience），但训练脚本明确不读。留空是为下一轮接入 `train_val.parquet` 早停时用，本轮强行接入会引入未经验证的收敛行为变化。

## 坑 / 已知问题

- **v0 与 v1 的 metrics 不可直接比较**：v0 `train ⊆ eval_all`（评估集包含训练时见过的 Normal 行），v1 修复了这个重叠。真实数据实测：v0 AUROC=0.634 / AUPRC=0.303（14717 样本），v1 AUROC=0.617 / AUPRC=0.327（13632 样本，因 `eval_all` 只保留 holdout Normal + 全部异常行，样本数减少）。两者不代表同一协议下的两次跑，后续写论文/报告严禁把两者放在同一张表里当作直接对比，必须注明协议版本。
- **Normalizer 在全部 Normal 行上 fit，泄漏了 `eval_normal_holdout` 的统计量到训练特征尺度**：最终整体 review（非单个 Task 的两阶段 review）抓到——`build_contract.py` 原实现在 `contract_version == "v1"` 分支执行前，已经用`fit(full[normal_mask])`（全部 Normal 行，含日后被划入 `eval_normal_holdout` 的行）拟合 Normalizer 的 min-max 统计量，再拿这份统计量 transform 全部行。Task 5 的时序切分只在**行集合**层面隔离了 train/eval，但 holdout 行的数值已经通过归一化尺度**渗透进训练特征**——这正是 CLAUDE.md Known Gotchas 里"eval 统计泄漏到特征"同一类问题的另一处实例，说明"看起来做了切分"不代表切分覆盖了所有会泄漏的环节，需要顺着数据流全程检查每一步统计量的 fit 范围。修复方式：`contract_version == "v1"` 时，Normalizer 的 fit 范围收窄到 `train_fit`（先跑一次 `split_normal_rows_temporal` 取 `train_fit` 的 `sample_id` 集合，再用该子集 fit），v0 路径（fit 范围本就是"全部 Normal 行"）不受影响。修复前记录的 v1 数字（AUROC=0.566/AUPRC=0.289）是泄漏状态下的结果，已被上面的 0.617/0.327 取代，不应再被引用。此坑提醒：任何"在 Normal 上 fit 统计量"的组件（Normalizer、未来可能的 scaler/encoder 等），只要涉及 v1 的三路切分，都要重新审视 fit 范围是否收窄到 `train_fit`，不能想着"反正整体都是 Normal 就一起 fit"。
- **窗数不足退化路径只在极短 case / mini fixture 上触发**：`split_normal_rows_temporal` 对 <2 个时间窗的 case 会把全部数据塞进 `train_fit`，另外两份为空但保留 schema。真实数据的两个 Normal case（108/156 窗）远超阈值，不会触发；`tests/fixtures/mini_dataset.yaml` 的 mini Normal case 只有 3 个时间窗，`train_val` 会退化为空，e2e smoke 测试已覆盖并断言互斥性依然成立，不是 bug。
- **`configs/fusion/gated.yaml` 尚不存在**：`train_baseline_v0.py` 模块 docstring 里提到的 `fusion=gated` 是前瞻性示例（对应 spec 排除范围内的未来工作），今天跑会报 Hydra "找不到 config" 错误，不是本轮遗漏。
- **`solve_hidden_dim_for_param_budget` 首版测试有覆盖盲区，已在 review 循环中修复**：code quality review 第一轮发现 `test_converges_close_to_target` 用 `target=count(128)` 恰好是精确命中，从未真正走到"无精确解，取更近邻居"这条路径。已在 commit `aaafdd2` 中补充 `test_converges_close_to_target_when_no_exact_hit_exists`（target 落在 `count(64)=1994` 与 `count(65)=2025` 之间，验证正确返回更近的 64）。这是本轮唯一一次 review 打回重做的情况，记录下来是提醒：涉及边界值的二分查找类工具，写测试时要主动构造"卡在两个候选之间"的场景，不能只测精确命中。

## 遗留 TODO

- **`train_val.parquet` 早停消费**：v1 已产出这份数据，但训练脚本不读。下一轮把 `training.early_stopping`/`training.scheduler` 接入训练循环时需要用到。
- **门控融合本身的实现**：本轮全部是基础设施，L0-L3 四个融合变体、门控机制、消融对比实验均未开始，是这三块基础设施要服务的下一步工作（entry 010 已有 novelty-check 结论）。
- **`solve_hidden_dim_for_param_budget` 接入 baseline 构造**：目前是孤立工具，未被任何训练脚本/模型工厂调用。用于隔离"门控设计" vs "非线性容量"两个混淆变量时才会用到（对应 010 记录的 L0→L1 confounding variable 问题）。
- **`configs/fusion/gated.yaml` 等未来融合变体的 config**：目前只有 `concat.yaml` 一份，新增融合机制时需要照 `concat.yaml` 的模式补对应 config + 在 `hydra.utils.instantiate` 签名上对齐 `modality_dims`/`input_dim` 关键字参数约定。
