# 026 · content_length_mean 归一化尺度爆炸修复（对称 clip）+ 六融合方式三方对比

- **日期**: 2026-08-22
- **PR**: #25（branch: `feature/content-length-bodyhash-features`）
- **类型**: Bugfix + Experiment
- **影响域**: `scripts/build_contract.py`, `artifacts/baseline_new_merge_*/metrics.json`, `dvc_new_merge/dvc.lock`

> ⚠️ **数字失效声明**：本 entry 记录的三方对比数字（baseline/unfixed/fixed）建立在 025（service 级标签粒度修复）**之前**的标签口径上，与 master 合并后已过期。本次 merge 的 6 个 `metrics.json` 冲突已采用 master（PR #24，含 025 的标签修复）落地的版本，不含本 entry 描述的 content_length clip 改动。合并后需在新标签口径下重新跑一遍 `dvc repro dvc_new_merge/dvc.yaml` 才能确认 content_length clip 修复在新口径下的实际效果。

## 做了什么

`9853217`（entry 023 之后、本 PR 前一个 commit）给 `ApiPreprocessor` 新增了 `content_length_mean`/`content_length_rel_shift`/`body_hash_mismatch_rate` 三个派生特征，修复 PATCH 类故障对现有 RED 特征全隐形的问题（entry 017）。重跑 `dvc_new_merge/dvc.yaml` 后 PATCH 类 AUROC 如期提升，但 REPLACE 类在 concat(L0)/independent_concat(L1) 上大幅退步——排查发现 `endpoint_red__client_content_length_mean` 在 `POST:/api/v1/orderservice/order/refresh` 这一 endpoint 上归一化后出现远超其他特征量级的极端值（`|value|>10` 占比 0.80，全特征集里其余列均 `<0.01`），在直接消费 `Normalizer` 输出、没有内建尺度保护的融合方式（L0/L1）下主导 SVDD 距离、掩盖其他模态贡献。

根因不是简单的长尾分布（`log1p` 变换实测反而更差，已排除）。真实原因是**fit 窗口与 eval 分布的电平错位**：`order/refresh` 唯一的 Normal case（采集时间早于故障批次）落在 content_length_mean ≈7.1-7.6M 的电平，而几乎所有故障注入 case 在非故障期都落在另一个更高的电平 ≈9.84M——这是采集时序上的系统性漂移，与任何真实故障都无关。`Normalizer` 只在 Normal case 上 fit `[lo, hi]`，eval 时的值大幅超出这个范围导致 min-max 归一化结果远超正常量级。第二个独立叠加的因素：8 个 v0 endpoint 里有 6 个在该列上 Normal-fit 零方差（"退化" group），命中 `Normalizer.transform()` 已有的跳过归一化逻辑（entry 013），原样保留原始字节数量纲（如 61~459 字节）——这部分与漂移问题无关，且退化 group 的裁剪会把这些 endpoint 本就互不可比的真实取值压扁成同一常数，反而抹掉尚存的可区分信息，因此本次修复明确不动它。

修复方式（用户经 `AskUserQuestion` 选定"对归一化输出做对称 clip"）：在 `scripts/build_contract.py` 的 `main()` 里，紧跟在已有的 `RATE_COLUMNS` clip(0,1) 之后、`validate_contract_df()` 之前，新增一段只对 `endpoint_red__client_content_length_mean` 生效的对称裁剪，边界 ±5.0，且用 `normalizer.skipped_groups()` 排除退化 group（这些 group 走 raw passthrough，不受此次裁剪影响）。`content_length_rel_shift`（`frac|>10|=0`）与 `latency_divergence`（`frac|>10|=0.0062`）经检查均未达到需要裁剪的程度，未纳入本次改动范围。裁剪边界 ±5 的选取：`travel/trips-left`（另一个非退化 endpoint）在真实 HTTPABORT 故障注入期的归一化取值实测最大幅度 ≈3.2，5 留足安全余量不误伤真实信号；`order/refresh` 本身几乎全部落在边界外，裁剪对它是预期中的按下限幅（问题的本质是电平错位，裁剪只能封顶损害，不能恢复干净信号）。

全量 `pytest tests/`（312 passed, 2 skipped）确认无回归。重跑 `dvc repro dvc_new_merge/dvc.yaml`（`scripts/build_contract.py` 是 6 个 train stage 的公共依赖，触发全部级联重建），产出三方对比（baseline=引入三特征前 / unfixed=引入三特征但未裁剪 / fixed=本次裁剪后），overall + 四种主要故障类型（ABORT/DELAY/PATCH/REPLACE）macro-AUROC：

| 融合方式 | 阶段 | overall | ABORT | DELAY | PATCH | REPLACE |
|---|---|---|---|---|---|---|
| concat (L0) | baseline | 0.7416 | 0.999 | 0.965 | 0.621 | 1.000 |
| | unfixed | 0.5626 | 0.805 | 0.861 | 0.409 | 0.608 |
| | **fixed** | **0.7219** | 0.992 | 0.959 | **0.822** | **0.989** |
| independent_concat (L1) | baseline | 0.6941 | 0.994 | 0.933 | 0.573 | 0.995 |
| | unfixed | 0.6640 | 0.933 | 0.946 | 0.688 | 0.847 |
| | fixed | 0.5662 | 0.860 | 0.908 | 0.248 | 0.461 |
| gated (L2) | baseline | 0.7118 | 0.999 | 0.958 | 0.513 | 0.999 |
| | unfixed | 0.7765 | 0.999 | 0.945 | 0.931 | 0.998 |
| | fixed | 0.7387 | 0.925 | 0.951 | 0.888 | 0.993 |
| reliability_gate (RG, softmax) | baseline | 0.6510 | 0.900 | 0.886 | 0.470 | 0.798 |
| | unfixed | 0.7125 | 0.949 | 0.914 | 0.730 | 0.978 |
| | fixed | 0.6735 | 0.940 | 0.435 | 0.479 | 0.550 |
| reliability_gate (indep_sigmoid) | baseline | 0.6778 | 0.932 | 0.563 | 0.486 | 0.592 |
| | unfixed | 0.5826 | 0.961 | 0.504 | 0.564 | 0.571 |
| | fixed | 0.5397 | 0.762 | 0.650 | 0.598 | 0.639 |
| deviation_weighted (DWF) | baseline | 0.7043 | 0.999 | 0.918 | 0.506 | 1.000 |
| | unfixed | 0.7450 | 0.996 | 0.931 | 0.932 | 1.000 |
| | **fixed** | **0.7178** | 0.998 | 0.920 | **0.937** | **1.000** |

（以上均为 seed=42 单种子，与本次修复要解决的问题本身同一种子对比；未按 entry 014/023 的方法论补 seed{1,2,3}，见"遗留 TODO"。）

## 关键决策（不在 commit 里）

- **裁剪范围只限 `content_length_mean` 一列，不做通用"极端值裁剪"框架**：`content_length_rel_shift`/`latency_divergence` 实测未达到需要裁剪的程度，CLAUDE.md 明确反对"为假设的未来场景设计"，本次只解决已实测确认的问题列，不引入配置化的裁剪框架或阈值参数。
- **裁剪逻辑放在 `build_contract.py` 而不是 `Normalizer` 内部**：用户在 `AskUserQuestion` 里明确选择"不碰 `Normalizer` 的 fit 逻辑"，理由是 fit-eval 电平错位是数据采集时序问题、不是归一化算法本身的缺陷，裁剪是应对"归一化后特征异常值"的下游防护措施，语义上和已有的 `RATE_COLUMNS clip(0,1)` 同属"contract 层面的边界约束"，放在同一处比塞进 `Normalizer` 更符合职责划分。
- **明确排除退化 group**：`skipped_groups()` 已经是脚本里现成的 API（原本只用于告警文案），复用它做裁剪的排除判断，而不是重新推导一套"哪些 group 是 raw passthrough"的逻辑。这保证了 entry 013 的既有设计意图（退化 group 保留原始量纲）不被本次改动破坏。
- **接受 L1/RG 的意外回退，不做进一步归因或缩小修复范围**：见下方"坑 / 已知问题"。用户在看到三方对比后明确选择"接受现状，直接收尾提交"，不追加多 seed 复测，也不把裁剪逻辑改成按 fusion 方式条件生效（后者即使做，contract 产物本身是 6 种融合方式共享的单一 parquet，无法在 contract 层面做到真正隔离，需要更大改动才能实现，性价比不高）。

## 坑 / 已知问题

- **修复对 concat(L0)/DWF 达到目的，但对 independent_concat(L1)/reliability_gate(RG) 产生了看起来与 content_length 无直接语义关系的回退**：L1 修复后 overall 0.566，比 baseline（0.694）和 unfixed（0.664）都差，PATCH/REPLACE 全面下滑（0.688→0.248，0.847→0.461）；RG(softmax) 修复后 DELAY 从 unfixed 的 0.914 崩到 0.435——DELAY 是纯延迟型故障，与 content_length/body_hash 没有已知语义关联。合理的解释框架（未验证，仅归因假设）：One-Class SVDD 是跨全部 endpoint 共享的单一超球面嵌入，裁剪只改变了 `order/refresh` 一个 endpoint 一列特征的训练期分布，但这个改动通过共享 encoder 的梯度反传，可能重新分配了整个嵌入空间对各 endpoint/各故障类型的判别能力分布——这与既有决策记录"融合梯度学不到特征相关性"（RG/DWF 权重公式非 label-driven，SVDD loss 从不接触异常标签）是同一根问题的另一种表现：任何一处特征分布改变都可能在其他毫不相关的位置产生连带效应，SVDD loss 本身不知道"哪个特征在哪个 endpoint 上应该贡献多少判别力"。
- **本次三方对比只跑了单一 seed（42），不满足 entry 014/023 建立的多 seed 方差报告惯例**：无法区分"L1/RG 的回退是本次修复引入的系统性效应"还是"单 seed 随机性"（entry 023 已记录 L0/L1/RG 在 new_merge 上对种子选择本就敏感，RG std 高达 0.0352~0.0554）。这是本次收尾的已知缺口，不是被忽略，见下方遗留 TODO。
- **裁剪本身不能恢复 `order/refresh` 的干净信号，只能封顶损害**：在 clip 边界 {1,2,3,4,5,10} 的验证扫描中，该 endpoint 的"裁剪触发比例"在整个范围内都停留在 0.65~0.95，说明问题的本质（电平错位）不是幅度问题，任何裁剪阈值都不会让这一列变得"干净"，只是把已知损坏的信号限制在一个不会压垮 SVDD 距离计算的范围内。

## 遗留 TODO

- **补 seed{1,2,3} 复测 independent_concat/reliability_gate(两个变体)，确认回退是否为系统性效应**：用户已明确决定本次不做（见"关键决策"），但如果后续要在 L1/RG 上下任何结论或把它们用作对比基线，必须先补这组多 seed 实验——当前的 fixed 数字不满足 entry 014/023 建立的报告惯例，不能直接引用做横向比较。
- **`order/refresh` 的 fit-eval 电平错位问题本身未解决，只是被裁剪压制了下游影响**：真正的修复需要重采一个采集时间与故障批次同期的 Normal case，或者在 contract 层引入除 min-max 之外的、对分布漂移更鲁棒的归一化方式（如按 case 内部相对值而非跨 case 绝对值），这两条路径本次均未评估可行性，留给未来重新审视 `content_length_mean` 特征设计时处理。
