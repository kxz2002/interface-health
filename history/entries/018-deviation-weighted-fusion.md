# 018 · DeviationWeightedFusion：逐特征偏离量加权融合最小改动验证（不达标）

- **日期**: 2026-07-27
- **PR**: 待开 PR（branch: `feature/deviation-weighted-fusion`）· **Commit**: `fbd7451..782abae`（6 commit：1 `[Docs]` 设计文档 + 3 `[Feature]` 核心类/config/from_contract + 1 `[Bugfix]` 列长度校验补测 + 1 `[Feature]` DVC 接入）
- **类型**: Experiment
- **影响域**: `src/fusion/deviation_weighted.py`, `configs/fusion/deviation_weighted.yaml`, `dvc_deviation_weighted/`, `tests/test_fusion_deviation_weighted.py`, `tests/test_fusion_from_contract.py`, `tests/test_hydra_instantiate.py`, `tests/test_e2e_deviation_weighted_smoke.py`

## 做了什么

entry 017 的 C 实测把"service 全盲"精确化为**信号淹没**：ABORT/REPLACE 故障下单特征 `client_error_rate` 的 oracle AUROC 达 0.993/1.000，但朴素等权融合把这一维和其余噪声维度平均掉，AUROC 掉到 0.278~0.499。本轮验证"逐特征可信度加权能否救回这个 gap"：新增 `DeviationWeightedFusion`（`src/fusion/deviation_weighted.py`）——对 `endpoint_red`/`service_metric+service_log` 两分支的每一维特征分别计算相对该 endpoint 自身 normal 基线的 z-score，用 `sigmoid(|z|-2.0)` 逐特征加权后直接 concat，**零可学习参数**（无编码器、无分支级门控，与 `ReliabilityGatedFusion` 的架构差异是本设计的核心）。退化列（全局零方差/全 NaN）权重强制为 1，`endpoint_id=None` 时退化为纯 concat（等价 L0）。接入 `from_contract` 钩子 + `dvc_deviation_weighted/dvc.yaml`（复用 RG 已产出的 `artifacts/contract_v1_expanded/`，两个新 stage 跨文件依赖已验证可被 `dvc dag` 正确解析）。在真实数据（`n_samples=12683`）上训练评估，按 `by_anomaly_type` headline 指标对照 spec §5 三档标准判定为**不达标**。

## 关键决策（不在 commit 里）

- **不做分支级门控（RG 式 softmax 竞争）**：entry 014 已诊断 RG 的 softmax 门控在无监督 SVDD loss 下会坍缩到 `w_svc≈1.0`。本设计刻意排除任何分支级竞争机制，保证结果能干净归因到"逐特征加权本身有没有用"，不被叠加的坍缩风险混淆——这是与 L0 的单变量对照（唯一变量是"是否逐特征加权"），不是完整方法提案。
- **退化列权重固定为 1，而非跳过或用兜底 std 计算 sigmoid**：`EndpointBaselineStats` 对全局零方差列的 `std` 兜底为 `1e-9`，若不特殊处理，非零偏移会被除出 1e6~1e9 量级的虚假 `|z|`——这是被除零放大的假信号，不是"淹没"问题要救的真实信号。选择"原样通过"（权重=1）而非"跳过该列（输出0）"，因为没有额外证据支持"该列当前异常"，是最保守的处理。
- **用 `by_anomaly_type` 而非 `overall` 作 headline**：entry 017 已实测 `overall` 把跨天/跨 run 的 Normal case 与故障 case 混合当正负样本，AUROC 因"这是哪个 run"的噪声水平虚高 0.1~0.25（entry 012/014 报的历史 headline 数字正是被污染的 `overall`）。`by_anomaly_type` 天然是 within-case 比较，不受跨 run 污染，本设计不修复切分逻辑本身，只在报告侧换用这个已存在但历史上没被当 headline 用的干净指标。
- **DVC pipeline 复用 RG 的 contract build 阶段，不重新 build**：与 RG 用同一份 `contract_v1_expanded`（同一 eval_all 样本集、同一 `fault_baseline_train_fraction`）是能直接对比的硬性前提，`dvc_deviation_weighted/dvc.yaml` 只加 train/eval 两个 stage，跨文件依赖 `artifacts/contract_v1_expanded/*`——`dvc dag` 验证过能正确解析到 `dvc_reliability_gate/dvc.yaml:build_contract_v1_expanded`，未触发计划里预案的"合并进 RG 文件"退回方案。
- **Task 1 review 发现的列长度校验补丁**：`__init__` 增加 `len(red_cols)==ep_dim`/`len(svc_cols)==svc_dim` 的 fail-fast 校验（`[Bugfix]` commit `433f853`），以及一个显式断言"零可学习参数"的测试——两者都是代码质量审查中补的护栏，不是最初实现遗漏的功能。

## 达标判定结果：不达标

按 spec §5 三档标准，对照 `by_anomaly_type` 的 ABORT/REPLACE 宏平均 AUROC（真实数字，`artifacts/baseline_v1_deviation_weighted/metrics.json`，`n_samples=12683`，与 RG 同一 contract）：

| 指标 | DeviationWeightedFusion | RG（entry 016 既有参考，同 contract） | Oracle（entry 017） |
|---|---|---|---|
| ABORT 宏平均 | **0.6321**（assurance=0.5887 / order=0.7088 / travel=0.5841 / travel2=0.6467） | 0.5970（0.5937 / 0.8900 / 0.2355 / 0.6689） | ≈0.993 |
| REPLACE 宏平均 | **0.5379**（assurance=0.5385 / order=0.3325 / travel=0.8194 / travel2=0.4611） | 0.6686（0.7311 / 0.3149 / 0.8354 / 0.7930） | ≈1.000 |
| Overall AUROC（存档，非判据） | 0.5107 | 0.5789 | — |

- **未达到"达标"门槛**：ABORT/REPLACE 宏平均需 ≥0.8，实际 0.632/0.538，远低于门槛，也远离 oracle。
- **不构成"部分达标"的模式**：spec §5 的"部分改善"描述的是离 oracle 有距离但方向一致的单向提升（如 0.4→0.6）。本次逐 case 层面方向不一致——ABORT 的 travel（0.236→0.584，明显变好）与 order（0.890→0.709，明显变差）互相抵消；REPLACE 净效果是变差（宏平均 0.669→0.538，assurance 与 travel2 两个 case 明显退步）。这不是"融合层已经部分放大信号但打分层跟不上"的一致性信号，更像是逐特征加权在不同 case 上产生了不可预测的正负扰动。
- **判定为"不达标"**：与 RG 基线基本无系统性差异，REPLACE 甚至更差。按 spec §5/§8 与用户预先声明的硬性原则，**跳过 Task 6（诊断脚本，仅"部分达标"档触发）**，不做任何后续深化尝试（不加可学习阈值、不改 SVDD 打分公式、不加分支级门控）——这是本设计明确接受的可能结果（spec §7 风险1），不代表实现有 bug。
- **DELAY/PATCH（非判据，仅存档）**：DWF 在这两类上反而比 RG 好（DELAY 宏平均 0.6462 vs 0.5043；PATCH 宏平均 0.5480 vs 0.4281），没有被拖累，但不改变 ABORT/REPLACE 未达标的结论。

## 坑 / 已知问题

- **Task 3 首次派发的实现子任务中途异常终止**：执行过程中一次 subagent 调用在只完成只读探索（读 schema.json 格式）后无输出地结束，没有落地任何文件改动。通过 `git status` 确认无改动后，重新派发完成了实际实现——记录是因为这类"看起来完成但实际什么都没做"的中途失败需要显式核查 git 状态才能发现，不能只看 agent 返回的文字报告。
- **代码质量审查中一条 Important 建议被判定不适用并驳回**：审查者建议在 `from_contract` 里加一条"`schema.json` 派生的 `red_cols`/`svc_cols` 与 sidecar 内部存的列表长度一致性断言"，理由是两者是独立派生、可能悄悄错位。但这与 spec 的"背景说明"里已经写明的设计决定直接冲突——`EndpointBaselineStats._red_cols`/`_svc_cols` 是私有属性，本设计刻意不触碰它们，避免依赖另一模块的实现细节，`schema.json` 的 `feature_groups` 被明确定为列顺序的唯一权威来源。采纳该建议需要给 `EndpointBaselineStats` 新增一个公开访问器，属于扩大改动范围，因此保留原实现，只接受了另外两条不涉及架构决策的 Minor 建议（测试覆盖/fixture 复用），未采纳。
- **单特征 oracle AUROC 高不代表"给这一维加权"就能救回信号**：本次最直接的教训——`client_error_rate` 单独看 oracle AUROC 0.993/1.000，但把它相对基线的偏离量转成权重、再乘回原始特征值并直接 concat 喂给等权 L2 距离的 SVDD，逐 case 结果有涨有跌、REPLACE 净效果反而变差。可能的原因（未验证，留给后续深化时诊断）：加权只改变了这一维的*数值幅度*，SVDD 的等权 L2 距离仍会把放大后的这一维和其余维度的平方误差同等相加，没有改变"这一维对最终距离的贡献占比取决于其余 17 维噪声水平"这一结构性问题。

## 遗留 TODO

（spec §8 明确排除项，本轮不做，原样搬入，是否深化留给用户看到本 entry 后另行决定）

- 不做可学习阈值（`threshold` 固定为 2.0，未做成 `nn.Parameter`）。
- 不做逐维可学习加权网络（本设计整个类零可学习参数）。
- 不改造 Deep SVDD 打分公式（仍是等权 L2 距离，`((z-center)**2).sum(dim=1)`）。
- 不修复跨 run 污染的切分逻辑代码（`by_anomaly_type` 是报告侧最小修复，代码侧改造仍是独立工作项）。
- 不新增 `service_metric`/`service_log` 第三分支。
- 若用户决定继续深化：下一步候选（可学习阈值/逐维可学习权重/改造 SVDD 打分公式，尤其后者——"坑"一节记录的教训指向等权 L2 距离可能是比融合层更值得优先改的瓶颈）需要开新一轮独立设计，不在本 entry 或本 PR 范围内。
