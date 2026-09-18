# 027 · 时间混淆诊断：平凡基线打败全部融合机制，重采延期后转入待办

- **日期**: 2026-08-25
- **PR**: 待 Task 3 建 PR 后回填 · **Commit**: `56c4561`（分支 `exp/new-merge-rerun-pr24-pr25`）
- **类型**: Experiment + Docs
- **影响域**: `scripts/analyze_temporal_confounding.py`, `artifacts/baseline_new_merge_*`, `artifacts/temporal_confounding/`, 研究方向/论文 framing, 数据采集协议（`data/new_merge` 重采计划）

## 做了什么

两件事。**一是补上 entry 026 要求的重跑**：`dvc repro dvc_new_merge/dvc.yaml` 在 025（标签三档化）+ 026（content_length clip）两个修复同时在位的口径下重跑六种融合机制，确认 026 的方向性结论在正确标签下依然成立（L0/L2/DWF/RG-softmax 改善，L1 回退），026 的"数字失效声明"至此解除。同时发现 RG `independent_sigmoid` 变体**完全塌陷**（score 恒为常数 0.32、AUROC 恰好 0.5、loss 全程冻在 0.320000），这是 025 与 026 首次共同在位才暴露的集成问题。

**二是响应用户提出的"串行采集基线偏移"质疑，做了一轮诊断，结论比原假设严重得多**。用户的原假设是"27 个 case 串行采集导致基线偏移，模型可以靠学习『这是第几个 case』来偷懒"。该假设被证伪，但诊断过程翻出两个更致命的问题：一个不含任何观测数据的时间位置标量、以及一个零参数零训练的 z-score 基线，**双双打败全部六种融合机制**。诊断脚本落盘为 `scripts/analyze_temporal_confounding.py`（一次性诊断脚本，不进 dvc pipeline，同 `analyze_gate_weights.py`/`analyze_modality_observability.py` 惯例），报告产物 `artifacts/temporal_confounding/report.md`。因 AnoMod 数据集正在筹划论文发表、近期不便重采，全部修正工作转入下方遗留 TODO。

### 重跑结果（seed=42）

| 融合方式 | 上次提交(仅025) pooled | 本轮(025+026) pooled | 变化 | 本轮 per-case 宏平均 |
|---|---|---|---|---|
| concat (L0) | 0.9021 | 0.9095 | +0.0074 | 0.9140 |
| independent_concat (L1) | 0.8767 | 0.8352 | −0.0415 | 0.8333 |
| gated (L2) | 0.8863 | 0.9120 | +0.0257 | 0.9131 |
| reliability_gate softmax (RG) | 0.6015 | 0.7026 | +0.1011 | 0.7042 |
| reliability_gate indep_sigmoid | 0.6896 | **0.5000** | 完全塌陷 | 0.5000 |
| deviation_weighted (DWF) | 0.8870 | 0.9200 | +0.0330 | 0.9238 |

### 诊断结果（全部可由 `scripts/analyze_temporal_confounding.py` 复现）

```
rel_pos 平凡基线（只用"窗口在 case 内的相对时间位置"，不看任何特征）
    pooled AUROC   0.9340        per-case 宏平均  0.9656
零参数 z-score 基线（每个 (case,endpoint) 用自己 baseline 阶段算 z-score）
    L2 ||z||  0.9415     max|z|  0.9403     mean|z|  0.9306
────────────────────────────────────────────────────────────
最好的学习模型 DWF                      per-case 宏平均  0.9238
```

## 关键决策（不在 commit 里）

- **证伪"case 身份 shortcut"，而不是顺着用户的假设往下做**：pooled 与 per-case 宏平均的 gap 全部落在 ±0.004（L0 −0.0044、L2 −0.0011、DWF −0.0037）。per-case 口径的正负样本都取自同一 case，case 身份在该口径下**在数学上不可利用**，两个数字却几乎相同 ⇒ 判别力不来自跨 case 基线差异。文献里 pooled/宏平均背离是真实现象（[video AD 审计](https://arxiv.org/html/2608.11985) 实测出过排名反转），但在本数据集上没有发生。**先证伪再深挖，避免了在错误方向上做 DANN/gradient-reversal 这类昂贵改造**（那是原假设成立时才该做的事）。
- **诊断脚本必须落盘，不能只留在临时命令里**：entry 014/016/018/022 的数字作废的直接原因是不可重跑。本轮所有数字先手工算、再固化进脚本、再跑一遍验证逐位一致，才写进 entry。
- **`rel_pos` 基线衡量的是数据集缺陷，不是模型作弊**：模型输入的 21 个特征里**没有任何时间戳字段**，所以 Deep SVDD 在结构上无法使用 `rel_pos`。另跑探针从特征回归 `rel_pos`（GroupKFold by case）得 R²=0.1026、Spearman ρ=0.1628 ⇒ 特征也几乎不间接携带时间位置。**两条证据合起来才能把"平凡基线赢了"正确归因给数据集而非模型**，缺任何一条都可能误判成"模型在偷时间信息"。
- **纠正"0.93 是理论上限"这个提法**：用户问"重采能否突破 0.93 上限"。0.93 不是上限而是**地板**——它是"什么都不学"的成绩，我们现在在地板以下。随机化 inject 位置**不会让 0.92 上升**（特征值一个都不变，模型输出一个 bit 都不会变），只会让平凡基线从 0.9656 塌到 ~0.5。重采改变的是"这个数字有没有资格被拿出来比"，不是数字本身。
- **重采是必要条件但不是充分条件**：重采能消除 rel_pos shortcut、能让归一化统计量不再由单一 478 行 case 决定，但**不能解决 z-score 基线打败模型**——那个基线用 per-case 自适应参照系，对采集漂移天然免疫，重采对它只有好处。光重采会得到"干净数据集上依然输给十行代码的模型"。
- **RG indep_sigmoid 塌陷本轮不修**（用户决定）：根因已定位（见下），但修它要动 `EndpointBaselineStats` 的退化列处理，属独立改动，且 RG 已被 025 判定为高方差不可用机制，优先级低于时间混淆问题。
- **多模态融合方向降格为消融维度，不作为主 claim**：与 CLAUDE.md 既有的"次要方向"定位一致，本轮补上外部证据——[Too Many Cooks](https://nkcs.iops.ai/wp-content/uploads/2026/01/Too_Many_Cooks_Assessing_the_Need_for_Multi-Source_Data_in_Microservice_Failure_Diagnosis.pdf) 实测剔除 log 后 DiagFusion F1 **提升 10%**、剔除 metric 后 CloudRCA **提升 40%**，原因是不加区分的融合稀释异常信号（与 entry 017/024 的"信号淹没"同源）。本轮 L0 与门控不相上下（0.9140 vs 0.9131）正是这个规律的又一次体现。

## 坑 / 已知问题

- **采集脚本用固定时序模板，inject 起始位置跨 case 几乎恒定**：27 个 case 的 inject 阶段起始 `rel_pos` 均值 0.586、**标准差仅 0.015**；布局恒为 `baseline 0.000→0.571 / inject 0.580→0.930 / recover 0.938→1.000`。一条 `if rel_pos > 0.58: anomaly` 的规则即得 per-case 宏平均 0.9656。这是 [Wu & Keogh 的 run-to-failure bias](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/)（异常集中在序列末尾而非随机分布）在 **case 内**的版本。审稿人跑一次这个基线就能否掉整篇——**任何投稿前必须先解决这条**。
- **归一化统计量与 SVDD 球心全部由 478 行、单一 Normal case 决定，且它是第 0 个采集的**：`train_fit` 只有 `Normal_20260728T161044Z` 一个 case（478 行），而采集从 `20260728T161044` 持续到 `20260729T090100`，跨度 17 小时。`Normalizer`/`EndpointBaselineStats` 用开头一小时的统计量去衡量 17 小时后采集的 case，**分子里混进了系统漂移量**。这是用户所说"基线偏移"的真实形态，但作用方向不是让模型偷懒，而是给所有后采集 case 一个递增的虚假偏离量。
- **零参数 z-score 基线打败全部学习模型**：`L2 ||z||` 0.9415 > DWF 0.9238（+0.0177）。该基线用每个 `(case, endpoint)` **自己 baseline 阶段**的 mean/std 做参照，漂移自动抵消，无需 Normal case、无需全局归一化统计量、无需 SVDD 球心、无需训练。这正是 [Quo Vadis, Unsupervised TAD (ICML'24)](https://arxiv.org/html/2405.02678v1) 的 L2-norm 简单基线，也是 AIOps 的标准做法（StepWise、DCASE per-section 同思路）。**含义是：Deep SVDD + 全局归一化这套架构在本任务上没有挣到它的复杂度。**
- **单特征最高只有 0.71，但 z-score 范数能到 0.94**：`client_5xx_rate`/`client_error_rate` 各 0.7139、`client_request_count` 0.7105，其余多在 0.5~0.6。说明信号是**分散在多特征上的弱信号需要聚合**；Deep SVDD 把它们压进低维球心距离时反而损失了信息。这也解释了本轮"L0 与门控不相上下"——融合机制间的差异被这个更上游的问题盖住了。
- **训练池里 5380 行 inject 阶段行被当作正常数据**：`fault_inject_nontarget_train_fraction=1.0` 下，故障 case 的"非目标 endpoint" inject 行整段吸进 One-Class 训练池。但故障会传播，非目标 endpoint 在故障期间并不真的正常 ⇒ 正常边界被故障数据污染。当初设 1.0 是为了平衡类别比例（entry 019/022），代价可能大于收益，需实测检验。z-score 基线不训练，所以免疫这个问题。
- **RG `independent_sigmoid` 完全塌陷，根因是退化列 z-score 爆炸压垮 sigmoid**：`EndpointBaselineStats` 的 ep 分支有 4 个零方差退化列（`trace_error_rate`/`trace_5xx_rate`/`client_error_rate`/`client_5xx_rate`），std 兜底为 epsilon（既有已知行为，本轮运行时有 WARNING 记录），使 z-score 爆到 ~1e9、`gate_mlp` logits 达 ~1e8、`sigmoid()` 饱和到恰好 0 或 1、梯度消失，训练从第一步就冻死。**`softmax` 变体因输出必须和为 1 而受保护**（两路不可能同时饱和到 0），`independent_sigmoid` 无此约束。这是 025（标签修复）与 026（content_length 新特征）**首次共同在位**才暴露的集成问题，任一 PR 单独测试都不会出现。**2026-09-18 定案：RG 整体退役、不再修复**——one-class loss 下没有任何梯度把门控权重与检测质量绑定，softmax 坍缩与本次 sigmoid 饱和是同一结构性病根的两种表象，改参照系救不了"门控学无可学"（设计文档 D5：`docs/plans/2026-09-18-contract-v2-percase-design.md`）。
- **`artifacts/baseline_new_merge_*/metrics.json` 已被本轮重跑覆盖**：旧的仅-025 口径数字已不在工作区，只在 git 历史里。上表已保留两个口径的对照，引用旧数字请查 git。

## 遗留 TODO

> **排期说明**：AnoMod 数据集近期在筹划论文发表，不便重新采集，故 P1（重采）及其依赖项整体延后。P0 不依赖重采，可随时开工。

**P0 — 不依赖重采，可立即做**

> **2026-09-18 定案**：P0 已整体转入 contract v2 迭代——决策与规格见 `docs/plans/2026-09-18-contract-v2-percase-design.md`（D1–D6），18 个 TDD 任务的拆分与排期见 `docs/plans/2026-09-18-contract-v2-percase-implementation.md`，PR-4 起实施；以下条目保留为原始需求记录，后续进展以两份文档为准。

- **把 `rel_pos` 平凡基线与 z-score 基线写进 `eval_baseline_v0.py` 作为强制对照**。以后每个实验的 AUROC 旁边必须并列这两个数字。这不是为了投稿好看，是为了每次改动都能判断"有没有真的超过平凡规则"。目前它们只在一次性诊断脚本里，容易被遗忘。
- **改归一化参照系为 per-case baseline 自适应，然后重跑六机制**。这是预期收益最大的单项改动——把 z-score 基线的漂移免疫特性搬进 pipeline。做成 contract 配置开关，与现有全局归一化并列作为消融维度，**不要直接替换**：per-case 归一化会丢弃"绝对水平"信息，而资源型故障的信号恰在绝对水平上（`Lv_P_CPU_preserve` 的 `cpu_usage_rate` 修正后 AUROC 0.9993，是很好的 canary，改完必须核查它有没有掉）。**2026-09-18 定案：用 per-case z-score 而非 min-max**——实测 fit 子集 per-(case,endpoint) baseline 共 208 组，取前 20% 窗口后 median 仅 16 行、26/208 组（12.5%）不足 5 行，min-max 只用两个极值统计量，此规模下单个离群窗口即可扭曲 scale；且打败本系统的平凡基线本身就是 per-case mean/std 参照，用同一参照系才能 apples-to-apples 回答"SVDD 比 L2 范数强在哪"（设计文档 D1）。
- **把 `fault_inject_nontarget_train_fraction` 退回 0.0 重跑**，检验"污染 One-Class 边界"与"平衡类别比例"哪个代价大。成本只是重跑一次。
- **per-case 宏平均升为主指标，pooled 降为附注；AUPRC 补为并列主指标**。本轮两者未背离，但这是正确口径且成本为零。AUPRC 的理由是 [对抗压力测试](https://arxiv.org/html/2607.11969v2) 证明 ROC 家族在少量异常段下方差大到可被 seed 挑选（N=9 即可刷到 SOTA），而 PR 家族有 prevalence 地板刷不上去——本项目已独立踩过一次（RG softmax seed42 单点 0.6015 vs 四 seed 0.858~0.878，entry 025）。
- **补 seed{1,2,3}** 给本轮 5 个未塌陷机制，确认方向性结论不是单 seed 运气。entry 025 的教训：单 seed 报告高方差机制连"方向"都会错。

**P1 — 重采（已延期，方案先记下来）**

- **随机化 inject 起始位置**，让 `rel_pos` 与标签解耦。单这一条就能杀掉 0.9656 的平凡基线。
- **Normal case 全程交错采集**，至少 4~6 个均匀分布在采集窗口内，而非集中在头部。这样 `train_fit` 才能覆盖漂移而不是只记住第一小时。
- **每个 (故障类型, 目标) 重复 3~5 次**（[RCAEval](https://arxiv.org/html/2412.17015v1) 做法），支撑配对统计与 paired bootstrap。
- **一半 run 用固定顺序、一半用随机置换顺序并交错执行**，然后 Kruskal-Wallis + Bonferroni 检验顺序效应。方法论与工具见 [USENIX ATC'23 Ordering Trap](https://www.usenix.org/system/files/atc23-duplyakin.pdf) / [OrderSage](https://github.com/ordersage/ordersage)——他们用 230 万次测量证明采集顺序能让性能偏置 50%+、改变 72% 的结论。
- **完整记录采集顺序元数据到 `case_metadata.json`**，使后续任何时候都能重做时间混淆诊断（本轮是靠 `case_id` 里的时间戳硬抽出来的，脆弱）。

**P2 — 依赖 P1 数据**

- **service 级聚合消融**：同数据同模型聚合到 service 级，预期显著高于 endpoint 级。这把"0.90 偏低"转化为"细粒度的代价"，是回应"为什么你的数字比 Eadro 的 F1 0.989 低"的核心实验。
- **DeepTraLog 同口径复现**（Train-Ticket + Deep SVDD + One-Class，F1 0.954 @ 17.6% 异常率，trace 级）——文献里**唯一**与本项目范式可比的锚点，建议列为主对比基线。
- **若 P0 做完 Deep SVDD 仍打不过 z-score 基线**，需诚实面对"在 endpoint×window 粒度上深度 One-Class 可能不是正确工具"这一可能。那种情况下论文的正确形态不是"我们提出了更好的检测模型"，而是 **"现有微服务故障注入 benchmark 存在时间混淆与漂移，平凡基线即达 SOTA，并给出规范的采集与评估协议"**——这个方向的文献缺口是真的（见下），且本轮已攒下三个实锤：rel_pos 0.9656、z-score 0.9415、单一 Normal case 位于时间起点。

**不要做**

- DANN / gradient reversal 消除 case 身份信息 —— 本轮已证伪 case 身份是 shortcut，做了没用。
- 继续增加融合机制变体、继续在当前 `new_merge` 上调参 —— 天花板已被平凡基线锁死，改动无法被解释。

## 文献定位（本轮调研，供后续引用）

- **Train-Ticket 类数据集本身偏容易，有实证**：[TSC 2025 benchmark](https://nkcs.iops.ai/wp-content/uploads/2025/10/A-Comprehensive-Benchmark-and-Empirical-Study-of-Trace-Anomaly-Detection.pdf) 统一口径重跑显示同样方法在 Train-Ticket 上 F1 97~99、在 AIOps2020 上仅 48~72。**不能主张"任务难所以 0.90 可以"**。
- **文献高分大量来自可疑操作**：[Are GNNs Actually Effective?](https://arxiv.org/html/2501.02766) 复现 Eadro 时 F1 从 0.989 掉到 0.907 并指出其 window-splitting 有 data leakage，同篇一个**平凡 MLP** 在 Train-Ticket 上打平所有 GNN（90.8 vs 90.7）；[RCA: How Far Are We?](https://arxiv.org/html/2408.13729v2) 发现一批因果方法不如随机基线；DeepTraLog/TraceVAE 均用 best-F1（测试集上搜阈值）。**这两篇可直接用于论证"文献报告值不可直接采信"**。
- **Train-Ticket 上无 AUROC 基线可对标**：14 篇 AD 工作只有 2 篇报 AUROC（MSTGAD 0.974~0.996，半监督+instance 级；TraceVAE 0.954~0.994 合成异常 / **0.830 真实故障**——后者是全篇最有参考价值的单个数字）。
- **per-endpoint × time-window 粒度的文献空白是真的**：[TOSEM 2025 综述](https://nkcs.iops.ai/wp-content/uploads/2026/01/Failure-Diagnosis-in-Microservice-Systems-A-ComprehensiveSurvey-and-Analysis.pdf) 的粒度分类法只有 service/instance/component 三档，**没有 endpoint 档**，API 被并入 service。所有主流工作检测单元为 system-window / instance / trace / code-region。最接近的 [ChainLSTM](https://arxiv.org/html/2607.10156v1) 是 preprint、检测单元仍是 trace、异常为合成。**但空白本身不是贡献，必须用 service 级聚合消融证明细粒度带来了什么**。另需在 threat-to-validity 里精确表述：metrics/logs 只有 service 级粒度（采集端架构限制），所以严格说是"endpoint 级 RED + service 级两层 join"。
- **"故障注入数据集串行采集导致时间混淆"在 AIOps/chaos engineering 数据集文献里没有被正面提出过**（这本身可能是可发表的缺口）。相邻领域有精确对应物：系统性能测量的 [ATC'23 Ordering Trap](https://www.usenix.org/system/files/atc23-duplyakin.pdf)、时序 AD 的 [run-to-failure bias](https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/)、医学影像的 [embedding→医院探针](https://proceedings.mlr.press/v219/compton23a.html)（从 embedding 预测医院来源接近满分，而疾病任务只有 70~80%）。
- **本项目已知的 `KILLPOD` 遥测中断坑在文献里是一般性缺陷类别**：[Fault Propagation-Aware Benchmark](https://arxiv.org/html/2510.04711) 把它命名为 **Signal-Loss Blind Spot**（PodKill 导致遥测中止），不是本数据集特有问题（对应 entry 016/023 剔除 `Lv_S_KILLPOD_gateway` 的处理）。

## 复现命令

```bash
# 六机制重跑（本轮已执行）
dvc repro dvc_new_merge/dvc.yaml

# 时间混淆与朴素基线诊断（不进 dvc pipeline，只读）
python scripts/analyze_temporal_confounding.py \
    --contract-dir artifacts/contract_new_merge_expanded \
    --scores-dir artifacts \
    --out artifacts/temporal_confounding/report.md
```
