# 028 · 下游异常检测 benchmark：F-full 三检测器一致 +0.09，PR #26 欠账补记

- **日期**: 2026-09-18（PR #26 合并当日补写；合并时未按项目规则写 entry）
- **PR**: #26 · **Commit**: `25cd20b`（benchmark 实际运行时代码版本 `2ec7460`，记录在每份 `metrics.json` 的 `git_commit` 字段）
- **类型**: Docs（补账；PR #26 本身是 Experiment）
- **影响域**: `scripts/benchmark_downstream.py`, `scripts/run_benchmark_sweep.py`, `artifacts/downstream_benchmark/`, 研究方向/论文 framing, 标签粒度/评估口径

## 做了什么

论文返修支撑材料，回应审稿意见 2（"只报字段可分性、没跑真实检测器"）。在论文发布的 **26 runs / 23,244 windows / 639 正样本**（2.7%）口径下（剔除 `Lv_D_TRANSACTION_timeout_*`，其 inject 阶段仅 4 个窗口，与论文 Table `tab:scale` 一致），跑 **2 特征臂 × 2 训练池 × 3 检测器 × 4 seed = 48 cell** 主网格，外加 8 个缺失值混淆消融 cell，共 56 个 run 目录，每个含 `scores.parquet` + `metrics.json`：

- 特征臂：**F-inherited**（18 维，9 trace RED + 9 个原有 client RED）vs **F-full**（21 维，加 `client_content_length_mean` / `client_content_length_rel_shift` / `client_body_hash_mismatch_rate` 三个新客户端字段）。
- 检测器：DeepSVDD（复用 `src/models/deep_svdd.py`）、sklearn `OneClassSVM(rbf)`、sklearn `IsolationForest`。OCSVM rbf 模式确定性，4 seed 数值相同。
- 训练池：`normal_only`（单一 Normal case 最早 80% 窗口，647 行）vs `expanded`（再吸收每个故障 case baseline 阶段最早 80%，11,960 行）。
- 显式排除 19 个元数据/标签/弱标签/异常信号列防泄漏；归一化 mean/std 只在训练池 fit；count/rate/coverage 列 NaN→0，延迟/内容完整性列 NaN 用**训练池均值**填补（review Critical #3 修掉了早先全列 0 填充）。

### Headline 结果（expanded 池，per-case macro AUROC，4 seed mean ± std）

| 检测器 | F-inherited (18) | F-full (21) | Δ |
|---|---|---|---|
| **DeepSVDD** | 0.866 ± 0.008 | **0.957 ± 0.008** | **+0.091** |
| OCSVM | 0.836 | 0.950 | +0.114 |
| IsolationForest | 0.733 ± 0.005 | 0.814 ± 0.003 | +0.081 |

三个**机理完全不同**的检测器方向一致，这正是审稿人要的 claim：新客户端可观测性字段的增量不是某个模型的运气。返修表建议以 F-full + expanded + DeepSVDD（per-case AUROC 0.957 / AUPRC 0.709）领衔，OCSVM 0.950 作为经典非深度基线，IF 报广度。

### 缺失值混淆消融（review Critical #2，DeepSVDD/expanded）

两个贡献最大的字段（`rel_shift` / `body_hash_mismatch`）NaN 率不对称（目标 endpoint inject 行 49% vs 非目标 15%），审稿风险是检测器只学到"目标 endpoint 上这列丢了"。实测：

| 特征集 | 维数 | per-case AUROC |
|---|---|---|
| `missing_indicator_only`（只用这 2 列的 is-NaN 指示） | 2 | **0.667** |
| `full`（F-full，均值填补） | 21 | 0.957 |
| `full_with_missing_indicator`（F-full + 2 个显式指示列） | 23 | 0.959（仅 +0.002） |

两条反证都成立：纯缺失模式只有 0.667（差 F-full 0.29），把缺失信号显式化也只 +0.002 ⇒ **增益压倒性来自字段真实取值，不是 NaN 模式**。（`content_length_mean` 两侧 NaN 率均为 0%，不参与该消融。）

## 关键决策（不在 commit 里）

- **平凡基线在本 benchmark 上的结果决定哪些数字能进返修表**（与 entry 027 的关键连接）：每份 `metrics.json` 内置两个 `_internal_only_` 基线。expanded 池上 rel_pos（不看任何特征的窗口相对位置）per-case AUROC = **0.5786**、z-score（F-full 特征上的 per-case baseline 自适应 L2 范数）= **0.7657**，**远低于 headline 0.957** ⇒ entry 027 在 contract 主链路上发现的时间混淆问题**不波及这组返修数字**，可以安全引用。但 `normal_only` 池上平凡基线反过来打败全部学习模型：rel_pos = **0.800**、F-full 臂 z-score = **0.791**，而该池学习模型的最高成绩仅 OCSVM F-full 0.781（DeepSVDD 更低：F-inherited 0.636 / F-full 0.723）——按当时决定，normal_only 池只作为 supporting cell 报告、平凡基线只存 `_internal_only_` 键，**不进返修表**。注意 z-score 基线在特征上计算、数值随特征臂而变（expanded 池四臂穷举：inherited 0.714 / full 0.766 / full+indicator 0.781 / indicator-only 0.667；normal_only 池 F-full 臂为 0.791），引用时要带上对应 cell 与池。
- **保留独立 benchmark 脚本而不是塞进 contract pipeline**：返修要的是"论文发布口径原样复现"，与 contract v2 改造期的代码解耦可避免数字随主线重构漂移。代价见坑 4。
- **NaN 填补区分列语义**：计数/比率/覆盖列"无事件⇒率为 0"是合法 0 值；延迟统计与内容完整性字段 0 填充等于伪造一个"最佳观测"，必须用训练池均值。这是 PR review 抓到后修的，不是一开始就对。

## 坑 / 已知问题

- **commit message / PR 描述里的 headline 数字已过时，是 review 修复前的旧数**：`25cd20b` 的 message 写的是 DeepSVDD 0.851→0.951（Δ+0.100）、OCSVM 0.832→0.941、IF 0.719→0.771。这些是 review 三个 Critical（训练池零真负样本 / 缺失值混淆 / 延迟列 0 填充）修复**之前**跑出来的，修复后未更新 message。**引用一律以 tree 内 `summary.md` / `_summary.tsv` 为准**：0.866→0.957、0.836→0.950、0.733→0.814。
- **"26 runs benchmark"的 per-case macro 实际只覆盖 16 个 case**：26 runs 里只有 16 个 `Lv_E_HTTP*` case 是 endpoint 级标签（有 `target_endpoint`）；其余 10 个 `Lv_P_*`/`Lv_S_*`/`Lv_D_*` service/case 粒度 case 在该脚本"endpoint-only"标签定义下正样本为 0，不进 macro（每份 `metrics.json` 的 `n_cases_with_both_classes_for_auroc: 16`）。**这实质上是 `Lv_E_HTTP*`-only benchmark**，论文里把"26 runs"与这些 AUROC 并列时必须同时声明此范围，否则是 silently overclaim。
- **PR 描述承诺的 `_summary_agg.tsv` 不存在**：commit message 称 sweep "落 `_summary.tsv` / `_summary_agg.tsv`"，实际 `run_benchmark_sweep.py` 只写 `_summary.tsv`（56 行逐 run 原始结果），tree 里也没有 agg 文件。聚合（mean/std over seeds）只能临时 groupby，没有固化产物；summary.md 里的表是手工/一次性生成的。
- **这是与 contract 并行的第二套特征/切分实现，存在口径漂移风险**：脚本**直读** `data/new_merge/<case>/_pipeline_out/tt_fused_15s.csv`，不经 contract parquet；21 维特征清单（`FEATURE_COLS_FULL`）在脚本里硬编码、吸收比例 `TRAIN_POOL_FRACTION=0.8` 硬编码、标签按 `phase=="inject" AND is_target_endpoint==1.0` 自行重算、不挂任何 DVC stage。与主链路的耦合点只有三处 import（`split_fault_phase_temporal`、`DeepSVDD`、`set_seed`）。contract 侧任何字段口径/切分/fraction 变化都不会自动传播到这里，反之亦然——未来若再跑此 benchmark 必须先逐项核对两套口径。
- **产物不在工作区时只能从 git tree 取**：`artifacts/downstream_benchmark/` 随 PR #26 进 master，在不含该 commit 的分支上工作区没有这些文件；本 entry 的全部数字核对都是通过 `git show 25cd20b:artifacts/downstream_benchmark/...` 完成的。引用前先确认所在分支包含 `25cd20b`。
- **overall AUROC 与 per-case macro 差 ~0.03~0.09**（F-full expanded：DeepSVDD 0.924 vs 0.957）：global ranking 被大 case 主导，macro 等权。返修主表用 per-case macro 与 entry 027 确立的口径一致，但两个数不要混用。

## 遗留 TODO

- **contract v2 落地后处置这套平行实现**：per-case 归一化 / fraction 0.0 对照等主线改造（见 contract v2 实施计划）完成后，要么废弃 benchmark 脚本并把三检测器对比搬进 dvc pipeline，要么显式对齐两套 21 维清单与切分参数；不能让两套口径长期并存。
- **补 `_summary_agg.tsv` 或在脚本里固化聚合产物**，避免 summary.md 的表只能靠手工 groupby 复现（entry 027 的教训：数字进文档前必须有可复现路径）。
- **10 个 service/case 级 case 在本 benchmark 完全没有 per-case 成绩**：若返修需要覆盖资源型故障（`Lv_P_*`），需按 service 粒度标签另跑一臂，不能引用这 16-case 数字冒充全量。
- **normal_only 池平凡基线（rel_pos 0.800 / F-full z-score 0.791）打败全部学习模型（最高仅 OCSVM 0.781）**是 entry 027 时间混淆问题在该切分下的又一个旁证（单 Normal case 647 行、位于采集窗口起点）；该池数字不进返修表，但在"采集协议修复后重跑"的清单里应包含此池，验证平凡基线是否随之回落。
- 复现入口：`python scripts/run_benchmark_sweep.py`（默认 `--out-root artifacts/downstream_benchmark`，支持断点续跑且校验文件内容而非仅存在性）；单 cell 用 `scripts/benchmark_downstream.py --feature-set {inherited,full,missing_indicator_only,full_with_missing_indicator} --pool-mode {normal_only,expanded} --model {deep_svdd,ocsvm,iforest} --seed`。
