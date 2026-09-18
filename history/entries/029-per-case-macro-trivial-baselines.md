# 029 · per-case 宏平均升主指标 + 平凡基线固化为常驻 dvc 参照线（entry 027 P0 落地）

- **日期**: 2026-09-19
- **PR**: #28 · **Commit**: 见本 PR merge（dvc stage 注册 `cc7aaa4`）
- **类型**: Feature + Experiment
- **影响域**: `scripts/eval_baseline_v0.py`, `scripts/score_trivial_baselines.py`, `dvc_new_merge/dvc.yaml`, `dvc_new_merge/dvc.lock`, `artifacts/trivial_baseline_*/`, `artifacts/baseline_new_merge_*/metrics.json`（仅新增键）, 标签粒度/评估口径

## 做了什么

entry 027 P0 的两件事落地（规格见 `docs/plans/2026-09-18-contract-v2-percase-design.md`，本 PR 是该计划 PR-2）：

1. **per-case 宏平均升为主指标**：`eval_baseline_v0.py` 顶层新增 `per_case_auroc_macro` / `per_case_auprc_macro` / `n_cases_with_both_classes` 三键——按 `case_id` 分组逐 case 算 AUROC/AUPRC，只统计同时含正负两类的 case（单类 case 的 AUROC 无定义，计入会静默拉偏均值），等权宏平均。与 PR #26 benchmark 的 per-case macro 同口径（entry 028 记录该 benchmark 只覆盖 16 个 endpoint 档 case；本契约口径按三档标签的 `is_endpoint_anomaly` 统计，覆盖全部含正样本的 case），两套数字从此可比。pooled `auroc`/`auprc` 保留为附注。
2. **三条平凡基线固化为常驻 dvc stage**：`scripts/score_trivial_baselines.py` 产出契约合规的 scores.parquet（10 列，过 `scores_v0` 校验），不打接口墙的洞——`eval_baseline_v0.py` 原样消费，与任何学习模型走同一条 eval 路径。3 基线 × (score+eval) 共 6 个 stage 注册在 `dvc_new_merge/dvc.yaml`，消费 `contract_new_merge_expanded`，产物落 `artifacts/trivial_baseline_<name>/`：
   - `rel_pos`：窗口在 case 内的相对时序位置（dense-rank 序号法），不看任何特征；
   - `zscore_l2`：逐特征 per-(case,endpoint) z-score 后取 `sqrt(Σz²)`；
   - `zscore_max`：同参照系取 `max|z|`。
   后两者是 **lean 口径**：mean/std 只 fit 在 `train.parquet`，不用 eval 侧 baseline 行。

### 三条基线的常驻参照数字（seed 无关，纯确定性）

| 基线 | per-case macro AUROC | pooled AUROC | macro AUPRC（per-case） |
|---|---|---|---|
| **rel_pos** | **0.9656** | 0.9303 | 0.6008 |
| zscore_l2 | 0.9539 | 0.9302 | 0.7862 |
| zscore_max | 0.9441 | 0.9232 | 0.7921 |

`rel_pos` 的 0.9656 **精确复现 entry 027 锚点**（`analyze_temporal_confounding.py` 手算/固化值），说明生产者经接口墙绕了一圈后口径没有漂移。对照：entry 027 口径下最好的学习模型 DWF per-case macro 0.9238——三条平凡规则至今仍全部在线上。

### lean vs transductive 单变量对照（方向与计划预期相反）

计划预期 transductive（fit 用 eval_all 的 baseline 行 11978 行，即"偷看 eval"）应该比 lean 高。同代码路径、唯一变量是 fit 源的实测：

| 口径 | l2 macro | max macro |
|---|---|---|
| transductive（eval baseline 11978 行 fit） | 0.9419 | 0.9401 |
| **lean（train.parquet fit）** | **0.9539** | **0.9441** |
| Δ（lean − transductive，macro） | **+0.0120** | **+0.0040** |

per-case macro 上 **lean 反而高 +0.012**（pooled 上方向相反，lean 低 −0.013）。机制：service 档 case（7 个，如 DISKIO/DNSFAIL 类资源/基础设施故障）的 baseline 段特征近乎恒定，transductive fit 时这些列零方差、std→NaN 被丢弃，等于该模态在这些 case 上**失明**；lean 的 train 池含 `fault_inject_nontarget_train_fraction=1.0` 吸收的 5380 行 inject 非目标行（entry 022/025），这些行给了原本恒定的特征真实尺度，零方差失明被修复。另注：entry 027 诊断脚本的 transductive 数字是 0.9415，同代码路径为 0.9419，差 0.0004 即 `MIN_BASELINE_WINDOWS` 闸门 / NaN 聚合等实现差异的量级，两者互相印证。

**教训："偷看 eval"不总是加分。** per-case 口径下，lean 参照的 train 池 inject 行反而修复了 service 档 case 的退化列失明；直觉里"transductive 信息更多必然更好"只在 pooled 口径、且 fit 段本身有足够变化时成立。

## 关键决策（不在 commit 里）

- **以后每个实验的 metrics.json 旁边必须并列这三条基线数字**，per-case macro 为主指标、pooled 为附注。这是 entry 027 P0 的核心动机——"有没有真的超过平凡规则"不能靠记忆偶尔跑一次诊断脚本来回答；常驻 stage 让任何 `dvc repro dvc_new_merge/dvc.yaml` 的消费者天然带着参照线。
- **z-score 选 lean 而不是 transductive 进 pipeline**：除上面实测 lean 在主指标上更高外，拿一个偷看过 eval 的参照审判模型，赢输都说不清；transductive 数字只作为一次性对照留在本 entry。
- **y_true 口径保持现状**（review 发现、已在脚本 docstring 显式声明）：`score_trivial_baselines.py` 输出的 `y_true` 取 `is_endpoint_anomaly`，而 `train_baseline_v0.py` 取 `is_anomaly`（`phase=="inject"`）——两个 scores 生产者的 y_true 口径不同。`eval_baseline_v0.py` 不读 y_true 列（只用 `is_endpoint_anomaly`），**当前影响为零**；未来直接按 `scores_v0` 契约读 y_true 的消费方需注意此差异。定案保持现状的理由：计划原文指定 `is_endpoint_anomaly`，且它与 eval 实际消费的标签一致。
- **零方差列用 std=1.0 哨兵而非 1e-9 兜底**：eval 侧任何非零差值被放大 1e9 倍是 history 013 的爆值与 entry 027 RG sigmoid 饱和的同根因；组在 fit 里不存在 / 特征缺测的 NaN 统一按 0 偏离处理（没有参照不判异常、缺测不等于异常）。

## 坑 / 已知问题

- **rel_pos 的两种等价格局只在等距采集下恒等**：本生产者按计划规格用 dense-rank 序号法 `(rank−1)/(n−1)`，entry 027 诊断脚本用时间戳 `(ts−min)/span` 法。当前数据是等距 15s 窗，两者 per-case macro 精确相同；但 eval_all 因训练池吸收出现空窗时，pooled AUROC 有微差（0.9303 vs 0.9340），未来非等距采集会正式分叉。引用 rel_pos 数字时需注明用的是哪一版。
- **退化哨兵测试用例原本无判别力**（PR review 抓到）：计划给的测试模板里 fit/eval 两侧常量相同，分子恒为 0，哨兵取 1e-9 还是 1.0 测试都能过——等于没测。修复为 eval 侧常量置 6.0 并断言 score 恰为 1.0（`|6−5|/1`），双向锁定哨兵与分子路径。
- **本 PR 改了 `eval_baseline_v0.py` → 所有消费它的既有 metrics.json 被刷新**：`dvc_new_merge` 下 6 个既有 eval stage（concat/independent_concat/gated/RG-softmax/RG-indep-sigmoid/DWF）全部重跑，但 diff **仅新增 per_case 三个键，pooled AUROC/AUPRC 与分层数字逐位不变**。**PR-4 的 v1 回归门（"v1 metrics 逐位不变"）应以本 PR 合并后的版本为基准**，否则会把新增键误判为回归。
- **裸 `dvc repro dvc_new_merge/dvc.yaml` 在本 PR 时点会连带重跑 build_contract，不能直接用**：磁盘上 `data/new_merge` 的内容哈希自 2026-08-24 起就相对提交的 `data/new_merge.dvc`（`2f049a3d…`）漂移——189 个**上游派生产物**（各 case `_pipeline_out/tt_fused_{5,10,15}s.csv` 与 `tt_traces_red_*report.md`，mtime 2026-08-24，疑为 entry 027 时期外部上游 pipeline 重跑）内容已变但从未 `dvc commit`。这些文件**不是** contract 的输入（build 消费原始 traces/api/log/metric，不读 `_pipeline_out/tt_fused_15s.csv`；后者只有 benchmark 脚本直读，见 entry 028），但它们在 `data/new_merge` 的 .dir 清单里，任何哈希漂移都会让 build stage 失效。本 PR 的应对：**不重跑 build/train**（否则 6 个既有 metrics 的"数字逐位不变"失去核验基础），改为对 3 个新 score stage 与全部 9 个 eval stage 逐个 `dvc repro --single-item`，contract/scores 依赖本地 cache 中锁定的对象（21 个 outs 的 cache 对象经核验全部在位，eval_all 恢复后 md5 与 lock 逐位一致）。另注：repro 启动时 DVC 会把磁盘上缺失但 .dir 登记为空哈希（d41d8cd9）的 27 个 `rabbitmq-*_stream.stderr.log` 自动 restore 成 0 字节文件并改写 `data/new_merge.dvc`（189 个漂移文件也使该文件 hash 变为 `9f2080cb…`），已 `git checkout` 还原该文件——**data/new_merge 维持 READ-ONLY，本次未提交任何数据 hash 变更**。
- **历史欠账：189 个 `_pipeline_out` 漂移文件的处置留给独立改动**：选项是 `dvc commit data/new_merge.dvc`（承认 8 月 24 日的上游重算结果）或从备份恢复旧版本；两者都属数据集版本变更，不该搭车进本 PR。在处置之前，任何人裸跑 `dvc repro dvc_new_merge/dvc.yaml` 都会触发全链路重跑，需用 `--single-item` 或先 `dvc checkout dvc_new_merge/dvc.yaml` 恢复锁定产物。
- 三条基线都是确定性计算（无随机数、无训练），不需要多 seed；这也意味着它们的数字是硬参照，任何时候 `dvc repro` 都必须复现同值——未来 contract 切分/标签口径变化引起基线数字漂移时，应视为口径变化的信号而非噪声。

## 遗留 TODO

- entry 027 P0 其余项在后续 PR：`fault_inject_nontarget_train_fraction` 退回 0.0 对照（PR-3）、per-case z-score 归一化搬进 contract + 六机制重跑（PR-4 起，见 contract v2 实施计划 18 任务拆分）。
- rel_pos 0.9656 的根治依赖 P1 重采（随机化 inject 起始位置 / Normal 交错采集），AnoMod 论文期延期，见 entry 027。
