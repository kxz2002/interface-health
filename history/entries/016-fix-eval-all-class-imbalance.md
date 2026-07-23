# 016 · 修复 `expand_train_pool` 导致的 eval_all 类别失衡（issue #16）

- **日期**: 2026-07-23
- **PR**: 待开 PR（branch: `bugfix/eval-all-class-imbalance`）· **Commit**: `096114a^..0c36cc3`（8 commit，含起点 `096114a`：2 `[Feature]` + 2 `[Test]` + 1 `[Bugfix]` + 1 `[Experiment]` 重跑 + 2 `[Chore]` 收尾，不含本次文档同步 commit）
- **类型**: Bugfix
- **影响域**: `src/contracts/split_fault_baseline.py`, `src/contracts/contract_config.py`, `scripts/build_contract.py`, `configs/contract/v1_expanded_pool.yaml`, `dvc_reliability_gate/dvc.yaml`, `tests/`, `CLAUDE.md`

## 做了什么

entry 014 记录、issue #16 跟踪的 eval_all 类别失衡问题：`expand_train_pool=true` 时，故障 case 的 `baseline` 阶段行被整段搬入训练池、整段从 `eval_all` 摘除，导致 eval_all 正负比从 ~50:50 崩到 ~84:16（负样本几乎被掏空）。本次修复改为按时间窗时序切分——新增 `split_fault_baseline_temporal`（两路切分函数），由新配置字段 `fault_baseline_train_fraction`（默认 `0.2`）控制：每个故障 case 内最早 20% 的时间窗进训练池，其余 80% 留在 `eval_all`。真实 29-case 数据上重跑后，eval_all 正负比（positive : negative，以 `is_endpoint_anomaly` 为标签列）从 ~84:16 回升至约 **24.62% : 75.38%**（即负样本占比从 ~16% 回升到 ~75%）。同时重跑了 RG（`reliability_gate` + `deep_svdd`）pipeline 刷新 `metrics.json`。

## 关键决策（不在 commit 里）

- **按时间窗时序切分，不按行随机切**：与 `split_normal_rows_temporal` 保持一致的防泄漏纪律——切分单位是时间窗而非单行，保证 `train.sample_id ∩ eval_all.sample_id == ∅` 由整窗归属直接保证，不会把同一时间窗的部分 endpoint 行分到 train、部分分到 eval（这类拆散在 service 级广播特征场景下会造成隐性泄漏）。
- **新写独立函数而非复用/改造三路 Normal 切分**：`split_fault_baseline_temporal` 与 `split_normal_rows_temporal` 刻意不共用同一实现，原因有二：(1) 前者是两路（train/eval，无 val），后者是三路；(2) 操作对象不同——一个是故障 case 的 `baseline` 阶段行，一个是 Normal case 全量行；边界退化时的兜底策略也不同（见下条）。
- **边界退化不做"至少 1 窗"兜底**：`n_train = int(n * fraction)` 允许某故障 case 贡献 0 行给训练池（窗数极少时）。这是刻意的——`split_normal_rows_temporal` 对 `train_fit` 做兜底是因为 `train_fit` 为空会导致 `Normalizer` 无法 fit；这里训练池已有纯 Normal 的 `train_fit` 打底，某故障 case 贡献 0 行完全不影响训练可行性。反过来若强行兜底"至少 1 窗进 train"，会在窗数极少的 case 上把仅有的窗吃进 train，让该 case 在 eval 里的 baseline 覆盖率归零——恰好是本次要修的问题的另一种形态。
- **默认 `fraction=0.2`，但这个数字的原始依据没有站住**：计划草稿基于一张估算表，预期 `fraction=0.2` 能把 eval 正负比拉回 ~45:55（接近原始 50:50）。真实数据重跑后实测是 24.62:75.38，比预期偏离更远（细节见下方"坑"）。已决定保留 `0.2` 作为默认值——它仍比修复前的 84:16 有实质改善，且 Normal 数据集后续扩充后这个比例本就要重新调——但不重写这条决策的叙事：目标没有完全命中，如实记录。
- **recover 阶段不动**：延续 entry 014 的既有决策，只切 `baseline` 行，`inject`/`recover` 行的训练/评估归属不受本次改动影响。

## 坑 / 已知问题

- **计划阶段的估算方法有缺口，不是切分逻辑本身有 bug**：计划文档基于一张设计草稿表格估算 `fraction=0.2` 能把正负比拉回 ~45:55，这个估算是按"inject 行数 vs 非-inject 行数"的粗粒度计数做的。但 `scripts/eval_baseline_v0.py` 实际算 AUROC/AUPRC 用的标签列是 `is_endpoint_anomaly`，而不是"是否 inject 阶段"本身——29 个真实故障 case 里有 16 个是 `endpoint` 级异常（`endpoint_raw2`），每个 inject 窗口内 8 个 endpoint 只有 1 个是真正的正样本，其余 7 个在同一窗口内仍是负样本。粗粒度估算没有把这个 1/8 折算进去，导致真实测出的比例（24.62:75.38）比原估算更不均衡。这是规划阶段的估算方法学缺口，不是切分代码的缺陷——`split_fault_baseline_temporal` 本身逻辑正确且有单测覆盖（`tests/test_split_fault_baseline.py`），问题出在"用什么口径估算最终效果"这一步。
- **entry 014 的 RG AUROC/AUPRC 数字不再可比**：eval_all 的组成已经改变，同一 RG 配置（`reliability_gate` + `deep_svdd`，seed=42）在新旧两版 eval_all 上的数字不能直接摆在一张表里比：
  - 旧（entry 014，修复前失衡的 ~84:16 eval_all，`n_samples=8221`）：AUROC=0.6024, AUPRC=0.5278
  - 新（本次修复后的 eval_all，`n_samples=12683`）：AUROC=0.5789, AUPRC=0.3268
  AUPRC 从 0.5278 掉到 0.3268 是**预期且正确**的结果，不是模型变差——entry 014 自己的横切主题笔记（history/index.md 第 107 条）就已经预判"训练池扩容改 eval_all 摘除逻辑容易在无意中破坏 eval 集合类别平衡"，旧的高 AUPRC 本身就是被类别失衡撑高的假象。AUROC 几乎没动（0.6024→0.5789），符合 AUROC 对类别比例远不如 AUPRC 敏感这一已知性质。
- **`Lv_S_KILLPOD_gateway`（n=221）AUROC 为 null，已确认非 bug**：新 `metrics.json` 里 `Normal`（n=236，按定义全负）、`by_anomaly_level.none`（n=236，同上）之外，`Lv_S_KILLPOD_gateway` 这个分层也是 `auroc: null`。专项核查发现：这个故障 case 的 eval 行 100% 为 `phase==baseline`（全部 221 行），即该 case 在本数据集里没有 inject/recover 行进入 eval_all（其最早的窗口——约 8 个窗口对应的 39 行——已被 `fraction=0.2` 吸收进训练池），导致该分层是单一类别（全负），AUROC 数学上无定义。与 `Normal`/`none` 是同一类现象，不是数据 pipeline 缺陷。
- **共享 fixture 的切分粒度限制，需要新 fixture**：现有 `mini_dataset.yaml` 的故障 case 每个只有 1 个 baseline 时间窗，`int(1 * 0.2) = 0`，永远切不出非零的训练池份额，无法验证"部分窗口被吸收"这条分支。新增 `tests/fixtures/split_fraction_mini.yaml`（故障 case 有 5 个 baseline 窗口，`int(5 * 0.2) = 1`，可验证非零切分）。构造该 fixture 时沿用了 Normal case 与故障 case 共享同一 endpoint 名字的技巧，避免 `EndpointBaselineStats` 在推理侧遇到训练侧未见过的 endpoint key 而抛 `KeyError`。
- **跨 fraction 取值比较 AUROC/AUPRC 仍然无效**：这条 entry 014 就有的告诫继续有效——任何未来调整 `fault_baseline_train_fraction` 后重跑的指标，必须连同当次 eval_all 的实际类别比例一起报告，不能只摆数字比大小。

## 遗留 TODO

- 尚未做 `fraction` 取值的敏感性扫描。当前 `0.2` 是基于一个已被证明不准确的估算选定的，不是在多个真实 fraction 取值上扫描实际效果后选出的最优值。
- 若后续要用 RG 的实验结论论证方向性判断，需要在**固定** fraction 下做多 seed 复跑，而不是像本次一样在变动的 eval 口径之间比较——固定口径是任何后续统计显著性论证的前提。
