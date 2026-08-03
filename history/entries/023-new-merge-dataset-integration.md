# 023 · new_merge 数据集接入 + fraction 校准 + KILLPOD_gateway 剔除 + L0/L1/L2/RG/DWF 全融合机制对比

- **日期**: 2026-07-31 ~ 2026-08-03
- **PR**: 待开 PR（branch: `exp/new-merge-baseline-rerun`）
- **类型**: Data + Experiment
- **影响域**: `data/new_merge.dvc`, `data/.gitignore`, `configs/data/new_merge.yaml`, `configs/contract/v1_new_merge.yaml`, `dvc_new_merge/dvc.yaml`, `artifacts/contract_new_merge_expanded/`, `artifacts/baseline_new_merge_concat*/`, `artifacts/baseline_new_merge_independent_concat*/`, `artifacts/baseline_new_merge_gated*/`, `artifacts/baseline_new_merge_reliability_gate*/`, `artifacts/baseline_new_merge_deviation_weighted*/`

## 做了什么

接入 `new_merge`（AnoMod 侧新采集的单一批次数据集，2026-07-28/29，原始 28 case）为独立数据源，不与 `anomod_v1`/`endpoint_raw2`/`new_ep1` 等历史批次混合（entry 017 已证实跨 run 合并会污染评估）。沿用 `dvc_new_ep1/dvc.yaml` 的隔离模式，新建 `dvc_new_merge/dvc.yaml`（`build_contract_new_merge_expanded` → 各融合方式的 train/eval stage），裸 `dvc repro` 不触发。首轮只跑 `fusion=concat`（L0），后续两轮陆续补齐 `independent_concat`（L1）/`gated`（L2）/`reliability_gate`（RG，softmax 默认 + independent_sigmoid 消融）/`deviation_weighted`（DWF），复用同一份 `contract_new_merge_expanded`，不重新 build。

`fault_baseline_train_fraction` 未直接借用其他数据集的已校准数值，而是实测对比 `1.0`（借用 `new_ep1`）与 `0.2`（借用 `endpoint_raw2`/`v1_expanded_pool`）两个取值下 `eval_all` 的实际类别比例，选定 `0.2`。过程中发现 `Lv_S_KILLPOD_gateway` 在 `eval_all` 里 100% 落在 `phase=='baseline'`（462 行，单一类别），`by_anomaly_type` 分层 AUROC 数学上无定义（`null`）——与 entry 016 在 `endpoint_raw2` 上遇到的同名 case 是完全相同的结构性缺陷（`ts-gateway-service` 既是 PodChaos 目标又是 SkyWalking trace 上报通道，pod 被杀后 inject 阶段 trace 永久缺失），非本次切分逻辑的 bug。确认该 case 未进入任何已提交历史（`git log --all -- data/new_merge.dvc` 为空）后，物理删除该 case 目录，重新 `dvc add` 生成 27-case 版本，重跑 contract 构建 + L0 训练评估。

清理前后对比：overall AUROC 0.718→0.742，`eval_all` 从 16271→15809 行（少 462），正负比 19.7:80.3（3113 正 / 12696 负）。清理后 `Lv_S_KILLPOD_gateway` 从 `by_anomaly_type` 中完全消失，其余分层同步小幅提升（去掉一个标签无效的 case，稀释效应消除）。

后续两轮在同一份 `contract_new_merge_expanded` 上补齐了 L1/L2 与 RG/DWF，均按 seed{1,2,3,42} 四组重跑（而非单 seed），遵循 entry 014 "多 seed 方差本身就是证据"的既有方法论。全部六种融合方式的多 seed 均值（AUROC，mean/std）：

| 融合方式 | mean | std | 4 组值 |
|---|---|---|---|
| concat (L0) | 0.7109 | 0.0259 | 0.7416, 0.7082, 0.6785, 0.7154 |
| gated (L2) | 0.6814 | 0.0230 | 0.7118, 0.682, 0.6563, 0.6755 |
| independent_concat (L1) | 0.6975 | 0.0094 | 0.6941, 0.7062, 0.6857, 0.7038 |
| reliability_gate softmax (RG) | 0.6575 | 0.0304 | 0.651, 0.6687, 0.6132, 0.6972 |
| reliability_gate indep_sigmoid | 0.6570 | 0.0480 | 0.6778, 0.6985, 0.6765, 0.5753 |
| deviation_weighted (DWF) | 0.7047 | 0.0061 | 0.7043, 0.7149, 0.7, 0.6997 |

结论：concat(L0) 仍是均值最高的方式，DWF 紧随其后且方差最小（std=0.0061，全部六种里最稳）。L0→L1→L2 的非线性容量/门控条件融合均未带来正向增量（与 entry 010/012 在旧数据集上的结论一致，new_merge 上甚至不增反降）。RG 两个变体都明显劣于 L0/DWF。

entry 014 在 `merged_v2`/`endpoint_raw2` 上发现的 softmax gate collapse（全部故障类型收敛到 `w_svc≈1.0`）在 `new_merge` 上**未见同等量级复现**：softmax 与 indep_sigmoid 均值几乎相等（0.6575 vs 0.6570），且 indep_sigmoid 的方差反而更大（0.0480 vs 0.0304，seed3=0.5753 明显偏低）。旧数据集上两者差距是 0.0574（0.6598 vs 0.6024，indep_sigmoid 明显更优），new_merge 上差距缩小到几乎持平——说明该坍缩现象是否发生、发生后收益有多大，与具体数据集的特征分布/故障类型组合相关，不是 RG 机制本身的普适缺陷，也不是能直接套用旧数据集结论的"结构性问题"。

## 关键决策（不在 commit 里）

- **fraction 必须实测而非借用**：即使 `new_ep1` 与 `new_merge` 的 `baseline_sec` 量级接近（1500s vs 1200s），两者的 `fault_baseline_total` 绝对行数和其他负样本项（fan-out 残留、Normal holdout）的相对大小不同，`fraction` 的实际效果只能通过实测 `eval_all` 的类别分布验证，不能靠 `baseline_sec` 类比推断。`1.0` 在 `new_merge` 上会让故障 case 的 baseline 阶段行全部被吸收，负样本地板被削到只剩 fan-out 残留（正负比严重偏正，不可用）；`0.2` 产出的绝对负样本行数（12696~13158）和比例都在entry 016 校准目标附近，故选 `0.2`。
- **剔除 KILLPOD_gateway 而非保留 null 分层**：曾考虑保留该 case 只是让 `by_anomaly_type` 该项显示 `null`（不参与宏平均），但用户判断"AUROC 数学上无定义"的 case 不该留在数据集里污染"数据集干净度"，且 entry 016 已有先例证明这是可重复出现的已知模式，非一次性意外。删除前核实了两件事：(1) 清理前的 `data/new_merge.dvc` 从未进入过任何历史 commit，删除零风险；(2) 当时下游所有 contract/baseline 产物均未提交，重跑无兼容性负担。
- **首轮范围只跑 concat(L0)，L1/L2/RG/DWF 分两轮追加**：避免把"数据集接入+清理"和"融合机制对比"两件事耦合在一次改动里；L1/L2 是在已有 uncommitted 状态下发现后经用户确认"提交 L0+L1+L2 全部结果"才合并进同一条实验线，RG/DWF 则是用户明确要求"重跑 RG 和最新 weighted fusion"后单独追加，两次都各自补了 seed{1,2,3,42} 四组而非单 seed。
- **拆两个 commit 提交**：`[Data]` commit（`data/new_merge.dvc` + `data/.gitignore`）与 `[Experiment]` commit（contract/dvc/config/metrics）分开，让"数据集清理"和"实验结果"在 git 历史上可独立审查、独立 revert。

## 坑 / 已知问题

- **`dvc remove` 对 134GB/2350 文件的 hardlink 缓存目录耗时 15+ 分钟**：进程在此期间处于 `D`（disk sleep）状态，`/proc/<pid>/wchan` 显示 `folio_wait_bit_common`，是真实的磁盘 I/O 等待，不是卡死；纯粹等待即可，无需干预。
- **手动跑脚本（不走 `dvc repro`）不会自动生成 `artifacts/*/.gitignore`**：这个文件通常由 DVC stage 的 `outs:` 处理自动产生，本次因为是手动 `python scripts/build_contract.py`/`train_baseline_v0.py`，需要照着 `artifacts/contract_v1/.gitignore`、`artifacts/contract_new_ep1_expanded/.gitignore` 等同级目录的既有约定手工补建，否则后续 `git add` 会误图把大体积 parquet/json/bin 一起纳入追踪。
- **pre-commit `end-of-file-fixer` 会在 commit 时静默改写文件**：第一次提交 `[Experiment]` commit 时该 hook 自动给 `metrics.json` 补了缺失的末尾换行，导致那次 commit 未成功（已 staged 快照与磁盘不一致）。按项目 git 安全约定不能 `--amend`，重新 `git add` 该文件后新建一次 commit 才成功。

## 遗留 TODO

- L0/L1/L2/RG/DWF 六种融合方式已在 `new_merge` 上全部跑通多 seed 对比，均未超过 concat(L0)/DWF 的水平；若后续要在 `new_merge` 上继续做融合创新，增量来源应转向门控/加权机制之外的方向（如 endpoint×time-window 时序结构），呼应 entry 010 横切主题"简单门控条件融合收益有限"的既有判断。
- AnoMod 侧已补齐 `client_content_length_mean/rel_shift`、`client_body_hash_mismatch_rate` 派生特征，但 `endpoint_red_preprocessor`/`TracePreprocessor` 尚未接入，本轮 contract 未消费，仍是待办（见 `configs/data/new_merge.yaml` 注释）。
- `merged_v2`（`anomod_v1`+`endpoint_raw2`+`normal_v2`）因 `endpoint_raw2` 已被物理删除（commit `326b885`），其链路（根 `dvc.yaml` 的 v0/v1、`dvc_reliability_gate/`、`dvc_deviation_weighted/`）已无法从原始数据 `dvc repro` 重建，目前只是接受现状、未做任何标记式处理（沿用 entry 021/022 的既有判断，见 `CLAUDE.md` 本次同步新增的提示行）。
