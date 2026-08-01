# 023 · new_merge 数据集接入 + fraction 校准 + KILLPOD_gateway 剔除 + 首次 L0 基线

- **日期**: 2026-07-31
- **PR**: 待开 PR（branch: `exp/new-merge-baseline-rerun`）
- **类型**: Data + Experiment
- **影响域**: `data/new_merge.dvc`, `data/.gitignore`, `configs/data/new_merge.yaml`, `configs/contract/v1_new_merge.yaml`, `dvc_new_merge/dvc.yaml`, `artifacts/contract_new_merge_expanded/`, `artifacts/baseline_new_merge_concat/`

## 做了什么

接入 `new_merge`（AnoMod 侧新采集的单一批次数据集，2026-07-28/29，原始 28 case）为独立数据源，不与 `anomod_v1`/`endpoint_raw2`/`new_ep1` 等历史批次混合（entry 017 已证实跨 run 合并会污染评估）。沿用 `dvc_new_ep1/dvc.yaml` 的隔离模式，新建 `dvc_new_merge/dvc.yaml`（`build_contract_new_merge_expanded` → `train_new_merge_concat` → `eval_new_merge_concat`），裸 `dvc repro` 不触发。本轮只跑 `fusion=concat`（L0），不含 DWF/RG，后续按计划继续补充。

`fault_baseline_train_fraction` 未直接借用其他数据集的已校准数值，而是实测对比 `1.0`（借用 `new_ep1`）与 `0.2`（借用 `endpoint_raw2`/`v1_expanded_pool`）两个取值下 `eval_all` 的实际类别比例，选定 `0.2`。过程中发现 `Lv_S_KILLPOD_gateway` 在 `eval_all` 里 100% 落在 `phase=='baseline'`（462 行，单一类别），`by_anomaly_type` 分层 AUROC 数学上无定义（`null`）——与 entry 016 在 `endpoint_raw2` 上遇到的同名 case 是完全相同的结构性缺陷（`ts-gateway-service` 既是 PodChaos 目标又是 SkyWalking trace 上报通道，pod 被杀后 inject 阶段 trace 永久缺失），非本次切分逻辑的 bug。确认该 case 未进入任何已提交历史（`git log --all -- data/new_merge.dvc` 为空）后，物理删除该 case 目录，重新 `dvc add` 生成 27-case 版本，重跑 contract 构建 + L0 训练评估。

清理前后对比：overall AUROC 0.718→0.742，`eval_all` 从 16271→15809 行（少 462），正负比 19.7:80.3（3113 正 / 12696 负）。清理后 `Lv_S_KILLPOD_gateway` 从 `by_anomaly_type` 中完全消失，其余分层同步小幅提升（去掉一个标签无效的 case，稀释效应消除）。

## 关键决策（不在 commit 里）

- **fraction 必须实测而非借用**：即使 `new_ep1` 与 `new_merge` 的 `baseline_sec` 量级接近（1500s vs 1200s），两者的 `fault_baseline_total` 绝对行数和其他负样本项（fan-out 残留、Normal holdout）的相对大小不同，`fraction` 的实际效果只能通过实测 `eval_all` 的类别分布验证，不能靠 `baseline_sec` 类比推断。`1.0` 在 `new_merge` 上会让故障 case 的 baseline 阶段行全部被吸收，负样本地板被削到只剩 fan-out 残留（正负比严重偏正，不可用）；`0.2` 产出的绝对负样本行数（12696~13158）和比例都在entry 016 校准目标附近，故选 `0.2`。
- **剔除 KILLPOD_gateway 而非保留 null 分层**：曾考虑保留该 case 只是让 `by_anomaly_type` 该项显示 `null`（不参与宏平均），但用户判断"AUROC 数学上无定义"的 case 不该留在数据集里污染"数据集干净度"，且 entry 016 已有先例证明这是可重复出现的已知模式，非一次性意外。删除前核实了两件事：(1) 清理前的 `data/new_merge.dvc` 从未进入过任何历史 commit，删除零风险；(2) 当时下游所有 contract/baseline 产物均未提交，重跑无兼容性负担。
- **本轮范围只跑 concat(L0)**：不在本次一起补 DWF/RG，避免把"数据集接入+清理"和"融合机制对比"两件事耦合在一次改动里；DWF/RG 留给后续独立实验迭代。
- **拆两个 commit 提交**：`[Data]` commit（`data/new_merge.dvc` + `data/.gitignore`）与 `[Experiment]` commit（contract/dvc/config/metrics）分开，让"数据集清理"和"实验结果"在 git 历史上可独立审查、独立 revert。

## 坑 / 已知问题

- **`dvc remove` 对 134GB/2350 文件的 hardlink 缓存目录耗时 15+ 分钟**：进程在此期间处于 `D`（disk sleep）状态，`/proc/<pid>/wchan` 显示 `folio_wait_bit_common`，是真实的磁盘 I/O 等待，不是卡死；纯粹等待即可，无需干预。
- **手动跑脚本（不走 `dvc repro`）不会自动生成 `artifacts/*/.gitignore`**：这个文件通常由 DVC stage 的 `outs:` 处理自动产生，本次因为是手动 `python scripts/build_contract.py`/`train_baseline_v0.py`，需要照着 `artifacts/contract_v1/.gitignore`、`artifacts/contract_new_ep1_expanded/.gitignore` 等同级目录的既有约定手工补建，否则后续 `git add` 会误图把大体积 parquet/json/bin 一起纳入追踪。
- **pre-commit `end-of-file-fixer` 会在 commit 时静默改写文件**：第一次提交 `[Experiment]` commit 时该 hook 自动给 `metrics.json` 补了缺失的末尾换行，导致那次 commit 未成功（已 staged 快照与磁盘不一致）。按项目 git 安全约定不能 `--amend`，重新 `git add` 该文件后新建一次 commit 才成功。

## 遗留 TODO

- DWF/RG 尚未在 `new_merge` 上跑，按用户计划后续继续补充实验（本条目范围只锁定 concat(L0)）。
- AnoMod 侧已补齐 `client_content_length_mean/rel_shift`、`client_body_hash_mismatch_rate` 派生特征，但 `endpoint_red_preprocessor`/`TracePreprocessor` 尚未接入，本轮 contract 未消费，仍是待办（见 `configs/data/new_merge.yaml` 注释）。
- `merged_v2`（`anomod_v1`+`endpoint_raw2`+`normal_v2`）因 `endpoint_raw2` 已被物理删除（commit `326b885`），其链路（根 `dvc.yaml` 的 v0/v1、`dvc_reliability_gate/`、`dvc_deviation_weighted/`）已无法从原始数据 `dvc repro` 重建，目前只是接受现状、未做任何标记式处理（沿用 entry 021/022 的既有判断，见 `CLAUDE.md` 本次同步新增的提示行）。
