# 008 · per-endpoint 精确标签接入评估

- **日期**: 2026-07-11
- **PR**: feature/per-endpoint-label-eval（待合并） · **Commit**: 0fb022c
- **类型**: Feature
- **影响域**: `src/preprocessors/trace_preprocessor.py`, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `scripts/eval_baseline_v0.py`, `src/contracts/contract_v0.py`

## 做了什么

承接 entry 006 遗留 TODO③：让 `endpoint_raw2` 的 `is_target_endpoint` 精确标签从原始 CSV 一路穿透到 `metrics.json`，产出真正意义上的 per-endpoint AUROC。新增两个 contract 列：`label_granularity`（`"endpoint"` = 精确标签 / `"case"` = case 级近似 fallback）、`is_endpoint_anomaly`（= `is_target_endpoint AND phase=='inject'`；fallback 时等于 `is_anomaly`）。`eval_baseline_v0.py` 的四层分层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增 `by_endpoint`）全部统一改用 `is_endpoint_anomaly`，不因数据源不同分裂成两套评估口径。

## 关键决策（不在 commit 里）

- **新增列而不是改 `is_anomaly` 语义**：`is_anomaly`/`y_true` 保持 case 级语义不变，满足 `scores_v0.py` 契约与既有强校验（`contract_v0.py` 的 `is_anomaly == phase=='inject'`），新标签走纯增量列，不需要动 contract v0 现有校验规则。
- **`by_endpoint` 对全部数据统一分组，不按 `label_granularity` 过滤**：最初设计考虑过"精确标签行才算 by_endpoint"、"双轨 AUROC 分开报告"等方案，最终否定——因为下游会有多个融合策略消融 + 多个检测器横向对比，标签可信度必须是 contract 层的路由规则（`label_granularity` 列），不能是 eval 脚本里的临时过滤逻辑，否则每新增一个下游组合都要重新决策一次。
- **`label_granularity` 判定依据是 case 是否有 `target_endpoint` 字段，不是 `anomaly_level`**：`anomaly_level` 是故障类型描述（如 `"endpoint"`/`"performance"`/`"database"`），恰好 `endpoint_raw2` 的 `anomaly_level` 取值是 `"endpoint"`，但这是巧合不是语义保证，用它判断标签精度会在未来数据源变化时失配。
- **`is_target_endpoint` 必须 AND `phase=='inject'`**：实测确认该字段在原始 CSV 里全程（baseline/inject/recover）为同一个值，标记的是"身份"不是"此刻状态"，直接用会把该 endpoint 的正常时段也误标为异常。

## 坑 / 已知问题

- **`train_baseline_v0.py` 的 `out_df` 是显式字段字典**：新增列不会自动从 `eval_all.parquet` 带过去，必须手动加进字段列表——这是当年 `is_target_endpoint` 在 `build_contract.py` 里被静默丢弃的同一种模式，本次专门写了 `test_train_baseline_v0_scores_carry_endpoint_label_columns` 做回归防线。
- **`compute_stratified_metrics` 无条件读取 `is_endpoint_anomaly`，但 `scores_v0.py` 契约未把它列为必需列**：code review 发现如果直接对一份历史遗留（本次改动之前生成）的 `scores.parquet` 跑 `eval_baseline_v0.py`，会因缺列抛出原始 `KeyError` 而非清晰的契约错误。本次范围内新生成的 `scores.parquet` 都会带这两列，不受影响，但重跑旧 artifact 前需要先用新版 `build_contract.py` + `train_baseline_v0.py` 重新生成。

## 遗留 TODO

- `metrics.json` 未新增噪声构成字段（如 `n_precise`/`n_fallback`），`label_granularity` 本次只落盘不汇总，若后续需要在论文里量化标签噪声对 AUROC 的影响，需要单独设计。
- 训练侧仍是纯 One-Class（`build_contract.py` 的 train split 固定只用 Normal），双标签并存（`is_anomaly` + `is_endpoint_anomaly`）已满足监督学习对比的标签前提，但 train/eval split 逻辑尚未参数化为"全量带标签进训练"。
- Contract v0 → v1 正式版本升级仍未做，本次全部改动是 v0 之上的增量列。
- `scores_v0.py` 契约的 `REQUIRED_COLUMNS` 未同步新增 `is_endpoint_anomaly`/`label_granularity`（它们目前是可选的额外列）；若未来要让 `eval_baseline_v0.py` 对缺列场景报出清晰错误而非裸 `KeyError`，需要评估是否该把这两列纳入契约必需列，或在 `compute_stratified_metrics` 里加显式校验。
