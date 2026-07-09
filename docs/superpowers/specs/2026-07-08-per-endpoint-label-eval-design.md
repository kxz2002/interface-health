# 设计 · per-endpoint 精确标签接入 Contract v0 评估

- **日期**: 2026-07-08
- **分支**: `bugfix/fix-metric`（或另开新分支，视落地时机而定）
- **范围**: 承接 `entries/006-endpoint-raw2-multi-source.md` 遗留的③（per-endpoint 标签 + 评估），只做最小闭环；双标签 schema 前瞻、contract v1 正式升版、监督学习训练侧改造均不在本次范围

## 1. 背景与目标

PR #6（entry 006）把 `endpoint_raw2` 接入 Contract v0 pipeline 时，刻意用 case 级"脏"标签（`phase=='inject'` 时窗内所有 endpoint 都标异常），把"数据接入"和"标签体系重构"解耦，将后者列为遗留 TODO ③（design doc `2026-07-03-endpoint-raw-ingestion-design.md` §5）。

`endpoint_raw2` 的 `case_metadata.json` 带 `target_endpoint`，`tt_traces_red_15s.csv` 带 `is_target_endpoint` 布尔列，精确标记了每个 case 里哪个 endpoint 是真正的故障目标——这是论文核心创新点（per-endpoint × time-window 粒度检测）的关键素材，目前完全没有被使用。

**目标**：让这份精确标签从原始 CSV 一路穿透到 contract parquet、scores.parquet、最终的 `metrics.json`，产出真正意义上的 per-endpoint AUROC，同时保证：
1. 这份标签是**可复用的 contract 契约列**，不是某个 eval 脚本的局部技巧——未来任意新增的融合策略/检测器都通过读 contract 直接继承。
2. `anomod_v1` 和 `endpoint_raw2` 的评估走**同一套体系**（同一份标签列、同一批全部数据），不因数据源不同而分裂成两套指标口径。

## 2. 探查确立的关键事实

1. **`is_target_endpoint` 语义是"身份"不是"此刻状态"**：实测 `Lv_E_HTTPABORT_assurance` case，目标 endpoint 的 `is_target_endpoint` 在 baseline/inject/recover **全程**为 `True`。它标记"这个 endpoint 是本次故障注入的目标"，与"此刻是否异常"无关。因此精确异常标签必须是 `is_target_endpoint AND phase=='inject'`，不能直接用 `is_target_endpoint`。
2. **该字段只存在于 `tt_traces_red_15s.csv`，且在 `TracePreprocessor.transform` 被丢弃**：`tt_endpoint_health_15s.csv`（client 侧）没有这列；`tt_traces_red_15s.csv`（server 侧）有，但 `TracePreprocessor.OUTPUT_COLUMNS`（`src/preprocessors/trace_preprocessor.py:20`）目前只保留 5 个特征列，`transform()` 第52行的显式列选择把 `is_target_endpoint`/`target_endpoint`/`anomaly_level` 全部丢弃。
3. **`anomod_v1` 的 case 没有 `target_endpoint` 字段**：实测 `Lv_D_cachelimit` 的 `case_metadata.json`，`target_endpoint` 键不存在（`.get()` 返回 `None`），`anomaly_level` 取值是故障类型（如 `"database"`），不是 `"endpoint"`。因此不能靠 `anomaly_level=='endpoint'` 兼职判断"标签是否精确"——这是语义借用，未来某数据源若 `anomaly_level` 恰好等于 `"endpoint"` 但没有精确字段会失效。
4. **`is_anomaly` 已被 contract v0 强校验锁定为 case 级**：`contract_v0.py:76` 强制 `is_anomaly == (phase=='inject')`，本次不修改这条校验，新标签走增量列。
5. **`scores_v0.py` 契约只强制 `sample_id`/`score`/`y_true` 三列，其余列不校验**；`metrics_v0.py` 的 `stratified` 字段属于契约里"允许任何额外字段"的可选部分，加 `by_endpoint` 键不需要改契约文件。
6. **`train_baseline_v0.py` 的 `out_df` 是显式字段列表**（第149-160行），不会自动继承 `eval_all.parquet` 的新列——这正是当年 `is_target_endpoint` 在 `build_contract.py` 里被静默丢弃的同一种错误模式，必须显式加进字段列表。

## 3. 设计

### 3.1 数据流转路径

```
tt_traces_red_15s.csv (含 is_target_endpoint)
  → TracePreprocessor.transform（新增保留该列，缺列时填 False 占位）
  → ep_df（merge 后逐行携带 is_target_endpoint）
  → _attach_label_columns（依据 case_meta 是否有 target_endpoint，判定 label_granularity；
                            计算 is_endpoint_anomaly = is_target_endpoint & phase=='inject'，
                            fallback 时 = is_anomaly）
  → contract parquet（新增 label_granularity, is_endpoint_anomaly 两列）
  → train_baseline_v0.py（显式带入 out_df）
  → scores.parquet（新增同两列，y_true 不变）
  → eval_baseline_v0.py（四层 stratified 全部改用 is_endpoint_anomaly 计算）
  → metrics.json（新增 stratified.by_endpoint，不改契约）
```

### 3.2 `TracePreprocessor` 改动

`src/preprocessors/trace_preprocessor.py`：

- `OUTPUT_COLUMNS` 新增 `"is_target_endpoint"`。
- `transform()` 在原始 `df` 缺该列时（`anomod_v1` 场景）显式填 `False` 占位，再纳入返回的 `df[[...]]` 选择，避免 `KeyError`。这一改动**不引入任何新判定逻辑**，纯粹是让同一份代码兼容两种输入 schema；真正决定"该 case 走精确标签还是 fallback"的判断发生在 `_attach_label_columns`，依据 `case_meta.get("target_endpoint")`，不依据这个占位列的值。

### 3.3 Contract 层改动

`scripts/build_contract.py:_attach_label_columns` 新增：

```python
if case_meta.get("target_endpoint") is not None:
    ep_df["label_granularity"] = "endpoint"
    is_target = ep_df["is_target_endpoint"].fillna(False)
    ep_df["is_endpoint_anomaly"] = is_target & (ep_df["phase"] == "inject")
else:
    ep_df["label_granularity"] = "case"
    ep_df["is_endpoint_anomaly"] = ep_df["is_anomaly"]
```

同一 case 内非目标 endpoint 的行同样标 `label_granularity="endpoint"`（因为它们同样能被准确判定"不是目标"，`is_endpoint_anomaly` 对它们自然是 `False`）。

`src/contracts/contract_v0.py`：`REQUIRED_LABEL_COLUMNS` 新增 `"label_granularity"`、`"is_endpoint_anomaly"` 两项，纯增量；不改动第76行现有的 `is_anomaly` 强校验。

### 3.4 `train_baseline_v0.py` 改动

`out_df` 构造（第149-160行）显式新增两行：

```python
"is_endpoint_anomaly": eval_df["is_endpoint_anomaly"].astype(int),
"label_granularity": eval_df["label_granularity"],
```

`y_true` 继续来自 `is_anomaly`，不变，满足 `scores_v0.py` 契约。

### 3.5 `eval_baseline_v0.py` 改动

`compute_stratified_metrics` 四层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增 `by_endpoint`）**全部**改用 `is_endpoint_anomaly` 而非 `y_true` 计算 AUROC/AUPRC：

```python
stratified["by_endpoint"] = {}
for ep, grp in df.groupby("endpoint_key"):
    stratified["by_endpoint"][ep] = {
        "auroc": _safe_auroc(grp["is_endpoint_anomaly"].values, grp["score"].values),
        "n_samples": len(grp),
    }
```

`by_endpoint` 对**全部数据**分组（不按 `label_granularity` 过滤），`anomod_v1` 和 `endpoint_raw2` 的行在同一分组、同一标签列下参与计算——两个数据源不再分裂成两套评估口径。`label_granularity` 列本次只落盘到 contract/scores parquet，供日后按需分析各分组标签噪声构成，`metrics.json` 本次不新增噪声占比字段（如 `n_precise`），留作遗留 TODO。

顶层 `overall`/`auroc`/`auprc` 字段（第55-59行的 `has_labels` 判定）同样改用 `is_endpoint_anomaly`。

### 3.6 文档

`docs/agent-docs/dataset-guide.md` 补记：
- `target_endpoint`/`is_target_endpoint`/`anomaly_level`（endpoint 场景下）字段口径，目前该文档零记录。
- Contract v0 新增列 `label_granularity`/`is_endpoint_anomaly` 的语义与产出规则。

## 4. 组件边界

- `TracePreprocessor`：只负责把原始列忠实传递到 `ep_df`，不做粒度判断。改动限于 `OUTPUT_COLUMNS` 和缺列占位，不影响其余 4 个特征列的行为。
- `_attach_label_columns`（`build_contract.py`）：唯一的"标签粒度判断"发生地，依据 `case_meta`（case 级信息）+ `ep_df["is_target_endpoint"]`（行级信息）产出两列，其余 pipeline 步骤不重复这个判断。
- `eval_baseline_v0.py`：只读 contract 已产出的 `is_endpoint_anomaly`/`label_granularity`，不重新计算标签，保持 train↔eval 契约解耦。

## 5. 测试策略

1. `test_trace_preprocessor.py`：新增 case 覆盖 `is_target_endpoint` 列存在/不存在两种输入，断言输出列包含该字段且缺列时填 `False`。
2. `test_build_contract_*`（扩现有 fixture）：
   - 有 `target_endpoint` 的 case：断言目标 endpoint 在 inject 窗口内 `is_endpoint_anomaly=True`、baseline/recover 窗口 `False`；非目标 endpoint 全程 `False`；`label_granularity` 全部为 `"endpoint"`。
   - 无 `target_endpoint` 的 case（如 anomod_v1 fixture）：断言 `is_endpoint_anomaly == is_anomaly`，`label_granularity` 全部为 `"case"`。
3. `test_contract_v0_schema.py`：断言 `REQUIRED_LABEL_COLUMNS` 包含新两列，缺列时 `validate_contract_df` 报错。
4. `test_train_baseline_v0.py`：断言 `scores.parquet` 输出包含 `is_endpoint_anomaly`/`label_granularity`，值与输入 `eval_all.parquet` 一致（防止显式字段列表漏加的回归）。
5. `test_eval_baseline_v0_stratified.py`：
   - 构造混合 `label_granularity` 的 mini DataFrame，断言四层全部用 `is_endpoint_anomaly` 而非 `y_true` 计算（构造 `is_anomaly` 与 `is_endpoint_anomaly` 取值不同的行，验证输出数值对应后者）。
   - 断言 `by_endpoint` 按 `endpoint_key` 分组，不因 `label_granularity` 不同而拆分成两组。

不测：`metrics.json` 噪声占比字段（未实现，留 TODO）；真实 `endpoint_raw2` 全量数据跑 `dvc repro`（留手动收尾验证）。

## 6. 落地流程

改动文件清单：

| 文件 | 改动 |
|---|---|
| `src/preprocessors/trace_preprocessor.py` | `OUTPUT_COLUMNS` 加列，缺列占位 |
| `scripts/build_contract.py` | `_attach_label_columns` 新增两列计算 |
| `src/contracts/contract_v0.py` | `REQUIRED_LABEL_COLUMNS` 新增两项 |
| `scripts/train_baseline_v0.py` | `out_df` 显式新增两列 |
| `scripts/eval_baseline_v0.py` | 四层全部改用 `is_endpoint_anomaly`，新增 `by_endpoint` |
| `docs/agent-docs/dataset-guide.md` | 补记字段口径 |
| `tests/test_trace_preprocessor.py` | 新增用例 |
| `tests/test_build_contract_*.py` | 扩现有 fixture |
| `tests/test_contract_v0_schema.py` | 断言新列 |
| `tests/test_train_baseline_v0.py` | 断言新列穿透 |
| `tests/test_eval_baseline_v0_stratified.py` | 断言四层口径统一 + `by_endpoint` |

收尾验证：
1. `pytest tests/` 全绿。
2. `dvc repro` 全链路跑通（或至少 `train_baseline_v0.py` + `eval_baseline_v0.py` 手动跑通），`metrics.json` 里 `stratified.by_endpoint` 非空、数值非 NaN。
3. 抽查若干 `endpoint_raw2` case，确认目标 endpoint 的 `is_endpoint_anomaly` 只在 inject 窗口为 True。
4. 写 history entry（编号承接 007 之后）+ 更新 index，把 §7 遗留 TODO 留痕。

## 7. 遗留 TODO（留给后续 PR）

- **`metrics.json` 噪声构成字段**：`by_endpoint`（以及其他三层）本次不报"该分组里精确标签/case 级近似标签各占多少行"，`label_granularity` 只落盘不汇总。后续若要在论文里量化标签噪声对 AUROC 的影响，需要在 `metrics.json` 加类似 `n_precise`/`n_fallback` 的字段。
- **双标签 schema 支持监督学习训练侧前瞻**：本次 `is_anomaly`（case 级）与 `is_endpoint_anomaly`（endpoint 级，含 fallback）已经并存，满足"两套标签列同时可用"，但训练侧（`build_contract.py` 的 train/eval split，目前固定只用 Normal 做训练）尚未参数化为"全量带标签数据进训练"，做监督学习对比前需要单独设计这部分。
- **Contract v0 → v1 正式版本升级**：本次全部改动走增量列，未触碰 `is_anomaly` 背后的 case 级假设和现有强校验。若未来要清理这条历史设计（如允许 `is_anomaly` 本身细化到 endpoint 级），需要走 contract v1 迁移，而不是继续叠加 v0 之上的增量列。
- **`label_granularity` 作为下游路由规则的复用验证**：未来新增特征融合策略（消融实验）或新的下游检测器（如 KMeans）横向对比时，需要验证它们的 eval 流程能否直接复用本次约定的四层 `stratified` 结构与 `is_endpoint_anomaly` 标签列，而不需要重新发明标签口径判断逻辑。
