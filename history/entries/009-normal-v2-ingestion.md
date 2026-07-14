# 009 · 30/60 分钟重采 Normal 数据接入 DVC pipeline

- **日期**: 2026-07-13
- **PR**: feature/multi-normal（待合并） · **Commit**: 56c82ec
- **类型**: Data
- **影响域**: `configs/data/`, `dvc.yaml`, `dvc.lock`, `artifacts/`, `tests/fixtures/merged_v2_mini/`, `tests/test_dataset_config.py`, `tests/test_build_contract_multi_root.py`, `tests/test_e2e_smoke.py`, `CLAUDE.md`

## 做了什么

用两份重采 Normal 数据（`data/normal_0711_30`、`data/normal_0711_60`，30min+60min）完全替换 `anomod_v1/Normal` 作为训练/评估用的正常样本来源。新建 `data/normal_v2/` wrapper 目录承载两份重采数据，满足 `_enumerate_cases` 对"root 下一层是具名 case 目录"的假设；归档旧 `anomod_v1/Normal` 到 `anomod_v1/_archive/Normal`，多套一层目录深度使其不再被 `_enumerate_cases` 扫到（数据物理保留，可逆，符合 `anomod_v1` 的 READ-ONLY 约束）。

新建 `configs/data/merged_v2.yaml`（三 root：`anomod_v1` + `endpoint_raw2` + `normal_v2`，`normal_source: data/normal_v2`），`dvc.yaml` 的 `build_contract` 阶段切换过去。`merged_v1.yaml` 保留不动，作历史快照。无任何 `src/`/`scripts/` 生产代码改动；新增测试覆盖（`tests/test_dataset_config.py` 的 `test_load_three_root_config_with_wrapper_normal_source`、`tests/test_build_contract_multi_root.py` 的 `test_enumerate_cases_multi_root_archived_case_not_scanned` 及 `test_enumerate_cases_multi_root_archiving_does_not_exclude_other_siblings`（归档机制不误伤同级有效 case）、`tests/test_e2e_smoke.py` 新增三个 e2e 用例）验证三 root 配置解析、归档深度排除、以及基于 `merged_v2_mini` fixture 的端到端 case 数量断言。同批次还同步更新了 `CLAUDE.md` 的数据集描述，反映三源合并后的 29 case 组成。

切换到 `merged_v2` 后重跑 `dvc repro`（`build_contract` → `train_v0` → `eval_v0`），`artifacts/baseline_v0/metrics.json` 的 AUROC 从 0.505（旧 Normal，metric 断流）提升到 0.661（AUPRC 0.320），验证新 Normal 数据对模型训练有实质帮助。产物见 `dvc.lock`、`artifacts/baseline_v0/metrics.json`、`artifacts/metrics.json`（commit `ffaf576`）。

## 关键决策（不在 commit 里）

- **为什么归档而非删除**：`anomod_v1/` 是 CLAUDE.md 明确约定的 READ-ONLY 目录；删除违反该约束且不可逆。归档（移动到更深一层目录）既排除了旧 Normal 参与训练，又保持数据物理可追溯、可逆。
- **为什么不硬化 `normal_source` 为真过滤条件**：探查发现 `normal_source` 字段目前只用于日志（`build_contract.py` 里判定 Normal 的真实逻辑是全局扫描 `anomaly_type.str.startswith("Normal")`，与 `normal_source` 值无关）。本次没有把它升级为真过滤依据，因为归档已经从数据层解决了"旧 Normal 混入"的问题，升级为硬过滤在当前只有一个待排除来源的情况下是过度设计（YAGNI）。
- **为什么新建 `merged_v2.yaml` 而不改 `merged_v1.yaml`**：`merged_v1.yaml` 被 `dvc.lock` 引用，代表一个已跑过的历史数据集组成快照；直接修改会破坏历史实验的可复现性。
- **为什么用"多套一层目录"而不是"改名加下划线前缀"来排除旧 Normal**：验证过 `_enumerate_cases` 用的 `root.glob("*/_pipeline_out")` 会匹配任意名字的直接子目录（不区分是否有下划线前缀），只有真正增加一层目录深度才能让它被跳过。

## 坑 / 已知问题

- **两批 Normal 采集参数不同**：`normal_0711_30/60` 与 `anomod_v1/Normal` 是不同批次采集（前者 2026-07-11，后者 2026-06-29，`tt_max_workers` 也不同：4 vs 5），三者时间窗互不重叠，物理上是独立的 run——本次替换认为这对"正常基线"是可接受的（正常运行状态不依赖具体采集批次），但如果未来发现 `tt_max_workers` 影响特征分布，需要重新评估。
- **重采数据的 trace CSV 多 4 列**：`normal_0711_30/60` 的 `tt_traces_red_15s.csv` 比 `anomod_v1` 多 4 列（`endpoint_service`/`target_endpoint`/`anomaly_level`/`is_target_endpoint`，entry 008 引入的字段）。`TracePreprocessor.transform()` 已对 `is_target_endpoint` 缺失做兼容处理，无需改代码，但如果未来这类新增字段被真正用作特征输入，需要重新检查所有历史 case 是否都有该字段。

## 遗留 TODO

- `normal_source` 字段仍然只是日志用途、不参与真实过滤——如果未来出现第二个需要排除的 Normal 来源，应该考虑把它升级为真正的过滤条件（当前 YAGNI 判断可能需要重新评估）。
