# 设计 · 30/60 分钟重采 Normal 数据接入 DVC pipeline

- **日期**: 2026-07-13
- **分支**: `feature/multi-normal`
- **范围**: 替换 Normal 数据来源（数据层 + 配置层 + pipeline 层最小改动），不涉及模型/评估逻辑改造

## 1. 背景与目标

`data/anomod_v1/Normal/` 存在 cadvisor 断流问题：核实发现该 case 的 metric 原始数据最后一条采集记录停在 case 时间窗口开始前约 7 小时，窗口内 121 个 15s 时间桶**全部 0 覆盖**——不是部分缺失，是整个窗口内 metric 模态完全没有数据。这是 entry 006/007 中记录的"metric 模态全 NaN"问题在 Normal case 上的数据层根因，007 已修复代码层的 NaN 传染 bug，但数据层缺口本身需要重采数据才能解决。

`data/normal_0711_30/`、`data/normal_0711_60/` 是两次独立重采（分别 30 分钟、60 分钟 baseline），核实其 metric 覆盖率为窗口内 15s 时间桶 100% 覆盖（30 分钟数据 121/121 桶，60 分钟数据 240/240 桶）。

**目标**：用这两份重采数据完全替换 `anomod_v1/Normal` 作为训练/评估用的正常样本来源，同时扩充正常样本量（两个 case 合计约 61 分钟正常数据 vs 原来约 30 分钟）。

## 2. 探查确立的关键事实

1. **`anomod_v1/Normal` 与 `normal_0711_30/60` 是不同批次采集**：`case_metadata.json` 显示前者采集于 2026-06-29（`tt_max_workers=5`），后两者采集于 2026-07-11（`tt_max_workers=4`）。三者时间窗互不重叠，物理上是三次独立的 run。
2. **`normal_0711_30`/`normal_0711_60` 是"裸 case 目录"，不是"root"**：目录结构直接是 `api_responses/`、`trace_data/`、`metric_data/`、`_pipeline_out/` 等，等价于 `anomod_v1/Normal/` 这一层，而不是像 `anomod_v1/`、`endpoint_raw2/` 那样在自己下面再套一层具名 case 子目录。而 `_enumerate_cases()`（`scripts/build_contract.py:47-51`）依赖 `root.glob("*/_pipeline_out")` 找 case，直接把这两个目录填进 `roots` 会扫到 0 个 case。
3. **列结构差异不影响 pipeline 兼容性**：`normal_0711_30/60` 的 `tt_traces_red_15s.csv` 比 `anomod_v1` 多出 `endpoint_service`、`target_endpoint`、`anomaly_level`、`is_target_endpoint` 四列（008 引入的新字段）。`TracePreprocessor.transform()`（`src/preprocessors/trace_preprocessor.py:52-53`）已对 `is_target_endpoint` 缺失做兼容处理（缺失时填 `False`），无需改动代码。
4. **`normal_source` 字段目前只用于日志，不参与真正的过滤**：`build_contract.py:355` 判定 Normal case 的实际逻辑是 `full["anomaly_type"].str.startswith("Normal")`，扫描**全部** roots 下所有 case，与 `normal_source` 配置值无关。`dataset_cfg.normal_source` 仅出现在 `LOG.info(...)`（`build_contract.py:363`）里。这意味着仅仅改 `normal_source` 字段值不会排除旧 Normal case——必须让旧 Normal 目录物理上不再被 `_enumerate_cases` 扫到。
5. **`_enumerate_cases` 只扫 root 下一层**：验证过 `root.glob("*/_pipeline_out")` 会匹配任意名字的直接子目录（改名加下划线前缀无效），但不会递归到孙子目录。因此把某个 case 目录挪深一层（在 root 和 case 之间插入一层新目录）即可让它不被扫到，不用改代码或加黑名单逻辑。
6. **`data/anomod_v1/` 是项目里明确约定的 READ-ONLY 目录**（CLAUDE.md《Data Rules》），不能物理删除或覆盖其原始数据；因此排除旧 Normal 采用"归档移动"而非"删除"，保持可逆。
7. **`.gitignore` 中 `data/anomod_v1/` 被排除在版本控制外**（`.gitignore:58`），`endpoint_raw2`、`normal_0711_*` 均不在 `.gitignore` 里，按惯例以 untracked 状态存在于工作区（同 entry 006 的既有做法），本次不改动这一惯例。

## 3. 设计

### 3.1 数据层改动

**新建 wrapper 目录，移动两份重采数据进去**，使其满足 `_enumerate_cases` 对"root 下一层是具名 case 目录"的假设：

```
data/normal_v2/
├── normal_0711_30/   (原 data/normal_0711_30 整体移入)
└── normal_0711_60/   (原 data/normal_0711_60 整体移入)
```

**归档旧 Normal**，多套一层目录深度使其不再被扫描到，数据物理保留、可逆：

```
data/anomod_v1/Normal/  →  data/anomod_v1/_archive/Normal/
```

移动后 `anomod_v1/*/​_pipeline_out` 这一 glob 只会看到 `anomod_v1/_archive/_pipeline_out`（不存在），不会递归进 `_archive/Normal/`，故不需要任何黑名单/排除逻辑。

### 3.2 配置层改动

新建 `configs/data/merged_v2.yaml`（`configs/data/merged_v1.yaml` 保持不变，作为历史可复现快照）：

```yaml
name: merged_v2
description: anomod_v1 (11 个 service 级故障 case，Normal 已归档至 _archive/) + endpoint_raw2 (16 case) + normal_v2 (2 个重采 Normal case，30min+60min)
roots:
  - data/anomod_v1
  - data/endpoint_raw2
  - data/normal_v2
normal_source: data/normal_v2
fused_window: 15s
```

`normal_source` 字段的语义和 `DatasetConfig.__post_init__` 的校验逻辑（`normal_source ∈ roots`）均不改动——归档已经解决了"旧 Normal 混入训练集"的问题，`normal_source` 保持"仅用于日志提示"的现状即可，不升级为硬过滤依据（YAGNI：当前没有第二个需要被排除的 Normal 来源）。

### 3.3 Pipeline 层改动

`dvc.yaml` 的 `build_contract` 阶段：

- `cmd` 中 `--dataset configs/data/merged_v1.yaml` → `--dataset configs/data/merged_v2.yaml`
- `deps` 中 `configs/data/merged_v1.yaml` → `configs/data/merged_v2.yaml`

`train_v0`、`eval_v0` 两个阶段无需改动——它们消费 `build_contract` 的输出 parquet，对上游数据集切换透明。

### 3.4 不改的部分（明确排除范围）

- `build_contract.py`、`dataset_config.py`、`TracePreprocessor`、`Normalizer` 均无需代码改动。
- `is_endpoint_anomaly`/`label_granularity`（008 引入）逻辑不变：`normal_v2` 两个 case 均为 Normal（`inject_start_ms/end_ms` 为 `null`），`phase` 全程为 `"normal"`，不影响标签计算路径。
- 不改造 contract v0 的 schema/校验规则。

### 3.5 文档更新

- `CLAUDE.md`：更新数据集描述（当前"合并两个数据源共 28 个 case"需要改为三源合计 29 case：`anomod_v1` 11 + `endpoint_raw2` 16 + `normal_v2` 2），更新目录结构注释中 `anomod_v1`（含 Normal 已归档说明）、`normal_0711_30/60`（改为已并入 `normal_v2/`，不再是"尚未接入"状态）的描述。
- `history/entries/`：新增一篇 entry 记录本次切换的决策（为什么替换、为什么归档而非删除、`normal_source` 未做硬化的理由），并更新 `history/index.md` 的列表与影响域索引。

## 4. 测试

- `tests/test_dataset_config.py`：新增/调整 fixture，验证 `merged_v2.yaml` 能正确加载三个 root，且 `normal_source` 校验通过。
- 集成测试（`tests/test_e2e_smoke.py` 或同类）：验证切到 `merged_v2` 后：
  - Normal case 数为 2（均来自 `normal_v2/`）
  - `data/anomod_v1/_archive/Normal` 不出现在枚举结果中
  - 全量 case 数为 29（11 + 16 + 2）

## 5. 影响范围小结

| 类别 | 改动 |
|---|---|
| 数据（物理） | 移动 `normal_0711_30/60` 到 `data/normal_v2/` 下；归档 `anomod_v1/Normal` 到 `anomod_v1/_archive/Normal` |
| 配置 | 新增 `configs/data/merged_v2.yaml` |
| Pipeline | `dvc.yaml` 的 `build_contract` 阶段切换 `--dataset` 及 `deps` |
| 代码 | 无 |
| 文档 | `CLAUDE.md`、`history/index.md` + 新 entry |
| 测试 | `test_dataset_config.py` 新 fixture；集成测试新增断言 |
