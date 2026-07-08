# 设计 · endpoint_raw 接入 Contract v0 pipeline（多数据源合并）

- **日期**: 2026-07-03
- **分支**: `feature/add-endpoint`
- **范围**: 本次 PR 只做 ①（数据接入 + 名单修正 + 多数据源参数化）；③（per-endpoint 标签+评估）拆为独立后续 PR

## 1. 背景与目标

项目要把两个数据集合并成**一个统一数据集**，供 Contract v0 pipeline（build_contract → train_v0 → eval_v0）使用：

- **anomod_v1**（41GB，12 case）：1 个 Normal + 11 个 **service 级**故障注入 case。log 已修复、时间覆盖完整。
- **endpoint_raw**（87GB，16 case）：全部为 **endpoint 级**故障注入（`Lv_E_HTTP{ABORT,DELAY,PATCH,REPLACE}_{assurance,order,travel,travel2}`），**无 Normal case**。每个 case 的 `case_metadata.json` 带 `target_endpoint`，`tt_traces_red_15s.csv` 带 `is_target_endpoint` 布尔列。

**目标**：让 `build_contract.py` 能同时吃两个数据源，合并产出一份统一 contract parquet，Normal 只来自 anomod_v1（endpoint_raw 无 Normal），跑通 `dvc repro`，得到合理的 One-Class 评估 AUROC/AUPRC。

**本次 PR 明确不做**：per-endpoint 精确标签、per-endpoint 评估、contract schema 变更、监督学习支持、`dvc add` 数据追踪、log 重采。这些拆入后续 PR。

## 2. 探查确立的关键事实（决策依据）

以下均经亲手复核（纠正过 subagent 的两处错误：CST/UTC 时区误判、inner-join 命中误判）：

1. **两个数据集目录结构一致**：均为 `trace_data/` + `metric_data/` + `api_responses/` + `log_data/` + `_pipeline_out/`，`_enumerate_cases` 以 `_pipeline_out/` 为锚点，天然兼容。
2. **endpoint_to_service.yaml 是存量 bug**：当前 8 个映射是 route/travelplan 组，**与真实数据 client 侧入口对不上**——其中 4 个（route + travelplan×3）在两个数据集 client 侧都不出现（死条目），导致 anomod_v1 baseline 实际只有 4 个活 endpoint。`docs/agent-docs/dataset-guide.md §6` 记录的真实 client 8 个入口才是正确口径。
3. **client 侧 endpoint 覆盖**（inner join 的硬约束）：endpoint 特征需 client（`tt_endpoint_health`）× server（`tt_traces_red`）按 `(endpoint_key, timestamp_window)` inner join。
   - anomod_v1 client 侧 8 个 endpoint = dataset-guide §6 记录的 8 个。
   - endpoint_raw client 侧 6 个 endpoint，是 anomod_v1 那 8 个的**严格子集**（仅 endpoint_raw 独有 = 0）。
   - endpoint_raw 全部 16 个 case 的 `target_endpoint` 在 inner join 后**都存活**。
4. **修正名单对 anomod_v1 的实际影响可控**：把名单改为真实 8 个后，anomod_v1 实际多进约 1994 行（assurance 944 + contact/{uuid} 945 + login 105），其余 server 侧内部调用 endpoint 因 client 侧无数据被 inner join 自动挡掉。这些新增行都是双侧齐全的干净外部入口，不引入特征异构。
5. **无历史 baseline 包袱**：数据集尚未定型，baseline 本就要在新数据集上重跑，名单修正不需向后兼容。
6. **endpoint_raw log 崩溃未修复**：全部 16 case 因 `fsnotify: too many open files` 在 baseline 早期崩溃，inject/recover 零日志覆盖。本次不处理，走现有 left-join → NaN 填 0 降级路径（pipeline 不崩，log 特征近似常量、无贡献）。

## 3. 设计

### 3.1 数据集配置层（方案 B）

新建 `configs/data/merged_v1.yaml`，声明合并组成：

```yaml
name: merged_v1
description: anomod_v1 (service 级) + endpoint_raw (endpoint 级) 合并数据集
roots:
  - data/anomod_v1
  - data/endpoint_raw
fused_window: 15s
normal_source: data/anomod_v1   # Normal case 只来自这里（endpoint_raw 无 Normal）
```

`build_contract.py` 入口从 `--data-root <单路径>` 改为 `--dataset <config路径>`。脚本读 config，对 `roots` 里每个目录分别 `_enumerate_cases` 再合并，其余流程（concat → 归一化 fit-on-Normal → 校验 → 写 parquet）不变。

**决策**：选方案 B 而非 A（`--data-root nargs='+'`）或 C（多 stage + merge）。
- B 把"数据集由哪些 root 组成"沉淀成受版本控制的 config，补上 CLAUDE.md "数据集版本记录在 config" 一直没落实的约定。
- 否决 C：它要求归一化拆成两段，与现有"fit 在 concat 之后统一做"的架构冲突，改动最大、最易出 bug。
- 否决 A：把数据集组成藏在命令行里，不利于版本追溯。

**决策**：直接切到 `--dataset`，不保留 `--data-root` 旧入口（项目无外部调用者，双入口是负担）。现有从未被引用的 `configs/data/anomod.yaml` 一并改造成同格式（`roots: [data/anomod]`），使其名副其实。

### 3.2 endpoint 名单修正

`configs/contract/endpoint_to_service.yaml` 整体改为 dataset-guide §6 的真实 8 个 client 入口 + service 映射：

```yaml
"GET:/api/v1/assuranceservice/assurances/types": ts-assurance-service
"GET:/api/v1/contactservice/contacts/account/{uuid}": ts-contacts-service
"POST:/api/v1/inside_pay_service/inside_payment": ts-inside-payment-service
"POST:/api/v1/orderservice/order/refresh": ts-order-service
"POST:/api/v1/preserveservice/preserve": ts-preserve-service
"POST:/api/v1/travel2service/trips/left": ts-travel2-service
"POST:/api/v1/travelservice/trips/left": ts-travel-service
"POST:/api/v1/users/login": ts-auth-service
```

- **service 名以数据 `endpoint_service` 列为准，排除 gateway 视角**：写代码前跑一次核对，用 endpoint 真实归属 service（数据里同一 endpoint 在 gateway 视角下会显示 `ts-gateway-service`，取真实归属那条）。上表 service 名为待核对初值。
- **不改"一份文件两用"结构**：该文件同时用作过滤白名单（`.isin()`）和 endpoint→service 映射，现状够用，只修正内容（YAGNI）。

### 3.3 Normal case 与归一化（不改代码，确认成立）

现有逻辑（`build_contract.py:307-316`）合并后天然正确：`normal_mask = anomaly_type.startswith("Normal")` 只匹配 anomod_v1 的 Normal，`normalizer.fit(full[normal_mask])` 只在 Normal 上 fit，endpoint_raw 的异常 case 只进 transform。

- endpoint_raw 的 6 个 endpoint ⊂ anomod_v1 Normal 的 8 个，per_endpoint 归一化下每个 endpoint 都能取到 Normal 统计量，无 endpoint 落空。
- 数值特性（非 bug，需知晓）：用 anomod_v1 Normal 统计量归一 endpoint_raw 数据，隐含"两数据集正常基线一致"假设。rate 列有 `clip(0,1)` 兜底；非 rate 列（如 latency）不裁剪，偏离正常可能出现负值或 >1，这是 One-Class 里"偏离即异常"的正常体现。

### 3.4 标签与 schema（本次不动，划界）

现有标签逻辑（`build_contract.py:211-231`）合并后照常工作：`phase`/`is_anomaly` 由 `inject_start_ms`/`inject_end_ms` 三段式推导，endpoint_raw 同格式直接生效；`target_service` 照读。

**本次明确不做**：
- 不用 `is_target_endpoint`。`TracePreprocessor:52` 继续只取 5 个特征列，`is_target_endpoint`/`target_endpoint`/`anomaly_level` 继续被丢弃。**endpoint_raw 本次仍用 case 级"脏"标签**（inject 窗内该 case 所有 endpoint 都标异常，与 anomod_v1 一致）。
- 不改 contract v0 schema，`REQUIRED_LABEL_COLUMNS` 不变，校验器不动。

**接受的事实**：endpoint_raw 的 per-endpoint 精确标签优势本次用不上，要等 ③（见 §5 遗留）用上 `is_target_endpoint` 后才兑现。这样切是为把"数据接入"和"标签体系重构"解耦，避免 PR 膨胀。

### 3.5 测试策略（TDD）

新增/修改：
1. `test_load_dataset_config`：解析 `merged_v1.yaml`，取出 `roots`/`normal_source`，字段缺失显式报错（非 silent）。逻辑抽成 `src/data/dataset_config.py` 独立小模块以便单测。
2. `test_enumerate_cases_multi_root`：两个含 `_pipeline_out/` 的临时目录，验证合并枚举出全部 case、sorted 稳定、无重复。
3. `test_endpoint_mapping_matches_dataset_guide`：断言 `endpoint_to_service.yaml` 的 key 集合 == dataset-guide §6 的 8 个真实 client endpoint。**把"配置与真实数据口径漂移"这个坑钉死的回归防线。**
4. 合并 e2e smoke（扩现有 `test_e2e_smoke.py`）：造 1-2 个 endpoint_raw mini fixture case，验证两个 mini root 合并跑通 build_contract、parquet 过 contract 校验、Normal 来自 anomod 侧、endpoint_raw case 都进 eval。
5. `test_build_contract_stage_uses_dataset_config`（扩 `test_dvc_pipeline_v0.py`）：断言 build_contract stage 的 `cmd` 用 `--dataset` 且指向真实存在的 config；`deps` 包含该 config 文件（防"config 改了但没列进 deps 导致 dvc 不重跑"的隐蔽坑）。

不测：`is_target_endpoint`（留 ③）、log 修复（数据层）、真实 87GB 数据（用 mini fixture）。

**DVC 测试划界**：本次只加"stage 参数/依赖一致性"静态测试，**不**上"真跑 dvc repro"集成测试——后者需 DVC 环境 + mini 数据被 DVC 追踪，CI 脆弱，成本高。真跑 `dvc repro` 留后续 PR 的能力建设 + 本次收尾手动验证。

### 3.6 落地流程与数据安全

改动文件清单：

| 文件 | 改动 |
|---|---|
| `configs/data/merged_v1.yaml` | 新建 |
| `configs/data/anomod.yaml` | 改造成同格式（`roots: [data/anomod]`） |
| `configs/contract/endpoint_to_service.yaml` | 整体改为真实 8 个 client 入口 |
| `src/data/dataset_config.py` | 新建，dataset config loader |
| `scripts/build_contract.py` | `--data-root`→`--dataset`；多 root 枚举；Normal 来源按 config |
| `dvc.yaml` | build_contract stage 改用 `--dataset`，deps 补 config |
| `tests/` | §3.5 的 5 类测试 |
| `tests/fixtures/` | endpoint_raw mini case |

数据安全（用户特别叮嘱）：
- 磁盘：根分区 631GB 可用，两数据集已在本地，合并 parquet 产物 <100MB，无空间风险。
- **本次不跑 `dvc add`**：87G+41G 推 remote 是独立大操作，不塞进本 PR。数据目录保持未追踪现状。
- `dvc repro` 会覆盖旧 baseline_v0 产物（写 `artifacts/`，已 gitignore）——预期行为，数据集变了 baseline 本就要重算。

收尾验证：
1. `pytest tests/` 全绿。
2. `dvc repro` 全链路跑通，`dvc metrics show` 看合并后 AUROC/AUPRC 合理（非 NaN、非 0.5）。
3. 确认 16 个 endpoint_raw case 都进 eval、Normal 只来自 anomod_v1。
4. 写 history entry 006 + 更新 index，把 ③ 作为遗留 TODO 留痕。

## 4. 组件边界

- `src/data/dataset_config.py`：纯配置加载，输入 yaml 路径，输出 dataclass（roots/normal_source/window）。无副作用，独立可测。
- `scripts/build_contract.py`：编排层，依赖 dataset_config + 4 个 preprocessor。改动限于入口参数和 case 枚举，单 case 处理逻辑不动。
- `configs/contract/endpoint_to_service.yaml`：数据契约的一部分，内容修正后由 test 3 钉住口径。

## 5. 遗留 TODO（留给后续 PR）

- **③ per-endpoint 标签 + 评估**（论文核心创新点）：用 `is_target_endpoint` 把 case 级 `is_anomaly` 细化为"只有故障目标 endpoint 在 inject 窗内才算异常"，做 per-endpoint 粒度 AUROC。涉及 contract schema 变更（加列 + 改标签口径），可能需开 contract v1。
- **监督学习前瞻**：若后续做监督学习，需 (1) 精确标签（③ 覆盖）；(2) 参数化 `build_contract.py:325` 的 One-Class train split（改为支持"全量带标签进训练"）；(3) 新增带分类头的监督 detector（EarlyConcatFusion 可复用，DeepSVDD 不可）。③ 设计标签 schema 时应前瞻性同时产出 case 级 + endpoint 级两套标签列，避免二次改 schema。
- **真跑 dvc repro 的集成测试能力**：本次只上静态 stage 一致性测试，端到端 DVC 执行测试留后续。
- **endpoint_raw log 重采**：fsnotify 崩溃需在采集端提高 inotify watcher 上限后重采，本次走 NaN 降级。
- **数据集 DVC 追踪**：anomod_v1 / endpoint_raw 是否 `dvc add` 推 remote，独立决策。
