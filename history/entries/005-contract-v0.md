---
date: 2026-06-26
pr_or_commit: feature/fusion-contract (25 commits, 637f49e → 558b54e)
type: Feature
impact_domains:
  - src/contracts/
  - src/preprocessors/
  - src/fusion/
  - src/models/
  - src/data/
  - scripts/
  - configs/contract/
  - dvc.yaml
  - tests/
---

# 005 · Contract v0 多模态数据融合接口墙

## 做了什么

实现了完整的多模态数据融合 Contract v0：从 raw 数据到可训练 tensor dict 的全链路，以及 One-Class（Deep SVDD）baseline 训练和分层 AUROC 评估。共 25 个提交，111 个测试（105 passed + 1 skipped + 5 bugfix commits）。

三层架构：**离线预处理**（ModalityPreprocessor × 4 → build_contract.py 产出 contract parquet）→ **在线加载**（ContractDataset → per-modality tensor dict）→ **建模**（EarlyConcatFusion → DeepSVDD → 分层 AUROC）。DVC 三个新 stage（build_contract/train_v0/eval_v0）串联全链路，`dvc repro` 可一键重现。

## 关键决策（不在 commit 里）

**1. 18-dim 特征分组：endpoint_red×10 + service_metric×5 + service_log×3**
否决了把 trace 和 api 合并为一个 modality 的方案——虽然简单，但失去了 per-modality ablation 的对比维度。将 trace/api 各自 5 维合并为 endpoint_red 10 维，主要是因为两者均属"客户端视角 RED"，共享归一化 scope（per_endpoint）。

**2. `_REPO_ROOT = Path(__file__).parents[N]` 全局约定**
所有预处理器和脚本的默认路径全部基于 `__file__` 而非调用方 cwd。源头是 T4 review 时发现 TracePreprocessor 用相对路径导致测试外跑报 FileNotFoundError。此后成为全项目强制约定。

**3. FusionModule 继承 `nn.Module, ABC`**
T13 初版只继承 `nn.Module`，ABC 约束完全失效（`nn.Module` 用普通 `type` metaclass，`@abstractmethod` 不注册）。修复：同时继承 `ABC`（带 `ABCMeta`），Python MRO 自动合并 metaclass。

**4. One-Class 数据流防泄漏**
- `Normalizer.fit()` 只传 `full[normal_mask]`（Normal case），异常 case 的统计量不进 fit
- `ContractDataset(nan_strategy="mean", fit_on_parquet=train.parquet)` — eval 路径用训练集均值填 NaN，而非 eval 自身均值
- `DeepSVDD.init_center()` 在训练 epoch 开始前调用，用全量 train_ds 的编码均值初始化

**5. Contract 校验顺序：normalize → clip → validate**
`validate_contract_df` 在 clip 之后调用（rate 类列 ∈ [0,1]），而 `service_log__event_rate` 是原始行计数（非 rate），归一化前可 >1 ——这是设计，不是 bug。T7 审查时捕获并在 T9 里确认。

## 坑 / 已知问题

- **RATE_COLUMNS 漏了 3 列**（T3 bugfix `cfed611`）：`net_rx_error_rate`、`net_tx_error_rate`、`service_log__event_rate` 最初缺失，导致这三列超出 [0,1] 时不报错（静默通过）。
- **TracePreprocessor 时间戳列名**：真实数据是 `timestamp_window`（epoch-ms int），规格文档写的是 `timestamp_window_ms`，需 rename。
- **ApiPreprocessor timestamp 格式**：`tt_endpoint_health_15s.csv` 的时间戳是 ISO 字符串，不是 epoch-ms；需 `pd.to_datetime(...).astype(int64) // 1e6`。另外 `latency_divergence` 需 JOIN 兄弟文件 `tt_traces_red_15s.csv`（endpoint_health 无 trace 列）。
- **`_build_sequence_indices` 物理行连续性依赖**（T11 review 指出）：初版存起始物理行号 + `iloc[loc:loc+seq_len]`，df 未排序时会跨 endpoint 静默出错。T12 修复为存整窗 index list + `df.loc[idx_list]`。
- **DeepSVDD `assert` 在 `-O` 下被剥离**（T14 review 指出）：`score()` 里用了 `assert center is not None`，Python `-O` 优化模式会跳过。修为 `raise RuntimeError`。

## 遗留 TODO

- **T19（docs）**：`docs/feature-selection-rationale-v0.md`——解释 18 个特征为何这样选（论文审稿人会问）。未做，任务范围内但非 TDD，可独立写作。
- **真实数据验证**：全部测试基于 mini_data_root（2 个 case，极少样本）。需用完整 11-case 数据集跑 `dvc repro` 验证 MetricPreprocessor 内存 <4GB 和 AUROC 是否合理。
- **MetricPreprocessor `intermediate_dir` 清理**：`_pipeline_out/metrics_filtered_{case_id}.parquet` 中间缓存在 data/anomod/ 下，READ-ONLY 目录——实际部署时需改 intermediate_dir 到 artifacts/。
- **Drain3 state 跨 case 复用**：`LogPreprocessor` 默认 `drain3_state_path=None`（每次重新训练模板）。跨 case 共享 state 会提高模板稳定性，但需要评估 train→eval 模板漂移的影响。
- **序列模式数据量不足**：mini fixture 所有 case 在单 endpoint 下连续步数均 <4，`test_sequence_dataset_shape` 永远 skip。需更长的 fixture 或降低 sequence_length 默认值来真正覆盖序列路径。
- **log 模态信号缺失（数据重采）**：log join 命中率仅 2%–27%（见下方「坑」），需重新采集保留完整日志。

## DVC 全链路验证（merge 前补充，2026-06-26）

PR merge 前用真实 11-case 数据跑通了 `dvc repro build_contract train_v0 eval_v0`，过程中发现并修复了三类 bug，同时确认了一个不可逆的数据采集限制。

### 修复的 Bug

**Bug 1：`_enumerate_cases` 路径锚点错误**
- 原因：`rglob("case_metadata.json")` 在真实数据中找到 `trace_data/case_metadata.json`，`p.parent` 返回 `trace_data/`，导致所有 case 因找不到 `_pipeline_out` 而被跳过；`Normal` case 无 metadata 文件完全缺失，共 11 个 case 全部处理失败。
- 修复：改为 `glob("*/_pipeline_out")` 以 `_pipeline_out/` 为锚点，同时修复 `_load_case_meta` 支持三种布局（直接路径 / `trace_data/` 子路径 / Normal 合成元数据）。
- 影响文件：`scripts/build_contract.py`

**Bug 2：`LogPreprocessor` 目录层级识别错误**
- 原因：`transform()` 假设 `log_data/<service-pod>/*.log` 两层结构，真实数据多一层 run-id：`log_data/<run-id>/<service-pod>/*.log`，导致 log 模态完全没有输出。
- 修复：新增 `_find_service_dirs()`，检测到直接子目录无 `.log` 文件时自动下探一层。
- 影响文件：`src/preprocessors/log_preprocessor.py`

**Bug 3：`LogPreprocessor` 时区错误**
- 原因：Spring Boot 日志时间戳是本地时间（CST/UTC+8），`pd.Timestamp(ts_str)` 不带时区直接当 UTC 处理，导致 log 窗口比 endpoint 数据快 8 小时，join 命中率为 0%。
- 修复：`_parse_line` 中加 `.tz_localize("Asia/Shanghai")` 再取 epoch ms。
- 影响文件：`src/preprocessors/log_preprocessor.py`

**Bug 4：`ContractDataset` 全 NaN 列守卫过严**
- 原因：`nan_strategy="mean"` 遇到全 NaN 列直接 raise，但 `trace_5xx_rate` 在 Normal case 下本来就全 NaN（dataset-guide 已记录），阻断了训练流程。
- 修复：改为 warning + 回退 0 填充，对应测试从「验证 raise」改为「验证 warn+0填充」。
- 影响文件：`src/data/contract_dataloader.py`、`tests/test_dataloader_point.py`

### 数据采集限制（不可修复）

**log 模态信号基本缺失**：真实数据集的 `log_data/_previous_*.log` 是指向 K8s pod 节点 `/var/log/pods/...` 路径的**断掉的符号连结**——采集时 pod 已销毁，历史日志丢失。每个服务只剩最后一次 log rotation 的当前文件，仅覆盖实验末尾几分钟（而非整个 15–30 分钟实验窗口）。

结果：log join 命中率 2%–27%（各 case 不同），`service_log__*` 三列大面积 NaN 填 0，实际等同常量特征，对 AUROC 无贡献。

**后续行动**：重新采集日志时需在 pod 存活期间直接拷贝 `/var/log/pods/` 下的历史文件，或改用 sidecar/FluentBit 持续采集。待数据重采后 log 模态才能提供有效信号。
