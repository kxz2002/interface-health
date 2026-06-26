# 多模态数据融合 Contract v0 设计文档

> 日期: 2026-06-26
> 状态: 设计已与决策者达成一致，待落地实现
> 上游文档: `docs/plans/2026-05-08-lit-review-and-research-directions.md`
> 相关知识: `/home/kxz2002/knowledge/wiki/code-research/modify-*.md`, `anofusion-*.md`, `mad-cmc-*.md`

---

## 1. 设计目标与背景

### 1.1 项目定位

本仓库的核心研究目标是 **per-endpoint × time-window 的单类异常检测（One-Class）**，数据集为 Train-Ticket 微服务系统的 11 个 case（1 Normal + 10 故障注入）。建模的前置条件是有一个**合理的多模态数据融合框架**。

### 1.2 v0 Contract 要解决的根本问题

数据探查（见 PR #2 后续讨论与 dataset-guide.md）暴露出三个本质矛盾：

1. **粒度异构**：trace / api_responses 可达 endpoint 粒度，metrics / logs 只能到 pod (service) 粒度
2. **时间粒度异构**：trace 是 span 级（毫秒），metrics 是 15s 等间隔采样，logs 是事件流（不规则）
3. **数据形态异构**：metrics 是结构化数值，logs 是非结构化文本，trace 是嵌套调用树

如果没有一个稳定的"数据交付边界"（Contract），则：
- 换预处理算法（如 log 从 Drain3 升级到 Hawkes）会牵动下游模型代码
- 换融合策略（如 early concat 升级到 cross-attention）会牵动 DataLoader
- 换检测器（如 SVDD 换成 LSTM-AE）会牵动整个 pipeline

Contract v0 的使命是**在三层（预处理 / 数据加载 / 模型）之间设一道稳定的接口墙**，让每一层都可以独立演进。

### 1.3 设计原则

- **预处理与模型解耦**：Contract 是两者之间的稳定边界
- **离线 + 模块化**：预处理器是独立 Python class，离线一次性生成 contract parquet；DataLoader 是轻量级 reader
- **融合是独立中间层**：FusionModule 既不属于 DataLoader 也不属于 Detector，是论文消融实验的核心维度
- **特征字段配置驱动**：通过 YAML 配置定义入选特征，便于扩展和 ablation
- **时序扩展性内置**：Contract 行级存储，DataLoader 层支持 PointDataset / SequenceDataset 切换

---

## 2. 总体架构

```
[离线阶段，一次性] Modality Preprocessors（可插拔，每个独立版本化）

  Raw Sources                 Pluggable Preprocessor             Contract Columns
  ─────────────               ──────────────────────             ────────────────
  trace_data/*.json    ───→   TracePreprocessor@v0      ───→    endpoint_red__*
                              v0: span 聚合 → RED
                              v1 候选: + Diffusion
                              v2 候选: + Transformer

  api_responses/*.jsonl ──→   ApiPreprocessor@v0        ───→    (合并进 endpoint_red)
                              v0: 窗口聚合 → RED

  metric_data/*.csv    ───→   MetricPreprocessor@v0     ───→    service_metric__*
                              v0: 32 选 5 + rate + 窗口对齐
                              v1 候选: + Z-score Transformer

  log_data/**/*.log    ───→   LogPreprocessor@v0        ───→    service_log__*
                              v0: Drain3 + 频率/错误/多样性
                              v1 候选: + Hawkes 强度
                              v2 候选: + BERT 语义
                                                                       │
                              ★★★ CONTRACT v0 BOUNDARY ★★★              ▼
                                                              artifacts/contract_v0/*.parquet
                                                                       │
─────────────────────────────────────────────────────────────────────── │ ─────────
[在线阶段] 训练 / 推理                                                 │
                                                                       ▼
  parquet ──→ ContractDataLoader ──→ per-modality dict ──→ FusionModule ──→ Detector
                ──────────────         ────────────────     ─────────────    ────────
                职责：                 输出：               可换插件：       可换插件：
                · 读 parquet           {                    · early_concat   · Deep SVDD
                · 应用 norm stats       "endpoint_red":     · gated_fusion   · LSTM-AE
                · NaN 填补              (10,) tensor,       · cross_attn     · Transformer
                · 序列窗口构造          "service_metric":    · MSCA-CGFM      · GNN
                  (可选 T=8/16)         (5,) tensor,        · GAT
                · 张量化                "service_log":
                                         (3,) tensor,
                                        "label": ...,
                                        "meta": ...
                                       }
```

**三层职责矩阵**：

| 层 | 职责 | 是否插件化 | 升级影响 |
|----|------|----------|---------|
| Modality Preprocessor | raw → per-modality 数值特征列 | ✅ 每个模态独立替换 | 仅影响对应模态的字段集，contract 升 minor 版本 |
| ContractDataLoader | parquet → per-modality tensor dict | ❌ 固定逻辑 | 不应升级 |
| FusionModule | per-modality dict → 统一表示 | ✅ 论文消融维度 | 不影响 contract |
| Detector | 统一表示 → 异常分数 | ✅ baseline 可换 | 不影响 contract |

---

## 3. Sample 定义

**一个 sample = 一个 `(case_id, endpoint_key, timestamp_window)` 三元组**

```python
sample_id = f"{case_id}__{endpoint_key}__{timestamp_window_ms}"
# 例: "Normal_planA__POST:/api/v1/preserveservice/preserve__1780972185000"
```

### 3.1 v0 覆盖范围

| 维度 | v0 范围 | v1 计划 |
|------|---------|---------|
| Endpoint 数 | 8 个外部可观测 endpoint | 扩展到 34 个 trace 可见 endpoint |
| Case 数 | 11 个（1 Normal + 10 故障） | 不变 |
| 时间窗大小 | 固定 15s | 通过 `window_size_s` 字段支持配置 |
| 总样本数 | ~10K 行（8 endpoints × 11 cases × ~117 windows） | ~40K 行 |

**为什么 v0 只用 8 个 endpoint**：

`tt_traces_red_15s.csv` 含 34 endpoints，但 `tt_endpoint_health_15s.csv` 只有 8 个（workload generator 只调用了 8 个外部入口）。inner join 后只保留 8 个有 client+server 双侧数据的 endpoint。v0 接受这个限制以避免特征异构（部分 endpoint 缺少 client 侧特征）。

**为什么扩展到 34 个 endpoint 留到 v1**：

26 个被排除的 endpoint 是微服务间内部调用，仅有 trace 侧数据。引入它们会带来两个问题：(1) 特征向量结构不均匀；(2) 调用链级别的关联建模需要服务依赖图，工程量大。v0 阶段优先把流程跑通。

---

## 4. Contract 字段定义

### 4.1 标识与对齐字段（6 个，不进模型）

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `sample_id` | str | 派生 | 复合主键 |
| `case_id` | str | case 目录名 | `Normal_planA_20260609T022635Z` |
| `endpoint_key` | str | trace + api join key | `POST:/api/v1/preserveservice/preserve` |
| `service_name` | str | **静态 YAML 映射表** | `ts-preserve-service` |
| `timestamp_window_ms` | int64 | trace window 起点（对齐锚） | `1780972185000` |
| `window_str` | str | 可读时间 | `2026-06-09T02:29:45Z` |

**静态 endpoint→service 映射**：
通过 `configs/contract/endpoint_to_service.yaml` 提供，v0 仅含 8 个外部 endpoint 的映射。不依赖 trace 数据动态推导，避免运行时风险。

### 4.2 特征字段（18 维数值，进模型）

列命名约定 `{modality}__{feature}`，方便按前缀提取分组。

**Group 1: `endpoint_red__*`（10 维，per-endpoint min-max 归一化）**

| 列名 | 单位/范围 | 归一化 |
|------|----------|--------|
| `endpoint_red__trace_request_count` | count | log1p + min-max |
| `endpoint_red__trace_latency_p50` | ms | log1p + min-max |
| `endpoint_red__trace_latency_p95` | ms | log1p + min-max |
| `endpoint_red__trace_error_rate` | [0,1] | 原值 |
| `endpoint_red__trace_5xx_rate` | [0,1] | 原值 |
| `endpoint_red__client_request_count` | count | log1p + min-max |
| `endpoint_red__client_latency_p95` | ms | log1p + min-max |
| `endpoint_red__client_error_rate` | [0,1] | 原值 |
| `endpoint_red__client_5xx_rate` | [0,1] | 原值 |
| `endpoint_red__latency_divergence` | ms | log1p + min-max（派生：client_p95 − trace_p95） |

**Group 2: `service_metric__*`（5 维，per-service min-max 归一化）**

| 列名 | 原始 Prometheus 指标 | 归一化 |
|------|---------------------|--------|
| `service_metric__cpu_usage_rate` | `rate(container_cpu_usage_seconds_total[5m])` | [0,1] 原值 |
| `service_metric__memory_usage_ratio` | `container_memory_working_set_bytes / container_spec_memory_limit_bytes` | [0,1] 原值 |
| `service_metric__net_rx_error_rate` | `rate(container_network_receive_errors_total[5m])` | log1p + min-max |
| `service_metric__net_tx_error_rate` | `rate(container_network_transmit_errors_total[5m])` | log1p + min-max |
| `service_metric__process_count` | `container_processes` | log1p + min-max |

**Group 3: `service_log__*`（3 维，per-service min-max 归一化）**

| 列名 | 含义 | 归一化 |
|------|------|--------|
| `service_log__event_rate` | Drain3 模板匹配后的事件总数 / 窗口长度 | log1p + min-max |
| `service_log__error_ratio` | ERROR/WARN 级别事件数 / 总事件数 | [0,1] 原值 |
| `service_log__template_diversity` | 窗口内唯一模板数 | log1p + min-max |

**缺失值约定**：
原始 modality preprocessor 缺数据时直接写 `NaN`，不做填补。模型读取时由 DataLoader 选择策略：
- v0 baseline：用 Normal 训练集的均值填补
- 比率类（error_rate 等）特例：无请求时填 0（无请求 = 无错误）

### 4.3 标签字段（8 个，训练时掩码，评估时使用）

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `phase` | str | `case_metadata.json` | `normal / baseline / inject / recover / unknown` |
| `is_anomaly` | bool | 派生：`phase == "inject"` | ground truth |
| `is_train_eligible` | bool | v0 全 `True`（Normal case） | 占位字段，预留未来样本筛选用 |
| `injection_start_ms` | int64 \| null | `case_metadata.json` | inject 阶段起点 |
| `injection_end_ms` | int64 \| null | `case_metadata.json` | inject 阶段终点 |
| `target_service` | str \| null | `case_metadata.json` | 被注入服务 |
| `anomaly_type` | str | case 目录名 | 分层评估 key（10 类故障 + Normal） |
| `anomaly_level` | str | `case_metadata.json` | `service / performance / database` |

---

## 5. 模态预处理器规约

### 5.1 抽象基类

```python
class ModalityPreprocessor(ABC):
    """所有 modality preprocessor 的统一接口。"""

    version: str  # e.g., "v0", "v1"

    @abstractmethod
    def fit(self, raw_paths: List[Path]) -> None:
        """从 Normal case 拟合 normalization 参数。"""

    @abstractmethod
    def transform(self, raw_path: Path, case_meta: dict) -> pd.DataFrame:
        """处理单个 case 的 raw 数据，返回符合 contract 列约定的 DataFrame。"""

    @abstractmethod
    def get_feature_columns(self) -> List[str]:
        """返回该模态产出的列名列表（用于 schema 校验）。"""
```

### 5.2 TracePreprocessor@v0

- **输入**: `_pipeline_out/tt_traces_red_15s.csv`（已有 pipeline 产物，复用）
- **逻辑**: 选择 8 个 endpoint 的行，重命名列为 `endpoint_red__trace_*`
- **输出**: 10 维特征中的 5 维（trace 部分）
- **工程量**: 极低，本质是 column rename + filter

### 5.3 ApiPreprocessor@v0

- **输入**: `_pipeline_out/tt_endpoint_health_15s.csv`（已有 pipeline 产物）
- **逻辑**: 选择 8 个 endpoint 的行，重命名列为 `endpoint_red__client_*`，与 trace 侧 inner join
- **输出**: 10 维特征中的 5 维（client 部分） + 派生 `latency_divergence`
- **工程量**: 低

### 5.4 MetricPreprocessor@v0

- **输入**: `metric_data/*.csv`（1.3GB ~ 3.0GB 每个 case，总量 ~26GB）
- **逻辑**:
  1. **预过滤**：先读 `datetime` 列定位 `[window_start_ms, window_end_ms]` 范围，再加载其余列（避免全量读 3GB）
  2. **指标选择**：保留 5 种指标（见 §4.2 Group 2）
  3. **rate 计算**：对 counter 类指标（cpu_seconds, network_errors）计算 15s 窗口内的 rate
  4. **service 聚合**：按 `(service_name, window)` 聚合（同 service 多 pod 取均值）
  5. **窗口对齐**：以 trace window 起点为锚，metric 时间戳用 floor 对齐
- **输出**: 5 维 service 级特征
- **工程约束**:
  - 必须使用 **pyarrow** 或 **pandas chunked reading**
  - **中间缓存**：把过滤后的 ~180K 行子集保存为 `artifacts/intermediate/metrics_filtered_{case_id}.parquet`，避免 DVC repro 重复读 3GB
  - 内存上限：单次处理峰值 < 4GB

### 5.5 LogPreprocessor@v0

- **输入**: `log_data/<case_id>/<service>/<service>_*.log`（pod 原始文本，~38 文件每 case）
- **逻辑**:
  1. **Drain3 解析**：对每个 service 的日志逐行解析，提取模板 ID。Drain3 模型从 Normal case 训练，故障 case 用同一个模型
  2. **窗口分桶**：按行的时间戳分桶到 15s window
  3. **统计聚合**：每 `(service, window)` 计算：事件总数、ERROR/WARN 事件数、唯一模板数
  4. **特征派生**：
     - `event_rate` = 总事件数 / 15
     - `error_ratio` = ERROR数 / 总事件数（无事件时 = 0）
     - `template_diversity` = 唯一模板数
- **输出**: 3 维 service 级特征
- **依赖**: `drain3-improved` 包（已在 environment.yml）

### 5.6 跨 preprocessor 的不变量

- **归一化参数只从 Normal case 拟合**（`fit` 只在 Normal 数据上调用）
- **故障 case 用同一套参数 transform**（One-Class 任务的核心要求）
- **fit 阶段产出 `normalization_stats.json`**，与 contract parquet 同目录

---

## 6. DataLoader 与 FusionModule 规约

### 6.1 ContractDataLoader

```python
class ContractDataLoader:
    """从 contract parquet 读取数据，返回 per-modality tensor dict。"""

    def __init__(
        self,
        parquet_path: Path,
        normalization_stats_path: Path,
        schema_path: Path,
        mode: Literal["point", "sequence"] = "point",
        sequence_length: int = 8,  # 仅 sequence 模式生效
        nan_strategy: Literal["mean", "zero", "raise"] = "mean",
    ):
        ...

    def __getitem__(self, idx) -> Dict[str, Any]:
        """
        返回 dict:
          {
            "endpoint_red":   tensor shape (D_er,) 或 (T, D_er),
            "service_metric": tensor shape (D_sm,) 或 (T, D_sm),
            "service_log":    tensor shape (D_sl,) 或 (T, D_sl),
            "label": {
                "is_anomaly": bool,
                "phase": str,
                ...
            },
            "meta": {
                "sample_id": str,
                "case_id": str,
                ...
            }
          }
        """
```

**关键约束**：
- **Sequence 模式禁止跨 case 边界**：分组键为 `(case_id, endpoint_key)`，滑窗只在组内进行
- **NaN 填补在 DataLoader 完成**，不在 contract 层
- **DataLoader 不做归一化**：归一化已在预处理阶段完成，DataLoader 只读

### 6.2 FusionModule

```python
class FusionModule(ABC):
    """将 per-modality dict 融合为统一张量。"""

    @abstractmethod
    def forward(self, modality_dict: Dict[str, Tensor]) -> Tensor:
        """
        输入: {"endpoint_red": (B, D_er), "service_metric": (B, D_sm), "service_log": (B, D_sl)}
        输出: (B, D_fused) 统一表示
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """供下游 Detector 配置输入维度。"""
```

**v0 实现**：`EarlyConcatFusion` —— 直接 concat 三个 tensor，`output_dim = D_er + D_sm + D_sl = 18`

**v1+ 候选**（不在本迭代实现，但 contract 已支持）：
- `GatedFusion` —— 类 MODIFy 的 MSCA-CGFM 门控融合
- `CrossAttentionFusion` —— 类 MAD-CMC 的双向跨模态 attention
- `GATFusion` —— 类 AnoFusion 的图流融合（需要服务依赖图）

---

## 7. 文件结构与版本演进

### 7.1 产物目录

```
artifacts/contract_v0/
├── train.parquet                   # Normal case，~800 行，is_anomaly=False
├── eval_all.parquet                # 11 cases 合并，~10K 行，含 ground truth
├── normalization_stats.json        # 从 Normal 拟合的 per-endpoint/per-service min/max
└── schema.json                     # contract 元信息

configs/contract/
├── v0.yaml                         # 特征字段配置（详见 §9）
└── endpoint_to_service.yaml        # 8 endpoint 静态映射表

artifacts/intermediate/             # MetricPreprocessor 中间缓存（gitignore）
└── metrics_filtered_{case_id}.parquet
```

### 7.2 `schema.json` 内容示例

```json
{
  "contract_version": "v0.0",
  "window_size_s": 15,
  "alignment_anchor": "trace_window_start_ms",
  "sequence_group_key": ["case_id", "endpoint_key"],
  "feature_dim": 18,
  "feature_groups": {
    "endpoint_red": {
      "columns": ["endpoint_red__trace_request_count", "endpoint_red__trace_latency_p50", ...],
      "preprocessor": "TracePreprocessor@v0",
      "normalization_scope": "per_endpoint"
    },
    "service_metric": {
      "columns": ["service_metric__cpu_usage_rate", ...],
      "preprocessor": "MetricPreprocessor@v0",
      "normalization_scope": "per_service"
    },
    "service_log": {
      "columns": ["service_log__event_rate", ...],
      "preprocessor": "LogPreprocessor@v0",
      "normalization_scope": "per_service"
    }
  },
  "evaluation_strata": ["overall", "by_anomaly_type", "by_anomaly_level"]
}
```

### 7.3 版本演进规则

| 升级 | 触发条件 | 影响范围 |
|------|---------|---------|
| `v0.0 → v0.1` | 特征字段增删（如 log 加 Hawkes 强度向量） | parquet 重新生成，schema.json 更新；DataLoader / Fusion / Detector 接口不变 |
| `v0 → v1` | sample 定义变化（如扩展到 34 endpoints） | 修改 contract 生成代码；DataLoader 不变 |
| `v0 → v2` | 标签语义变化（如改用真实根因标注） | 大变更，触发新一轮 brainstorming |

---

## 8. 关键不变量（Implementation 必须保证）

1. **归一化参数只从 Normal case 拟合**，故障 case 用同一套参数 transform
2. **Sequence 构造不跨 case 边界**：分组键 `(case_id, endpoint_key)`
3. **时间戳对齐到 trace window 起点**，metric/log 用 floor 对齐到 15s 桶
4. **NaN 表示模态层面真实缺失**，不混淆 0 值
5. **`endpoint_to_service.yaml` 必须与 train.parquet 同时存在**
6. **MetricPreprocessor 必须按 case window 预过滤**，禁止全量加载 3GB CSV

---

## 9. v0 特征选择论据

### 9.1 配置驱动的字段集（`configs/contract/v0.yaml`）

```yaml
contract_version: v0.0
window_size_s: 15

modalities:
  endpoint_red:
    preprocessor: TracePreprocessor
    preprocessor_version: v0
    features:
      - trace_request_count
      - trace_latency_p50
      - trace_latency_p95
      - trace_error_rate
      - trace_5xx_rate
      - client_request_count
      - client_latency_p95
      - client_error_rate
      - client_5xx_rate
      - latency_divergence
    normalization: per_endpoint_min_max

  service_metric:
    preprocessor: MetricPreprocessor
    preprocessor_version: v0
    features:
      - cpu_usage_rate
      - memory_usage_ratio
      - net_rx_error_rate
      - net_tx_error_rate
      - process_count
    normalization: per_service_min_max
    candidates_pool:    # 不入 v0 contract，但记录候选供 v1 ablation
      - node_load5
      - container_cpu_cfs_throttled_periods_total
      - container_memory_failcnt
      - container_spec_memory_limit_bytes
      - container_network_receive_bytes_total
      - container_network_transmit_bytes_total
      - node_memory_MemAvailable_bytes
      - node_disk_io_time_seconds_total
      # ... 共 27 个候选

  service_log:
    preprocessor: LogPreprocessor
    preprocessor_version: v0
    features:
      - event_rate
      - error_ratio
      - template_diversity
    normalization: per_service_min_max
    candidates_pool:    # v0.1 可加入的高级特征
      - hawkes_intensity_vector
      - template_id_sequence
      - bert_semantic_vector
```

### 9.2 v0 默认特征的选择依据

| 选择维度 | 入选特征 | 论据 |
|---------|---------|------|
| **RED 方法学**（Google SRE 标准） | trace/client request_count, latency_p95, error_rate, 5xx_rate | RED 是 SRE 行业事实标准（Four Golden Signals） |
| **MODIFy / AnoFusion / DAM 同款** | trace RED + service CPU/Mem + log frequency | 主流多模态融合论文均使用类似的 5±2 维 service 级特征 |
| **数据可用性约束** | 排除 fs/process_fds 等业务定制指标 | 数据探查报告确认这些信号弱、无信噪比 |
| **粒度对齐性** | 只保留 pod 级，丢弃 node 级（如 node_load5） | node 级会被多 pod 共享，污染 service 信号 |
| **派生特征价值** | `latency_divergence` = client_p95 − trace_p95 | 反映网络/网关延迟，是网络类故障的高价值信号 |

### 9.3 后续 ablation 计划

v0 选择的论据需要在 v1 通过实验进一步证明，作为论文的特征分析章节：
- **v0.1 实验**：从 `candidates_pool` 轮流加入候选特征，统计每个的边际贡献（AUROC 提升）
- **v0.2 实验**：用 MI / SHAP / Permutation Importance 给特征重要性打分
- **论文章节归属**：方法论的"特征工程"小节，回答审稿人"为什么这些特征"

---

## 10. 交付物与验收标准

### 10.1 交付物清单

**代码**：

```
src/preprocessors/
├── base.py                            # ModalityPreprocessor 抽象基类
├── trace_preprocessor.py              # v0：复用 tt_traces_red_15s.csv
├── api_preprocessor.py                # v0：复用 tt_endpoint_health_15s.csv
├── metric_preprocessor.py             # v0：32 选 5 + rate + 窗口对齐
└── log_preprocessor.py                # v0：Drain3 + 频率统计

src/contracts/
└── contract_v0.py                     # parquet schema 校验器（沿用 contract_v0 命名）

src/data/
├── contract_dataloader.py             # PointDataset + SequenceDataset
└── normalization.py                   # per-endpoint/per-service min-max fit/transform

src/fusion/
├── base.py                            # FusionModule 抽象基类
└── early_concat.py                    # v0 baseline：dict → flat tensor

scripts/
├── build_contract.py                  # raw → contract parquet 的离线 pipeline
├── train_baseline_v0.py               # Deep SVDD + EarlyConcat 训练
└── eval_baseline_v0.py                # 评估，输出分层 AUROC

configs/
└── contract/
    ├── v0.yaml                        # 特征字段配置
    └── endpoint_to_service.yaml       # 静态映射表

dvc.yaml                               # 新增 build_contract → train_v0 → eval_v0 三个 stage
```

**产物**：

```
artifacts/contract_v0/
├── train.parquet                      # ~800 行
├── eval_all.parquet                   # ~10K 行
├── normalization_stats.json
└── schema.json

artifacts/baseline_v0/
├── scores.parquet                     # 复用现有 scores_v0 contract
└── metrics.json                       # AUROC（overall + by_anomaly_type + by_anomaly_level）
```

**测试**：

```
tests/
├── test_trace_preprocessor.py
├── test_api_preprocessor.py
├── test_metric_preprocessor.py        # 含 chunked reading 内存上限验证
├── test_log_preprocessor.py           # 含 Drain3 模板一致性验证
├── test_contract_v0_schema.py         # 沿用现有 scores/metrics 校验器风格
├── test_normalization.py              # fit 仅用 Normal，transform 跨 case 一致性
├── test_dataloader_point.py
├── test_dataloader_sequence.py        # 验证 (T, D) 形状，序列不跨 case 边界
├── test_early_concat_fusion.py
└── test_e2e_contract_pipeline.py      # 端到端：raw → contract → load → fuse → score
```

**文档**：

```
docs/plans/2026-06-26-multimodal-contract-design.md       # 本设计文档
docs/agent-docs/feature-selection-rationale-v0.md         # 18 维特征选择的详细论据（§9 展开版）
```

### 10.2 验收标准

| # | 标准 | 验证方法 |
|---|------|---------|
| 1 | `dvc repro` 端到端跑通 | 命令行无错误退出 |
| 2 | Schema 校验通过 | `contract_v0.py` 校验器无违规 |
| 3 | 数据量符合预期 | `train.parquet` 约 800 行，`eval_all.parquet` 约 10K 行 |
| 4 | 可复现性 | 同 seed 重跑两次，parquet 文件 hash 一致 |
| 5 | 测试覆盖 | 所有 preprocessor + dataloader + fusion + e2e 测试 pass |
| 6 | Baseline 可用 | Deep SVDD + EarlyConcat 产出 AUROC，整体值落在 [0.6, 1.0]（不要求高，不能是随机 0.5） |
| 7 | 分层评估能跑出 | `metrics.json` 含 `by_anomaly_type`（10 类故障 + Normal）和 `by_anomaly_level`（service/performance/database）的 AUROC |
| 8 | MetricPreprocessor 内存上限 | 处理 3GB CSV 时峰值内存 < 4GB |

### 10.3 本迭代明确 NOT 交付

- Diffusion / Hawkes / Transformer 等高级 preprocessor
- MSCA-CGFM / GAT 等高级 fusion
- 多模型对比（v0 只用 Deep SVDD baseline）
- 特征选择 ablation（标为 v0.1 任务）
- Sequence DataLoader 实际投入使用（写好代码并测试，但 baseline 用 Point 模式）
- 服务依赖图构建（v1 引入 GATFusion 时再做）

---

## 11. 后续迭代规划

### 11.1 v0.1（特征工程深化）

- 从 `candidates_pool` 引入候选特征，跑 ablation
- 输出"特征重要性排名"作为论文图表
- log 模态升级：加入 Hawkes 强度向量

### 11.2 v0.2（融合策略对比）

- 实现 GatedFusion / CrossAttentionFusion
- 在同一 baseline detector 上对比三种融合策略
- 输出"融合策略 ablation"作为论文核心章节

### 11.3 v1（覆盖面扩展）

- Sample 定义扩展到 34 个 endpoints
- 引入服务依赖图，实装 GATFusion
- 引入 SequenceDataset，对比 LSTM-AE / Transformer detector

### 11.4 v2（标签升级，待定）

- 若获得真实根因标注，contract 升级为多任务标签
- 引入定位任务（per-service 二分类）

---

## 12. 相关文档与参考

- **数据探查报告**：本会话上下文（5 个 subagent 并行探查的综合结果）
- **数据集字段口径**：`docs/agent-docs/dataset-guide.md`
- **历史决策记录**：`history/index.md` 与 `history/entries/`
- **文献综述**：`docs/plans/2026-05-08-lit-review-and-research-directions.md`
- **关键论文**:
  - MODIFy (JSS 2026) — 三模态融合 + 扩散去噪 + GAT
  - AnoFusion (KDD 2023) — 异构图流融合
  - MAD-CMC (IPM 2025) — 对比多模态聚类
  - DAM (ICWS 2022) — 简单拼接 baseline
