# TT Dataset Guide

> 目标读者：接手本数据的 agent。本文描述数据的物理组织、字段口径、pipeline 逻辑和已知问题，供快速上手查阅。
> 最后更新：2026-06-19

---

## 一、数据集定位

本数据集来自 Train-Ticket 微服务系统，在 k3s + SkyWalking 环境下采集，用于 **per-endpoint × time-window 的单类异常检测（One-Class Classification）** 研究。

- 训练输入：Normal case 的多模态 RED 特征向量（全部 label=0）
- 评估标签：case_metadata.json 中的 `inject_start_ms` / `inject_end_ms`（强标注，精确到毫秒）
- 正样本（评估）：inject 阶段内的 endpoint × 时间窗口
- 负样本（训练+评估）：Normal case 全程 + 异常 case 的 baseline/recover 阶段

---

## 二、目录结构

数据根目录：`data/anomod/`

```
data/anomod/
├── Normal/
├── Lv_P_DISKIO_preserve/
├── Lv_P_NETLOSS_preserve/
├── Lv_S_HTTPABORT_preserve/
├── Lv_S_DNSFAIL_preserve_no_order/
├── Lv_S_KILLPOD_preserve/
├── Lv_S_KILLPOD_order/
├── Lv_S_KILLPOD_gateway/          ← ⚠️ inject 阶段 trace 缺失，见已知问题
├── Lv_D_cachelimit/
├── Lv_D_CONNECTION_POOL_exhaustion/
└── Lv_D_TRANSACTION_timeout/
```

每个 case 的内部结构统一：

```
<case_name>/
├── api_responses/          # 原始客户端 HTTP 请求日志（JSONL）
├── trace_data/             # 原始 SkyWalking trace JSON + case_metadata.json
├── metric_data/            # Prometheus 系统指标（CSV long-format，~850万行/case）
├── log_data/               # 各 pod 原始日志（按 pod 子目录）
└── _pipeline_out/          # Pipeline 产物（建模直接使用）
    ├── tt_traces_red_15s.csv
    ├── tt_traces_red_15s_raw.csv
    ├── tt_traces_red_15s_report.md
    ├── tt_endpoint_health_15s.csv
    ├── tt_fused_15s.csv              ← 最终建模表
    └── tt_traces_red_window_comparison.md
```

**注意**：run_0616_02 系列 case（8 个）还有 5s/10s 的 traces 和 health 产物，但没有 fused；5s/10s 仅供消融对比，不是主要建模窗口。Normal 和 run_0612_03 系列（DISKIO/NETLOSS）仅有 15s fused。**唯一三批全覆盖的窗口是 15s。**

---

## 三、Case 清单

| case 名 | 类型 | 注入的异常 | 注入目标服务 |
|---|---|---|---|
| Normal | baseline | — | — |
| Lv_P_DISKIO_preserve | Performance | StressChaos（磁盘 IO） | ts-preserve-service |
| Lv_P_NETLOSS_preserve | Performance | NetworkChaos（丢包） | ts-preserve-service |
| Lv_S_HTTPABORT_preserve | Service | HTTPChaos（中止请求） | ts-preserve-service |
| Lv_S_DNSFAIL_preserve_no_order | Service | DNSChaos（域名解析失败） | ts-preserve-service |
| Lv_S_KILLPOD_preserve | Service | PodChaos（杀 pod） | ts-preserve-service |
| Lv_S_KILLPOD_order | Service | PodChaos（杀 pod） | ts-order-service |
| Lv_S_KILLPOD_gateway | Service | PodChaos（杀 pod） | ts-gateway-service ⚠️ |
| Lv_D_cachelimit | Database | 缓存容量限制 | — |
| Lv_D_CONNECTION_POOL_exhaustion | Database | 连接池耗尽 | — |
| Lv_D_TRANSACTION_timeout | Database | 事务超时 | — |

**anomaly_type 解析规则**（由 case 目录名前缀决定）：
- `Normal` → `Normal`
- `Lv_P_*` → `Performance`
- `Lv_S_*` → `Service`
- `Lv_D_*` → `Database`
- `Lv_C_*` → `Code`

---

## 四、原始数据格式

### 4.1 api_responses（客户端侧）

文件：`api_responses/*.jsonl`，JSON Lines 格式，每行一个 HTTP 请求记录。

```json
{
  "timestamp": "2026-06-09T02:29:46Z",
  "method": "POST",
  "url": "http://localhost:30467/api/v1/users/login",
  "status_code": 200,
  "latency_ms": 730.67
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `timestamp` | ISO 8601 UTC | 请求时间 |
| `method` | string | HTTP 方法（GET/POST/PUT 等） |
| `url` | string | 完整请求 URL，含 localhost:30467 代理端口 |
| `status_code` | int | HTTP 响应状态码 |
| `latency_ms` | float | 端到端延迟（毫秒） |

### 4.2 trace_data（服务端侧）

文件：`trace_data/<case_id>_skywalking_traces_*.json`（已从分片合并），以及 `trace_data/case_metadata.json`。

**case_metadata.json 字段**（评估标签来源）：

```json
{
  "case_id": "Lv_P_DISKIO_preserve_20260612T070311Z_em",
  "anomaly_type": "Lv_P_DISKIO_preserve",
  "is_anomaly_case": true,
  "normal_run_id": "normal_0609_planA_sliced",
  "window_start_ms": 1781247797057,
  "window_end_ms": 1781249040226,
  "inject_start_ms": 1781248097062,
  "inject_end_ms": 1781248702291,
  "tt_max_workers": 4,
  "baseline_sec": 300,
  "inject_sec": 600,
  "recover_sec": 120,
  "chaos_yaml": "Lv_P_DISKIO_preserve.yaml",
  "target_service": "ts-preserve-service",
  "anomaly_level": "performance",
  "chaos_kind": "StressChaos",
  "recoverable": true,
  "schedule_mode": "three_phase"
}
```

关键字段：`inject_start_ms` / `inject_end_ms` 是毫秒级时间戳，是强标注的唯一来源，用于划分 baseline/inject/recover 三个阶段。

### 4.3 metric_data

文件：`metric_data/<case_id>_metrics_*.csv`，Prometheus long-format，每行一条指标采样。

字段：`metric_name`, `timestamp`（秒级 unix），`datetime`, `value`, 以及各种 kubernetes label 列（`container`, `pod`, `node` 等）。规模约 850 万行/case。**当前 pipeline 未使用此模态。**

### 4.4 log_data

文件：`log_data/<case_id>_*/`，按 pod 子目录存放原始日志文本。**当前 pipeline 未使用此模态。**

---

## 五、Pipeline 产物 Schema

以下以 15s 窗口为准（建模主窗口）。

### 5.1 tt_fused_15s.csv（最终建模表，28 列）

一行 = 一个 endpoint × 一个 15s 时间窗。

**标识列**

| 列名 | 类型 | 说明 |
|---|---|---|
| `case_id` | str | case 目录名（含时间戳，如 `Lv_P_DISKIO_preserve_20260612T070311Z_em`） |
| `anomaly_type` | str | 见第三节解析规则 |
| `timestamp_window` | str | 窗口起始时间 ISO UTC（`%Y-%m-%dT%H:%M:%SZ`），向下取整到 15s |
| `endpoint_key` | str | `METHOD:normalized_path`，如 `POST:/api/v1/users/login` |
| `method` | str | HTTP 方法，大写 |
| `normalized_path` | str | URL 归一化后的路径，如 `/api/v1/contactservice/contacts/account/{uuid}` |

**服务端 RED（来自 trace_data，字段前缀 `trace_*`）**

| 列名 | 类型 | 说明 |
|---|---|---|
| `trace_request_count` | int | 窗口内 Entry span 数 |
| `trace_latency_mean` | float | 平均延迟（ms） |
| `trace_latency_p50` | float | 中位延迟（ms），样本 < 2 时 NaN |
| `trace_latency_p95` | float | p95 延迟（ms），样本 < 5 时 NaN |
| `trace_latency_p99` | float | p99 延迟（ms），样本 < 5 时 NaN |
| `trace_error_rate` | float | 错误请求占比（is_error=True） |
| `trace_5xx_rate` | float | 5xx 响应占比（仅有 status 的 span）；**Normal case 下 ~99.9% NaN**，因为 SkyWalking 大多数 span 不携带 HTTP status code |
| `trace_4xx_rate` | float | 4xx 响应占比（仅有 status 的 span）；同上，Normal case 下 ~99.9% NaN |
| `trace_status_coverage` | float | 有 HTTP status 的 span 比例 |

**客户端 RED（来自 api_responses，字段前缀 `client_*`）**

| 列名 | 类型 | 说明 |
|---|---|---|
| `client_request_count` | int | 窗口内客户端请求数 |
| `client_error_rate` | float | status_code ≥ 400 的比例 |
| `client_2xx_rate` | float | 2xx 响应比例 |
| `client_4xx_rate` | float | 4xx 响应比例 |

**标注列**

| 列名 | 类型 | 说明 |
|---|---|---|
| `phase` | str | `baseline` / `inject` / `recover` / `normal`（Normal case） / `unknown` |
| `injection_start_ms` | int/NaN | 注入开始毫秒时间戳（来自 case_metadata.json） |
| `injection_end_ms` | int/NaN | 注入结束毫秒时间戳（来自 case_metadata.json） |
| `target_service` | str/NaN | 被注入的服务名 |
| `weak_is_anomaly` | int | 弱标注（0/1），仅供历史兼容，评估以 phase=inject 为准 |
| `label_confidence` | str | `normal` / `high` / `medium` / `low` / `unknown` |
| `latency_anomaly_signal` | int | 延迟信号（0/1），p95 超基线 MAD 阈值 |
| `error_anomaly_signal` | int | 错误率信号（0/1），超基线均值 + 0.1 |
| `5xx_anomaly_signal` | int | 5xx 信号（0/1），超基线均值 + 0.1 |

### 5.2 tt_traces_red_15s.csv（服务端 RED，25 列）

包含 5.1 中所有服务端 RED 列 + 标注列，另有 `window_str`（ISO 时间字符串）。`timestamp_window` 在此文件是 epoch-ms 整数（与 fused 中的 ISO 字符串不同，fused 时已转换）。

### 5.3 tt_endpoint_health_15s.csv（客户端 RED，15 列）

| 列名 | 说明 |
|---|---|
| case_id, anomaly_type, timestamp_window | 标识（timestamp_window 为 ISO 字符串） |
| endpoint_key, method, normalized_path | endpoint 标识 |
| request_count, error_rate | 请求量和错误率 |
| latency_mean, latency_p50, latency_p95, latency_p99 | 延迟分位数（样本不足时 NaN） |
| status_2xx_rate, status_4xx_rate, status_5xx_rate | 状态码分布 |

---

## 六、Endpoint 清单（Normal case，15s 窗口，8 个 endpoint）

```
GET:/api/v1/assuranceservice/assurances/types
GET:/api/v1/contactservice/contacts/account/{uuid}
POST:/api/v1/inside_pay_service/inside_payment
POST:/api/v1/orderservice/order/refresh
POST:/api/v1/preserveservice/preserve
POST:/api/v1/travel2service/trips/left
POST:/api/v1/travelservice/trips/left
POST:/api/v1/users/login
```

注：异常 case 的 trace 侧因故障影响，可能出现部分 endpoint 缺失或出现 Normal 中没有的 endpoint。fused join 为 inner join，只保留客户端和服务端都能对上的行。

---

## 七、数据量统计（15s 窗口）

| Case | fused 行数 | 类型 | 备注 |
|---|---|---|---|
| Normal | 799 | 负样本基线 | phase=normal: 799 |
| Lv_P_DISKIO_preserve | 349 | anomaly | baseline:107 / inject:202 / recover:40 |
| Lv_P_NETLOSS_preserve | 350 | anomaly | baseline:107 / inject:202 / recover:41 |
| Lv_S_HTTPABORT_preserve | 630 | anomaly | baseline:290 / inject:283 / recover:57 |
| Lv_S_DNSFAIL_preserve_no_order | 628 | anomaly | baseline:290 / inject:281 / recover:57 |
| Lv_S_KILLPOD_preserve | 571 | anomaly | baseline:290 / inject:281 / recover:— ⚠️ |
| Lv_S_KILLPOD_order | 474 | anomaly | baseline:282 / inject:192 / recover:— ⚠️ |
| Lv_S_KILLPOD_gateway | 279 | anomaly ⚠️ | baseline:279 / inject:— / recover:— |
| Lv_D_cachelimit | 625 | anomaly | baseline:283 / inject:281 / recover:61 |
| Lv_D_CONNECTION_POOL_exhaustion | 628 | anomaly | baseline:287 / inject:283 / recover:58 |
| Lv_D_TRANSACTION_timeout | 386 | anomaly | baseline:287 / inject:47 / recover:52 |
| **合计** | **5,719** | — | — |

⚠️ KILLPOD_preserve / KILLPOD_order 无 recover 阶段数据（pod 被杀后服务未能恢复上报）。

---

## 八、Pipeline 运行方式

两步按顺序跑，第二步依赖第一步的产物：

```bash
# 步骤 1：提取服务端 RED + 强/弱标注
TT_DATA_ROOT=<run目录>  \
TT_OUTPUT_DIR=<case目录>/_pipeline_out \
TT_CASE_FILTER=<case_dir_name> \
python process_tt_traces.py

# 步骤 2：提取客户端 RED + 融合
TT_DATA_ROOT=<run目录>  \
TT_OUTPUT_DIR=<case目录>/_pipeline_out \
TT_CASE_FILTER=<case_dir_name> \
python build_endpoint_health.py
```

**关键约束：`TT_DATA_ROOT` 必须指向 run 级别**，不能指向 case 目录。若指向 case 目录，会把 `api_responses/` 或 `trace_data/` 子目录名当作 case_id，导致 join 全部失败。

---

## 九、process_tt_traces.py 逻辑摘要

**数据流**：

```
Traces JSON
  → [parse_single_trace_file]  提取 Entry span（type='Entry' + endpoint_name 非空）
  → [_load_all_trace_files]    合并全部文件为 DataFrame
  → [groupby + _calc_red]      按 (case_id, anomaly_type, timestamp_window, endpoint_key) 聚合 RED
  → tt_traces_red_{ws}s_raw.csv
  → [apply_weak_labeling]
      ├─ _compute_normal_baseline  MAD 法计算各 endpoint 正常基线
      ├─ _apply_anomaly_signals    三信号（latency/error/5xx）
      ├─ label_confidence 四档（normal/high/medium/low）
      └─ _load_case_metadata_map + classify_phase  强标注 phase/injection_*_ms（注：fused 表列名是 injection_start_ms/injection_end_ms，来自 case_metadata.json 的 inject_start_ms/inject_end_ms）
  → tt_traces_red_{ws}s.csv
  → [generate_quality_report] → tt_traces_red_{ws}s_report.md
```

**Entry span 提取条件**：
- `span['type'] == 'Entry'`
- `span['endpoint_name']` 非空且匹配 `^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS):(/.*)`
- 不匹配则走 fallback：从 `trace.summary.endpoint_names` 取第一个可解析 endpoint

**RED 指标计算关键点**：
- `trace_5xx_rate` / `trace_4xx_rate`：分母是有 status_code 的 span 数（不是总 span 数），无 status 的窗口为 NaN
- 分位数：p50 需 ≥ 2 样本，p95/p99 需 ≥ 5 样本，否则 NaN
- `window_str` 列仅在 raw/labeled 文件中存在，fused 时已删除

**弱标注（三信号 + MAD）**：
- latency 阈值 = Normal endpoint p95 的 median + 3 × MAD（MAD=0 时用 median × 1.1）
- error/5xx 阈值 = Normal endpoint 均值 + 0.1
- 无 Normal baseline → 三信号全置 0，label_confidence='unknown'（仍继续强标注 merge）
- 无 baseline 的 endpoint 仅 5xx 有绝对回退（0.4）；error_rate/latency 不设回退（4xx 主导/量级差异大）

**强标注（phase）**：
- `classify_phase(timestamp_window_ms, meta)` 对比 inject_start_ms / inject_end_ms
- 无 metadata → Normal case 返回 'normal'，其他返回 'unknown'

---

## 十、build_endpoint_health.py 逻辑摘要

**数据流**：

```
api_responses/*.jsonl
  → [_load_all_api_responses]  扫描 JSONL，解析 timestamp/method/url/status_code/latency_ms
  → [_make_endpoint_key]       normalize_url + METHOD 大写 → endpoint_key
  → [_aggregate_endpoint_health]  按 (case_id, anomaly_type, timestamp_window, endpoint_key) 聚合
  → tt_endpoint_health_{ws}s.csv
  → [build_tt_fused]
      ├─ 读 tt_traces_red_{ws}s.csv（labeled 优先，fallback raw）
      ├─ traces timestamp_window 从 epoch-ms 转 ISO 字符串
      └─ inner join on [case_id, endpoint_key, timestamp_window]
  → tt_fused_{ws}s.csv
```

**URL 归一化规则**（`normalize_url`，与 process_tt_traces.py 各一份，必须同步）：
- UUID（`[0-9a-f]{8}-...-[0-9a-f]{12}`）→ `{uuid}`
- 纯数字段 → `{id}`
- 符合 `KNOWN_SEMANTIC` 集合、snake_case 或 camelCase → 保留
- 其他随机 ID → `{param}`

**Fused join 关键点**：
- join 为 `inner join`，`validate='one_to_one'`（重复 key 会报错）
- health 侧 `client_*` 列重命名后拼接到 traces 全列右侧
- join hit rate < 60% 时告警；= 0% 常见原因：TT_DATA_ROOT 层级错误、endpoint_key 不同步、多 case 污染
- labeled vs raw 自动选择：若 OUTPUT_DIR 有跨 case 残留，自动降级到 raw（基于 case_id 集合校验）

---

## 十一、已知问题

### KILLPOD_gateway inject 阶段数据缺失

ts-gateway-service 既是 PodChaos 目标又是 SkyWalking trace 上报通道。Pod 被杀后 trace 无法上报，inject 阶段数据永久缺失。

**处置建议**：从评估集中排除该 case，或仅用 baseline/recover 阶段作负样本。

### 窗口密度与 NaN 问题

并发负载（4 worker）下各窗口的 p95 NaN 率：

| 窗口 | p95 NaN 率 | 中位请求数 | 建议 |
|---|---|---|---|
| 5s | ~54% | ~4 | 不达标 |
| 10s | ~27.5% | ~8 | 勉强 |
| **15s** | **~10%** | **~11** | **最小可用，主结果** |
| 90s | <5% | ~80 | 质量最好，但窗口数太少 |

### weak_is_anomaly 仅供历史兼容

新采集数据（run_0612_03、run_0616_02）都有强标注，评估应以 `phase == 'inject'` 为准，不要依赖 `weak_is_anomaly`。

---

## 十二、磁盘占用

总计约 44 GB（全部 raw 数据 + pipeline 产物）。

| 数据类型 | 主要来源 | 大小范围 |
|---|---|---|
| trace_data | run_0612_03 最大（~3.6G/case）| 300M ~ 3.6G |
| metric_data | 固定采样频率，基本一致 | ~1.3G ~ 3.0G |
| api_responses | 很小 | < 5M |
| log_data | 中等 | 3M ~ 21M |
| _pipeline_out | 极小（建模用） | < 10M |

pipeline 产物（`tt_fused_*.csv`）合计不超过 100M，99% 磁盘被 raw 数据占用。
