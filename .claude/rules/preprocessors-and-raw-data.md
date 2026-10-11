---
paths:
  - "src/preprocessors/**"
  - "configs/data/**"
  - "scripts/benchmark_downstream.py"
  - "scripts/run_benchmark_sweep.py"
  - "scripts/analyze_temporal_confounding.py"
  - "notebooks/**"
  - "data/new_merge/**"
  - "tests/test_*preprocessor*.py"
  - "tests/test_metric_perf_spike.py"
---

# 预处理器与原始数据

从根 CLAUDE.md 的 Known Gotchas 迁出（entry 033），原文保留。根文件只留特征粒度、`trace_*` 错误列与 KILLPOD 判据的一句话摘要。

- Log parsing 使用 `drain3-improved`（import 路径与原 drain3 相同：`from drain3 import TemplateMiner`）；原 drain3 包因 cachetools==4.2.1 锁定与 mlflow 3.x 不兼容，已弃用
- Drain log parsing 结果依赖输入顺序，不同顺序可能产生不同模板
- **Drain3 输出是聚合统计量，不是 raw token**：LogPreprocessor 产出的是 event_rate/error_ratio/template_diversity 三列，template_id 只作为分组 key 计算多样性，不直接进入特征
- **Log join 静默 NaN**：log 特征经 left join 接入，若某 service 在该时间窗口无日志（如 service 挂了），特征为 NaN，后续均值填补会静默抹掉这个"service 无响应"信号
- **`MAX_CONTENT_CHARS=2000` 截断对 log 特征有实测可量化的损耗，不是"行为不变"（见 history/entries/019、020）**：截断只对占比 88%+ 的短行不改变 Drain3 template 归类，但超长行（占比约 12%，真实数据里可达近百万字符）本身的 template 归类会变，落到 `service_log__template_diversity` 特征上，约 60% 的 15s 窗口取值会变化（平均绝对偏差 ~0.017，最大 ~0.16；`event_rate`/`error_ratio` 不受影响）。阈值 2000→3000 CPU 几乎零代价但损耗改善有限（~14%），继续加大阈值到 20000 才有中等改善但 CPU 涨到 2.3 倍，逼近数据集真实离群值（几十万字符）时 CPU 暴涨到 80 倍。调整该阈值前先读 entry 020 的实测曲线，不要凭直觉猜测。该常量现已在 master（`src/preprocessors/log_preprocessor.py:47`）；`v0`/`v1`/`v1_expanded`（RG）三条历史管线已提交的 `metrics.json` 是无截断旧代码的产物，且数据已永久丢失、无法重跑（entry 021/025）
- 多模态时间对齐时，降采样会丢失 metrics 高频 spike 信号，需谨慎选择对齐策略
- **多模态粒度架构约束**：traces/api_responses 可做 endpoint 级特征；metrics（Prometheus）和 logs（原始文本）只能做 service 级特征，无 per-endpoint 粒度——这是采集端的架构性限制，重采也不会变。融合设计的真实形态是 endpoint 级 + service 级两层 join 键，需提前接受。**标签精度**（区别于特征粒度）取决于数据源提供了哪些 target 字段，共三档，见 `contract-and-labels.md`
- **"target service 同时是 chaos 目标又是 trace/log 上报通道"的 case 会导致 inject 阶段该模态数据永久缺失、AUROC 数学上无定义**：已在两个独立数据集上复现同一模式——`endpoint_raw2` 上的 `Lv_S_KILLPOD_gateway`（entry 016）与 `new_merge` 上的同名 case（entry 023，已剔除）。根因都是 `ts-gateway-service` 被 PodChaos 杀掉后，它自己既是 SkyWalking trace 上报通道又是被杀目标，pod 死后没有任何组件能继续上报该 case 的 trace，导致 inject 阶段 0 行、`eval_all` 里该 case 100% 落在 baseline phase（单一类别），`by_anomaly_type` 分层 AUROC 为 `null`。接入任何新数据集时，若含 `KILLPOD`/`PodChaos` 类故障且目标服务本身承担可观测性上报职责，需要提前排查该 case 是否有这个缺陷，而不是等 metrics.json 里出现 `null` 才发现。**判据是被杀目标是否承担上报职责，不是故障类型**：同批 `Lv_S_KILLPOD_order`/`preserve` 的 inject trace 完好（NaN=0，`trace_request_count` 中位数 62/18 vs baseline 7，AnoMod PAPER_SPEC §6.3 解读为重试风暴），只是没有 recover 阶段。gateway 的缺陷在 AnoMod 原始拷贝上复核成立：其 `tt_fused_15s.csv` 只有 561 行 baseline、inject 0 行（2026-10-11）
- **服务端状态码只出现在 5xx span 上，`trace_*` 错误列名不副实**（AnoMod PAPER_SPEC §6.4；2026-10-11 在 new_merge 的 `tt_fused_15s.csv` 上复核）：`trace_status_coverage>0` 仅 10.8% 行（2578/23868；PAPER_SPEC 称各 case 中位数均为 0，根因未查，疑 SkyWalking span tag 配置）。这些行上 `trace_error_rate` 恰等于 `trace_status_coverage`、其余行为 0，`trace_5xx_rate` 只在这些行非空且恒为 1.0，即带状态码的 span 看起来全是 5xx（推测正常 span 不记状态码），2xx 与"状态未知"无法区分。所以 `trace_5xx_rate` 是退化列（非空即 1.0；contract 侧大面积 NaN 与 entry 031 的 n=0 组源于采集，不是预处理 bug）；`trace_error_rate` 实为"5xx span 占比"，与 `client_error_rate>0` 几乎同现（2576/2578 行，客户端另多 387 个错误行）。两列仍在 v1/v2 的 13 维 `endpoint_red` 里；错误信号以 `client_error_rate`/`client_5xx_rate` 为准，`trace_request_count`/延迟列不受影响。PAPER_SPEC 说两列都"不可作为可用特征"，对 `trace_error_rate` 偏强。另：本批采集没有 `coverage_data` 模态，只有 trace/api_responses/metric/log 四种
