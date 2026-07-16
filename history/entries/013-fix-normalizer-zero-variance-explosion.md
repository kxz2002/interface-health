# 013 · 修复 Normalizer 零方差 group 的除零放大 bug

- **日期**: 2026-07-16
- **PR**: bugfix/fix-endpoint-red-latency-corruption（待合并）
- **类型**: Bugfix
- **影响域**: `src/data/normalization.py`, `scripts/build_contract.py`, `tests/test_normalization.py`, `CLAUDE.md`

## 做了什么

修复 reliability-gate 融合机制数据信号分析阶段发现的问题：`artifacts/contract_v1/eval_all.parquet` 里 `endpoint_red__client_latency_p95` / `endpoint_red__latency_divergence` 两列约 25% 行含 1e9~1e13 量级的离谱数值（不是 NaN，是"算错了"）。

根因定位在 `src/data/normalization.py` 的 `Normalizer.transform()`：`per_endpoint_min_max` 归一化时，若某 endpoint 在 Normal fit 集合（`train_fit`，v1 时序切分出的那一部分）里该列只出现过单一取值（如 `POST:/api/v1/orderservice/order/refresh` 的 `client_latency_p95` fit 集合只有 1 个 unique 值 `22679.9015ms`），`lo == hi`，`hi - lo = 0`。旧实现用 `max(hi - lo, 1e-9)` 托底除数防止除零崩溃，但托底本身是 bug：eval 侧该 endpoint 的真实值只要与 fit 常量有任何差值，`(value - lo) / 1e-9` 就会被放大 1e9 倍。反推验证：某行归一化后值 `-1.726e13`，还原原始延迟 `5417.588ms`，量级完全合理，证实是放大而非脏数据。

排查过程中确认这不是 preprocessor 计算 bug（`ApiPreprocessor`/`TracePreprocessor` 独立跑单个 case 时输出干净），也不是 `build_contract.py:126` 的 trace/api inner join 问题（join 后、归一化前的 `ep_df` 数值正常）——问题严格发生在归一化这一步。

修复：把"跳过归一化"的判定条件从"仅 NaN"扩展到"NaN 或零方差"（`hi - lo < 1e-9`），去掉 `max(hi-lo, 1e-9)` 的除数托底，退化 group 统一跳过归一化、保留原始量纲。这是对 PR #7（entry 007）已有约定的直接延伸，不是新模式。

扫描全部归一化列发现零方差退化不止影响这两列，还有 `trace_error_rate`/`trace_5xx_rate`/`client_error_rate`/`client_5xx_rate`（受 `RATE_COLUMNS` 的 `clip(0,1)` 兜底掩盖了症状）、`service_metric__net_rx/tx_error_rate`、`service_metric__process_count`（不在 `RATE_COLUMNS` 里，同样实际爆到过 1e9 量级）、`service_log__error_ratio`。修复后重跑 `build_contract.py --config v1.yaml --dataset merged_v2.yaml` 全部消失，行数不变（14717 行），`pytest tests/`（195 passed, 2 skipped）与修复前一致。

## 关键决策（不在 commit 里）

- **跳过归一化保留原值，而非把除数下限设更小的正数**：曾考虑把 `1e-9` 换成更极端的下限（如 `1e-6` 或按列动态设阈），但这只是把爆炸阈值往后推，没有解决"lo==hi 时 min-max 本质上无法定义 scale"这个根本问题。跟 entry 007 全 NaN 的处理保持同一个哲学：统计量不可靠时，跳过运算比"硬凑一个能算的数"更安全。
- **判定用 `hi - lo < 1e-9` 而非 `hi == lo`**：float 精度下严格相等在理论上可能被浮点误差绕过（如 `hi-lo` 算出 `1e-15` 而非精确 0），用小量容差而非精确比较更稳健，代价是需要显式选定容差值——沿用旧实现里本来当托底用的 `1e-9`，含义从"除数下限"变成"退化判定阈值"，数值不变但角色变了。
- **不动 preprocessor/join 层**：最初怀疑是时间戳泄漏或 join 逻辑错误（memory 里 subagent 的初步推测），但复现后确认两处都是干净的，问题只在归一化这一步——避免了误修没问题的代码。

## 坑 / 已知问题

- **零方差退化的 blast radius 比最初报告的两列大得多**：最初只发现 `client_latency_p95`/`latency_divergence`，扫描 `normalization_stats.json` 全部列的 `lo==hi` 分组后发现还有 6 个其他列受影响，其中 `service_metric__process_count` 不受 `RATE_COLUMNS` 的 `clip(0,1)` 保护，实测爆到过 1e9。做任何"只查这两列"的局部修复都会漏掉这些同根同源的列，必须在 `Normalizer` 底层统一修。
- **零方差退化的根本原因是 Normal 数据量太小**：这几个 endpoint 在 v1 时序切分后的 `train_fit`（仅 838 行 Normal）里样本太少、且当前 Normal 数据源（`normal_v2`，2 个 case）该 endpoint 的调用频率低，导致该列在 fit 窗口内没有观测到真实变异——不是数据错误，是数据不够。跳过归一化只是让特征层面不产生数值污染，语义上这些列在这些 endpoint 上就是"该窗口没有可用的 scale 参照"，模型侧仍是弱信号。

## 遗留 TODO

- 若后续 Normal 数据（`normal_v2`）扩量或换 fit 窗口切分策略，这几个 endpoint 的零方差退化可能自然消失，届时 `skipped_groups()` 的告警列表会变化，属于预期行为，不需要额外处理。
- reliability-gate 分支（[[decision_contract_v1_temporal_split]]、`docs/superpowers/specs/2026-07-16-reliability-gate-fusion-design.md`）依赖的 z-score 偏离量计算，现在可以放心接这份修复后的 `artifacts/contract_v1`，不会再被 1e13 级数值污染。
