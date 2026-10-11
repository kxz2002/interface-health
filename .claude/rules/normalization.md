---
paths:
  - "src/data/normalization.py"
  - "src/data/endpoint_baseline_stats.py"
  - "scripts/build_contract.py"
  - "tests/test_normalization.py"
  - "tests/test_endpoint_baseline_stats.py"
  - "tests/test_build_contract_v2_fit_scope.py"
  - "tests/test_build_contract_nan_propagation.py"
---

# 归一化

从根 CLAUDE.md 的 Known Gotchas 迁出（entry 033），原文保留。v2 的量级 sanity 见 `contract-and-labels.md`。

- **Normalizer 退化 group（全 NaN 或零方差）不做归一化，保留原值**：某 group（或 global scope）在 fit 集合（仅 Normal）里全 NaN，或只出现过单一取值（`hi-lo=0`）时，`transform()` 必须跳过该 group 的归一化运算——全 NaN 会把 `[nan, nan]` 统计量通过减法/除法扩散污染其他数据完好的 case；零方差若不跳过、只是把除数下限到 `1e-9`，会把 eval 侧任何非零差值放大 1e9 倍，产出 1e9~1e13 量级的离谱数值（`endpoint_red__client_latency_p95`/`latency_divergence` 曾实际命中，2026-07-16 修复，见 history 013）。跳过后保留原始量纲；若某列同时受 `RATE_COLUMNS` 的 `clip(0,1)` 约束，需确认原始量纲天然落在 `[0,1]`，否则 clip 会把正常值误判为异常。**以上是 min_max（v0/v1）行为；v2 的 z_score 路径不同**：退化统计量走"一级 shrinkage（组 → 跨 case 汇总，w=n/(n+10)）+ 两级退化回退（汇总 → 全局 → 1.0 哨兵）"的回退链，不存在"跳过/透传原始量纲"的组（per-case 下原始值跨 case 不可比），`skipped_groups()` 恒不含 z_score 列（见 history/entries/031）
- **fit 范围**：v1 的 `EndpointBaselineStats`/`Normalizer` 始终只在纯 Normal 的 `train_fit` 上 fit，不受 `expand_train_pool` 影响；v2 的 Normalizer fit = `train_fit` ∪ 各故障 case baseline 前 20%，且不产 EBS sidecar（shrinkage 已收进 Normalizer，见 history/entries/031）
