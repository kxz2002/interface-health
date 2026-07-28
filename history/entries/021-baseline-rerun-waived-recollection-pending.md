# 021 · 决定不重跑 v0/v1/RG 既有基线，MAX_CONTENT_CHARS drift 风险接受不处理

- **日期**: 2026-07-29
- **PR**: 待定（承接 entry 020 的 review 修复讨论）
- **类型**: Docs（决策记录，未改代码/未重跑任何 pipeline）
- **影响域**: `artifacts/baseline_v0/`, `artifacts/baseline_v1/`, `artifacts/baseline_v1_reliability_gate/`（决定维持现状，不重新构建）

## 做了什么

entry 020 量化 `MAX_CONTENT_CHARS` 截断的实测损耗之后，进一步核查发现该常量是本分支（`feature/deviation-weighted-fusion`）新引入、`master` 上不存在的逻辑，导致 v0/v1/v1_expanded（RG 专属）三条既有 contract 构建管线的 `dvc status` 显示 deps 已过期——已提交的 `metrics.json` 是旧代码（无截断）跑出来的数字，理论上重新 `dvc repro` 会产出不同结果。本条目记录讨论后的决定：不重跑这几条既有 pipeline，接受这个 drift 现状不做处理。

## 关键决策（不在 commit 里）

- **不补跑 v0/v1/RG 基线，而不是先补跑再合并**：表面看这是一个"技术债务"——代码变了但产物没同步更新。但用户指出一个更根本的判断：这批基线依赖的数据集（`merged_v2` = anomod_v1 + endpoint_raw2 + normal_v2）本身已经在 entry 017 被诊断出结构性缺口——PATCH 类故障"语义损坏但 HTTP 状态码可能仍是 200"，对所有 endpoint RED 特征隐形（oracle AUROC≈0.5），必须靠 api_response 提供响应体内容才能识别，而现有数据采集根本没有采集响应体。这意味着无论重跑多少次，这批数据集在 PATCH 类故障上都测不出真实信号——不是"代码 bug 导致结果不准"，是"数据本身缺这个模态"。既然这批数据集已经被判定需要重采（补采响应体）才有继续研究的价值，用旧代码重跑一遍旧数据集拿到的新数字，价值有限：它只是把"无截断"换成"截断"，测的还是同一批注定要被替换的数据，不会改变 PATCH 隐形这个已有结论，也不会产出任何后续实验会依赖的新信息。
- **接受 drift 现状，不做"标记为过期"式的处理**：本可以在 `metrics.json` 旁边加一个 stale 标记或在 CLAUDE.md 加一条"这几份数字是旧代码产物"的强提醒，但决定不做——因为下一步真实要做的事是重采数据（补 api_response 响应体），到那时这批 `merged_v2` 相关的 contract/baseline 会被整批替换，不是修一个字段能解决的局部问题，提前做标记式的缝补意义不大，直接留给重采后的下一轮工作。

## 坑 / 已知问题

- **"重跑成本"和"重跑价值"是两个独立问题，不能因为成本低就顺手做**：重跑 v0/v1/RG 三条 pipeline 本身技术上不难（`dvc repro`，约 20-30 分钟），但如果目标数据集本身即将被淘汰，重跑产出的数字不会被任何后续工作引用，纯粹是为了"让仓库看起来一致"而消耗时间——这不是本条目想鼓励的判断依据。判断是否重跑，应该先问"重跑后的数字会被谁使用、用来支撑什么结论"，而不是先问"重跑贵不贵"。
- **entry 019 的历史记录曾经引入过一个不准确的因果叙事**（见该 entry 本次的更正）：把"同一次会话里一起修的两个问题"误记成"分两轮修的"，本质原因是当时写 entry 时没有用 `git log -S` 核实"之前已修复"这个前提，凭对话记忆下了结论。这次的核查过程（`git log -S "MAX_CONTENT_CHARS"`、`dvc status`、跨数据集单文件规模对照实测）说明：涉及"过去是否已经处理过某问题"的表述，写 history 前最好用版本控制工具查证一次，而不是依赖上一轮 session 的措辞。

## 遗留 TODO

- 重采数据（补 api_response 响应体，解决 PATCH 类故障隐形问题）一旦启动，`merged_v2`（anomod_v1 + endpoint_raw2 + normal_v2）及其派生的 `contract_v0`/`contract_v1`/`contract_v1_expanded` 全部产物会被替换——`MAX_CONTENT_CHARS` drift 问题会随之自然消失（新数据集会用新代码从头构建），不需要单独修复旧产物。
- 若重采计划推迟或取消，`artifacts/baseline_v0|v1|v1_reliability_gate/metrics.json` 与当前代码的 drift 会一直存在，届时需要回头补跑；本条目只是记录"当前判断不需要"，不是"永久不需要"。
