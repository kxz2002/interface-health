你要在 `interface-health` 仓库里执行一份已经写好的实现计划，把 `endpoint_raw2` 数据源里的 per-endpoint 精确故障标签（`is_target_endpoint`）接入现有的 Contract v0 评估 pipeline，最终让 `metrics.json` 产出真正意义上的 per-endpoint AUROC。

## 你需要读的文件（按顺序）

1. `docs/superpowers/specs/2026-07-08-per-endpoint-label-eval-design.md` —— 设计文档，讲清楚了**为什么**这么设计（关键决策、被否定的方案、探查确立的事实）。执行计划前必须先读这个，理解背后的取舍。
2. `docs/superpowers/plans/2026-07-08-per-endpoint-label-eval.md` —— 你要执行的完整实现计划，7 个 Task，每步都有具体代码和命令，不需要自己设计实现细节。

## 执行方式

用 **superpowers:subagent-driven-development** skill 执行这份计划：每个 Task 派一个独立的 subagent 去做，做完在两个阶段做 review，通过后再进入下一个 Task。不要自己在主 session 里直接改代码——计划已经把每一步的代码和测试都写死了，你的角色是派发和 review，不是重新设计。

## 开始前必须确认的环境状态

1. 检查当前分支：应该在 `feature/per-endpoint-label-eval` 上（`git branch --show-current`）。如果不在，先确认清楚为什么，不要自己随意切分支或创建新分支。
2. 检查 `docs/superpowers/plans/2026-07-08-per-endpoint-label-eval.md` 和对应的 spec 文件是否已经提交。如果还是 untracked/uncommitted 状态，正常——计划里的 Task 不涉及提交这两个文档本身，但如果你发现工作区有你不认识的未提交改动，先向用户确认，不要覆盖或丢弃。
3. 运行一次 `conda run -n interface python -m pytest tests/ -q` 确认当前测试基线是全绿的（可能有 1 个已知的 skip，属于既有行为），再开始改动。如果基线本身不绿，先停下来报告，不要在一个已经破损的基线上继续叠加改动。

## 关键背景（避免范围漂移）

- 这份计划是**纯增量改动**：不改 `is_anomaly`/`y_true` 的既有语义，不改 contract v0 现有的强校验规则。新增两列 `label_granularity`（`"endpoint"` = 精确标签 / `"case"` = case 级近似 fallback）和 `is_endpoint_anomaly`（= `is_target_endpoint AND phase=='inject'`；fallback 时等于 `is_anomaly`）。
- `eval_baseline_v0.py` 的四层分层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增的 `by_endpoint`）全部统一改用 `is_endpoint_anomaly`，`by_endpoint` 对全部数据统一按 `endpoint_key` 分组，**不**按数据源或 `label_granularity` 拆成两套评估体系——这是经过和用户来回讨论确认的关键设计决策（拆成两套体系是最初的错误设计，已被否定），如果执行过程中发现某个 Task 的实现思路又滑向"精确标签和近似标签分开算"，停下来重新对照 spec 的 §3.5，不要自己改回去。
- `train_baseline_v0.py` 的 `out_df` 是显式字段字典，新增列不会自动从 `eval_all.parquet` 带过去，必须手动加进字段列表——这是历史上 `is_target_endpoint` 被静默丢弃过一次的同一种坑，Task 4 专门写了回归测试防止再犯，务必让这个测试真的跑起来并且是有效的（不是加了列但测试没有实际验证数值一致性）。
- 每个 Task 内部是 TDD 节奏（先写失败测试→跑测试确认失败→写实现→跑测试确认通过→commit）。不要跳过"确认测试先失败"这一步，这是用来验证测试本身有效的，不是形式主义。

## 完成标准

按计划 Task 7 的"完成后自查清单"逐项核对：全量测试绿、`is_anomaly`/`y_true` 语义不变、`by_endpoint` 没有分裂成两套体系、`train_baseline_v0.py` 显式带上新列、文档和 `history/index.md` 都已更新、遗留 TODO 写进 history entry。全部确认后，向用户汇报完成情况，不要自行合并分支或推送远程。
