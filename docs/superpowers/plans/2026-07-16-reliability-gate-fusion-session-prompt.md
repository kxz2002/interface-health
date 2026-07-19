你要在 `interface-health` 仓库里执行一份已经写好的实现计划——把 Deviation-Conditioned Reliability Gate Fusion（`ReliabilityGatedFusion` + `EndpointBaselineStats` + Contract v1 训练池扩容）接入现有的 Hydra 可插拔融合机制，跑通实验并产出对比结果。

## 你需要读的文件（按顺序）

1. `docs/superpowers/specs/2026-07-16-reliability-gate-fusion-design.md` —— 设计文档，讲清楚**为什么**这么设计（两分支偏离量表示、softmax 竞争性门控、训练池扩容的取舍、已识别的风险）。执行计划前必须先读这个。
2. `docs/superpowers/plans/2026-07-16-reliability-gate-fusion.md` —— 你要执行的完整实现计划，9 个 Task，每步都有具体代码、测试和命令，不需要自己重新设计实现细节。

## 执行方式

用 **superpowers:subagent-driven-development** skill 执行这份计划：每个 Task 派一个独立的 subagent 去做，做完在两个阶段做 review，通过后再进入下一个 Task。不要自己在主 session 里直接改代码——计划已经把每一步的代码和测试都写死了，你的角色是派发和 review，不是重新设计。

Task 8/9 涉及真实训练/评估（多 seed 主对比、2×2 训练池归因、路由消融、可解释性分析），在真实 `merged_v2` 数据上跑会比较慢——这些步骤如实执行、如实记录数值，不要用占位符糊弄过去；耗时长的可以在 subagent 内用后台方式跑，但结果必须真的等到、真的读到再写进 history entry。

## 开始前必须确认的环境状态

1. 检查当前分支：应该在 `feature/reliability-gate-fusion` 上（`git branch --show-current`）。
2. 工作区里已有一批与本计划无关的 untracked 文件/目录（`NOVELTY_DOSSIER.md`、`artifacts/l1*`、`artifacts/l2*`、`data/endpoint_raw2/`、`data/normal_v2/`）——这些是之前实验遗留的产物，不要删除也不要纳入本轮 commit（除非计划某个 Task 明确要求改动它们）。
3. 运行一次 `conda run -n interface python -m pytest tests/ -q` 确认当前测试基线全绿，再开始改动。如果基线本身不绿，先停下来报告，不要在破损基线上继续叠加改动。所有 Python 命令都要用 `conda run -n interface`（repo 约定环境，不是 conda base——之前直接用 `python3` 会报 `ModuleNotFoundError: No module named 'src'`）。

## 关键背景（避免范围漂移）

- 前置阻塞项已解除：spec §6 点名的 Normalizer 零方差除零放大 bug 已在 PR#13（history entry 013）修复，当前 `artifacts/contract_v1/` 是修复后的产物，可直接使用，不需要重新排查。
- **防泄漏是硬约束，不是可选加固**：`EndpointBaselineStats`/`Normalizer` 的 fit 范围必须严格锁在纯 Normal 的 `train_fit`（不是扩容后的 `train.parquet`，也不能碰 `eval_normal_holdout`）；Task 6 训练池扩容后，`eval_all` 必须同步摘除被吸收的故障 case baseline 行——计划里各有专门测试钉住（`test_build_contract_v1_normalizer_excludes_holdout`、`test_eval_all_excludes_train_pool_rows_no_leakage`）。如果执行中发现某个改动会让这些测试"需要改断言才能通过"，停下来——这说明泄漏被引入了，不要放宽断言迁就实现。
- 训练池只吸收故障 case 的 **baseline** 阶段行，不吸收 **recover**（系统未稳定回正常态，分布未验证）——如果发现实现滑向"顺便也吸收 recover"，停下来对照 spec §3.4 确认。
- `ReliabilityGatedFusion.forward` 的 `endpoint_id` 参数缺省时必须退化为均匀权重 `[0.5,0.5]`（不查表）——这是向后兼容 L0/L1/L2 现有调用方式的关键行为，不要在实现中要求 `endpoint_id` 必填。
- Task 8/9 产出的 AUROC/AUPRC 等数值必须是真实跑出来的（参照 entry 011/012 的既有约定），不要用占位符占坑；如果某个消融/归因实验结果不如预期（如 gate 在困难样本上无优势），如实记入 history entry，不要掩盖。
- 每个 Task 内部是 TDD 节奏（先写失败测试→跑测试确认失败→写实现→跑测试确认通过→commit）。不要跳过"确认测试先失败"这一步，这是用来验证测试本身有效的，不是形式主义。

## 完成标准

按计划末尾"最终验证"清单逐项核对：全量测试绿、v0/v1 既有 pipeline（L0-L2）冒烟不受影响、reliability_gate 新链路端到端冒烟通过、spec §4 的五项完成标准（主对比/训练池归因/路由消融/可解释性/稳健性）逐项自查、history entry 014 已写且 `history/index.md` 已更新、spec §7 明确排除项已搬入遗留 TODO。全部确认后，向用户汇报完成情况（含关键实测数值），不要自行合并分支或推送远程。
