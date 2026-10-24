# Service模型探索阶段收尾

## 阶段结论

本阶段保留原服务先验GRU（4个历史窗、hidden32、50轮）作为当前研究主模型。冻结BERT历史融合、当前窗语义评分及七个GRU改进候选均保留独立配置与结果，没有覆盖原模型。部分候选的平均指标略升，但没有达到预声明的三seed AUROC/AP均改善要求；这不证明原模型或超参数全局最优。

## 已完成结果与报告

以下为相同主评价口径下seeds1/2/42均值，精确逐seed、q99误报/召回、分Case/服务及限制以各报告为准。

| 方案 | AUROC | AP | 证据 |
|---|---:|---:|---|
| 原服务先验GRU | 0.823970 | 0.680061 | [路线A](route-a-final-analysis.md) |
| 默认历史BERT＋GRU | 0.813235 | 0.667020 | [路线A](route-a-final-analysis.md) |
| Normal调参历史BERT＋GRU | 0.812421 | 0.670064 | [调参协议](temporal-tuning-protocol-20261023.md) |
| 无可用性先验 | 0.641188 | 0.481034 | [机制消融](route-a-final-analysis.md) |
| 全局可用性先验 | 0.647154 | 0.488636 | [机制消融](route-a-final-analysis.md) |
| BERT零输入控制 | 0.823583 | 0.674574 | [机制消融](route-a-final-analysis.md) |
| BERT过去历史逆序 | 0.813585 | 0.666777 | [机制消融](route-a-final-analysis.md) |
| 当前窗BERT语义-only | 0.632881 | 0.175363 | [当前窗评分](current-window-semantic-results.md) |
| 当前窗语义＋GRU | 0.671317 | 0.413700 | [当前窗评分](current-window-semantic-results.md) |
| GRU历史8窗 | 0.825033 | 0.688651 | [前三候选](gru-improvement-analysis.md) |
| GRU hidden64 | 0.825666 | 0.671800 | [前三候选](gru-improvement-analysis.md) |
| 因果残差GRU | 0.802927 | 0.569942 | [前三候选](gru-improvement-analysis.md) |
| 服务×特征误差校准 | 0.821468 | 0.688617 | [四方向](gru-four-directions-results.md) |
| Normal状态＋GRU max融合 | 0.760610 | 0.548412 | [四方向](gru-four-directions-results.md) |
| 服务条件化GRU | 0.821220 | 0.678095 | [四方向](gru-four-directions-results.md) |
| 短长历史GRU | 0.824664 | 0.686470 | [四方向](gru-four-directions-results.md) |

这些行不是完整Task10矩阵，也不包含全部历史方法。状态-only辅助对照见四方向报告；IF、OCSVM、原CLS及训练坍缩诊断见 [Task9/10](model-task9-gpu-b.md) 和 [模型诊断](model-collapse-investigation.md)，不将不同输入或单seed结果混写为同一多seed消融。

## 可以支持的判断

- 服务条件可用性先验在本数据上有明显贡献，不能单凭这一消融证明GRU时序贡献、新颖性或跨系统泛化。
- 当前冻结BERT表示及两种已测试接入方式没有提供稳定增量，不据此否定日志语义的所有路线。
- 更低Normal预测误差不保证更好检测：残差Normal总损失下降约23.3%，检测却下降。增加容量或分支也没有保证收益。
- 8窗与短长历史有有限均值收益，但随机seed与q99风险未稳定改善，继续保留原GRU，不挑选较好seed。
- 状态max融合会受弱分支长尾影响；误差缩放可能同时放大正常尾部。它们是本实现的失败证据，不是全部同类方法的否定结论。

## 协议和结论边界

selected17、整Case隔离、Normal-only拟合、Unknown排除不变。主评价3297窗（445正、2852负）、外层Normal校准2853窗、完整测试12598窗。异常测试不用于拟合参考或挑阈值。内部Normal验证来自outer fit，前三Case拟合预处理、最新完整Case留出。

同一测试集已经多轮查看，全部结论属于探索；较早候选存在先看外层结果后补Normal验证的限制。三seed只反映训练随机性，不是独立故障重复。误差/融合参考由内部模型迁移至最终四Case模型是近似，不是最终模型的精确OOF。当前窗语义融合曾用外层Normal校准估计融合尺度，独立性与后续四方向协议不同，不能统称所有路线都只用内部Normal选融合参考。

本session停止继续BERT、POOL/DNS及第五方向搜索。POOL/DNS的观测/干预证据与另一worktree探索保持独立。历史Task10全矩阵和新增数据盲测仍未完成。

## 文件保留和同步

- 本机工作位置：`.worktrees/model-improvement`，分支 `exp/service-model-improvement`；不修改主目录master或另一session的worktree。
- gpu-b计算输出：`~/Code/Repos/interface-health/outputs/service/`；raw、checkpoint、逐样本Parquet、日志与MLflow数据库留远端，不加入PR。源码、配置、测试、协议和精简结果摘要入Git。
- 截至收尾检查，远端最终隔离worktree `~/Code/Repos/interface-health-runs/gru-four-final-de5a31d` 与本机实验收尾提交 `de5a31d` 一致且clean。新增阶段文档提交后另部署新的最终隔离worktree，并对全部Git跟踪文件逐项核对SHA256。
- 远端旧目录 `~/Code/Repos/interface-health` 仍是历史 `feature/service-model-infrastructure`、HEAD `21fdc3e`，不能称它与本机当前分支一致。其76个未提交文件中，53个与最终已提交文件完全一致，15个匹配Git中的历史版本，其余8个已逐项diff：1份旧报告、4个旧功能/测试版本、3个仅排版不同的文件；未发现需要从远端新增回收的独有实现。保留全部旧文件，不reset/clean。
- 八项diff具体覆盖：`model-task9-gpu-b.md`、`src/service/modeling/{baselines,numeric,training}.py`、`tests/test_service_model_{baselines,cli,evaluation,training}.py`。最终已提交版本包含后续修复或相同逻辑的格式规范，不反向复制旧文件。
- 四方向后台PID已退出，未启动新训练。只读同步与验证不更改实验输出。

## 接手入口

原模型及候选配置选择见 [运行说明](temporal-model-running.md)。四方向精确run ID、模型commit及验证证据见 [四方向报告](gru-four-directions-results.md)；前轮完整MLflow核验见 [完成核查](gru-improvement-completion-audit.md)。历史原因和遗留事项见history040–043，不重复新建同一迭代的history entry。

下一阶段应先明确独立验证数据与要验证的机制，再决定是否扩大搜索；本PR汇总已完成的实现和证据，不声称论文创新已得到最终认证或Task10全部完成。

## PR收尾验证

- 原GRU及默认BERT配置相对PR基线 `bd3e441` 无diff。
- 本分支23个变更Python文件逐文件Black检查与isort检查通过；此前四方向联合96项测试通过。
- gpu-b完整回归需要 `PYTHONPATH` 指向隔离源码根目录；未设置时子进程导入 `src` 失败。首次错误环境检查已停止，不能将它记为通过。补充 `PYTHONPATH="$PWD"`、固定线程与离线环境后，`python -m pytest tests/ -x -q` 在 `de5a31d` 隔离代码上完成：950 passed、5 skipped、14条第三方弃用警告，耗时667.48秒。之后仅修改阶段文档，未改变被测Python源码。日志留gpu-b `/tmp/service-closeout-pytest-fixed.log`。
- `6cafe95` 收尾副本本机/gpu-b均clean，924个Git跟踪文件的路径及内容SHA256聚合一致：`dbec144dbfec63805cd627ebf36d38380d421a67a530f84e89113b2ed8f30290`。追加测试记录后再次同步至最终收尾提交，保留所有旧隔离副本。
