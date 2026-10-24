# 043 · 原GRU四方向独立验证

- **日期**: 2026-10-23
- **Commit**: d9f84b3、f34bf34、72f33dc、42fd350、f90ec82、cf71905
- **类型**: Model / Experiment / Test / Docs
- **影响域**: src/service/modeling/、scripts/service_gru_*、scripts/service_temporal_model.py、scripts/audit_gru_*、configs/service/、tests/test_service_*、tests/test_gru_*、docs/research/service-20261019/

## 做了什么

承接042结束后的新授权goal，独立实施服务×特征误差校准、正常状态与时序融合、服务条件化GRU、4/8窗双分支GRU。协议和实验结果分别保存在 `gru-four-directions-protocol.md` 与 `gru-four-directions-results.md`，原配置与产物保持独立。

PR阶段收尾追加 `model-exploration-closeout.md`，统一串起history040–043的报告与结果；逐项比对gpu-b历史dirty目录，保留旧文件，使用最终隔离worktree承载本分支同步副本。不重复为同一迭代新建history条目。

## 关键决策

- 四方向分别验证，不将上一方向设置叠加下一方向，不根据Fault评价反调参数。所有外层三seed预声明为1、2、42，不依单seed结果取消后续seed。
- Normal预测损失只用于诊断，不保证检测效果。前两方向复用原预测器和内部独立Normal checkpoint；后两方向重新训练，但数据、预处理、标签、评价集合和优化协议固定。
- 误差尺度和融合尺度由内部最新完整留出Normal Case估计，外层cal/test不参与参考拟合。内部三Case模型到最终四Case模型的坐标/分数迁移是近似，明确保留这一限制，不声称精确OOF。
- 服务条件化字典仅由fit服务建立，未知服务数值偏置为零；checkpoint显式记录架构、字典、参数量及偏置范数。双分支使用两个hidden16并不严格等参数，不把容量差异隐藏成纯历史消融。
- 远端使用Git bundle和全新隔离worktree，避免覆盖历史dirty仓库或另一session产物。源码和小型摘要入Git，完整输出、checkpoint、逐样本分数及MLflow数据库留远端。
- 审计从实际checkpoint恢复模型，比较冻结配置和fit-only字典；独立重算外层历史ID，核查同Case/Service/split、过去窗及15秒连续性。内层未保存context产物，不能将代码测试说成已核验实际内层历史产物。
- 四方向均完成三seed，未达到预声明稳定替代要求；短长历史均值双指标略升但seed1退步，保留为探索候选而不替换原GRU，不挑选表现较好seed。结束本轮四方向搜索，不自动新增第五方向。

## 坑 / 已知问题

- 记录器会向发布配置补充 `tracking_uri` 和 `experiment_name`，checkpoint只保存模型配置；审计必须显式验证这两个元数据字段，不能将其误判为模型漂移，也不能宽泛忽略任意未知字段。
- 完整评价分组含逐样本ID，直接复制可超过Git文件大小闸门。仓库只保留逐分组指标和必要训练历史，完整产物仍留远端；不绕过大文件钩子。
- 状态分支即使Normal参考稳健也可能出现外层长尾；固定max融合可能放大弱分支，不能预设更多分支必然改善检测。
- 这些结果来自反复查看过的历史测试集，只能作探索，不代表新故障盲测或独立故障重复，未保证超参数最优。

## 遗留 TODO

- 任何进一步搜索另立假设并优先取得新增盲测证据；本轮四方向完成后不自动扩展第五方向。
- 历史Task10完整矩阵不因本轮架构实验自动关闭；BERT和POOL/DNS不在本轮继续搜索。
