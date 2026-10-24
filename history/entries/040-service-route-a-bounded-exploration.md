# 040 · Service 路线 A 有界机制探索

- **日期**: 2026-10-23
- **PR / Commit**: exp/service-model-improvement；8bb258c、3ad3675、a8e9e30、2d71a39
- **类型**: Experiment / Model / Docs
- **影响域**: src/service/modeling/temporal.py、scripts/service_temporal_model.py、scripts/analyze_service_route_a.py、configs/service/、tests/test_service_model_*、docs/research/service-20261019/

## 做了什么

承接039，在独立worktree及gpu-b明确commit部署中完成无先验/全局先验、冻结BERT零输入/因果历史逆序两轮机制实验，各三seed。统一复核历史九组及新增十二组的声明产物、父链和精确样本身份，加入只读可复跑汇总脚本。结果及证据见`docs/research/service-20261019/route-a-final-analysis.md`，不在history重复固化指标。

## 关键决策

- 原无BERT服务先验GRU仍为主基线，冻结BERT保留可运行的负向对照；独立配置及输出共存，不覆盖checkpoint。
- 全局先验只汇总Normal fit计数，对总计数做一次Beta平滑。服务条件参考的贡献与时序贡献分开解释，未因该对照退步宣称GRU必需。
- zero保留BERT参数结构；reverse_past仅逆序同Case/Service/split过去历史的日志有效位置，保留数值历史；需报告实际变化覆盖，不能等同原模型或随机语义破坏。
- 单seed只验收实现和产物，预定全部三seed；Fault不参与候选/参数/阈值/特征选择，独立Normal校准不兼作选参验证。
- 两轮无稳定改善，结合完整Normal Case留出搜索的过训练证据，按预定规则停止扩大模型/BERT搜索。这是有界研究决定，不声称参数最优或其他BERT路线无效。
- POOL/DNS未检出与未证明可检测影响分开，不改标签或删除事件；优先另一路新版本受控复现及直接影响证据。

## 坑 / 已知问题

- zero与无BERT结构、初始化路径不同，指标近似不能解释为逐样本复现。reverse仅部分prepared窗口变化，不能把全prepared覆盖当作主集覆盖。
- BERT没有独立语义评分；更低Normal预测损失不保证检测增益。服务内排序、总体排序与全局阈值召回不可混同。
- 三seed不等于独立故障重复；单类Case AUC/AP保留null，KILL极少正窗不支持泛化声明。
- 本机多文件Black曾挂起，逐文件成功后仅跳过该hook；不要批量终止其他session进程。

## 遗留 TODO

- 039的完整Task10矩阵、静态规则/自由偏置及独立时序贡献消融未全部完成，不因本轮结束而关闭。
- 重启模型探索需新的独立Normal验证/盲测及预注册假设，不用本轮Fault反向调参。
- POOL/DNS受控复现由另一路推进，本轮未新增采集或替代该任务。
