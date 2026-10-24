# 044 · 原GRU八方向探索与实际产物审计

- **日期**: 2026-10-24
- **Commit / PR**: 分支 `exp/service-model-improvement-session`；实验基础HEAD f3bacdc，包含2df99b8。运行时overlay由每run源码SHA256绑定，收尾统一提交源码、配置、测试与小型证据，并向master提出PR。
- **类型**: Model / Experiment / Test / Docs
- **影响域**: src/service/modeling/、scripts/service_eight_directions.py、scripts/audit_service_eight_directions.py、configs/service/、tests/test_service_eight_directions.py、docs/research/service-20261019/

## 做了什么

承接用户在043收尾后的新goal授权，按冻结优先级完成双头、多尺度、分组GRU、原型、机制组合、异方差、TCN-GRU和遮蔽预训练。详细协议、结果及机器可检查摘要见[八方向报告](../../docs/research/service-20261019/gru-eight-directions-results.md)，不在history重复结果数据。

## 关键决策

- 每方向均补齐预声明seed，不按首个seed或Fault结果淘汰、追加模型或调整超参数。Normal预测改善不等于检测改善，不同训练目标的loss不能直接比较。
- 原型和机制复用同seed原GRU，避免训练随机差异混入分数组合效应。内层留出Normal同时提供组合尺度，因此组合的内层分数只作诊断；外层独立Normal校准阈值。
- 内层重新拟合数值和日志预处理，保存实际内外fit与context IDs。审计恢复实际checkpoint、重算分数和完整评价，并只读核验MLflow逐epoch记录；历史原GRU另做逐样本完全相等对照。
- 原远端dirty仓库保留，通过bundle与明确文件清单将本机代码同步到独立worktree；源码hash校验后训练。大产物与MLflow留远端，只将小型审计摘要入仓库。
- 收尾逐文件核对本机与gpu-b隔离副本，仅将本轮相关文件提交；用户本机的未跟踪代理配置及远端原仓库历史修改均保留。隔离副本的运行时overlay先作限定路径stash备份，再快进到本机最终提交，保留原实验provenance与源码快照。
- 分组GRU均值提升但未通过预声明逐seed稳定门槛，因此保留原GRU默认。参数量略增，未开展等参或去门控消融，不能将均值收益当作独立结构贡献。
- 本轮是已看过测试集后的探索，不称盲测或独立故障重复；完成八方向后结束该goal，不自动扩展第九方向，也不自动关闭历史Task10矩阵。

## 坑 / 已知问题

- 内存dataclass含tuple，JSON落盘为list；直接对象比较会误报预处理漂移。审计先统一JSON表示，再比较全部状态和hash；测试确保真实统计变化仍被拒绝。
- CPU与CUDA的GRU重放存在浮点差异。严格分数审计使用冻结配置的训练设备和相同随机/后端设置，不以放宽容差掩盖差异；不能承诺跨设备逐值复现。
- PR #39独立审查发现：若持久化history与MLflow同时缺少部分epoch，原检查只遍历已有记录会漏报，甚至允许某个scope为空。审计现按冻结配置要求完整inner/outer阶段与epoch顺序，并核对MLflow训练记录总数和最终指标唯一性。该修复只加强审计，不改变冻结训练代码、模型或评分。
- 原型尺度从内层模型迁到外层模型只是一种固定近似，可能恶化排序；异方差NLL还需保存原坐标MSE供诊断。
- POOL/DNS盲区没有因更换结构自动消失；缺少直接观测时，不能把注入等同已知全服务异常。

## 遗留 TODO

- 若后续另获授权，对分组GRU开展等参、去门控消融及新增独立Case验证；当前结果不足以稳定替代默认模型。
- POOL/DNS优先补充可观测性与实际业务影响证据，不在本goal继续盲目增加模型。
