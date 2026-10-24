# 原GRU有界改进：完成核查

范围以本轮goal为准：最多三个Normal-only候选，保留稳定且无泄漏的改进；不承诺一定发现胜出架构，不把历史Task10矩阵纳入本轮已完成事项。

|要求|可检查依据|结论|
|---|---|---|
|保留原模型、配置、checkpoint和输出|原配置相对`e46750c`无diff；全部新运行使用独立输出名；原三seed声明产物仍可核验，指标未变化；232项相关测试包含旧绝对预测、BERT及checkpoint兼容路径|满足|
|审阅真实MLflow过程及误差分量|`audit_gru_candidates.py`按manifest的tracking URI只读查询；最终JSON保存逐epoch及Normal数值/可用性/服务分量；750轮记录与训练历史完全一致|满足|
|最多三个明确机制候选|协议及提交`a41aa70`、`da7f843`、`50400a7`分别对应历史长度、容量、因果残差；各三seed已发布|满足|
|Normal-only选择，校准与测试隔离|内部实际fit7291、validation2857来自outer fit四个Normal Case，预处理重新拟合；残差预处理hash及精确ID与原内部验证一致；闸门证据提交`8ce7289`；不调整原阈值规则|满足本轮选择边界；首轮先见外层结果的探索偏差明确保留|
|固定主评价协议及无新增时序泄漏|最终审计检查12个外层run的fit/cal/test/main精确ID、版本和prepared父绑定一致；因果上下文边界、缺观测参考、Normal-only训练及checkpoint测试通过；残差参考只读取原历史context|满足新增架构范围；不宣称对整个数据集完成独立真值认证|
|只推荐三seed稳定改善的候选|三个候选的`paired_metric_gate`均false，未改默认原GRU配置；失败配置和输出只作为探索记录|保留原GRU，不推广新候选|
|失败分析、文档及history|`gru-improvement-analysis.md`给出逐seed、q99和事件退步、Normal与检测目标不一致及解释边界；history042及index已更新|满足|
|不继续BERT或POOL/DNS搜索|本轮新配置仅上述三个候选，未新增语义训练或相关故障搜索；已有记录保持|满足|
|代码测试|232项相关测试通过；后续审计汇总变更的闸门测试4项通过；Black逐文件与其他pre-commit检查完成|满足|

最终摘要为`gru-improvement-final-audit.json`，绑定运行commit、MLflow run ID和审计脚本SHA256。原始数据、模型权重、逐样本Parquet及MLflow数据库仍在gpu-b。收尾提交及远端独立worktree的HEAD/clean状态、关键文件SHA256由最后同步命令核验；不会自动push或创建PR。

研究结论为“这三个有界候选均未稳定整体胜出”，不是“原GRU全局最优”。后续若继续搜索，应另立假设并优先增加独立盲测数据，本轮不扩展第四候选。
