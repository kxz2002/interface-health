# 042 · 原先验GRU的有界Normal-only改进

- **日期**: 2026-10-24
- **Commit**: a41aa70、da7f843、0e495c0、8ebf4db、50400a7、8ce7289
- **类型**: Model / Experiment / Test / Docs
- **影响域**: src/service/modeling/temporal.py、scripts/service_temporal_model.py、scripts/service_gru_normal_validation.py、scripts/audit_gru_candidates.py、configs/service/、tests/test_service_model_temporal.py、tests/test_gru_*、docs/research/service-20261019/

## 做了什么

承接041的原GRU改进任务，冻结更长上下文、更大隐藏维度及因果残差预测三个候选，采用独立配置、提交和run_id。原配置、checkpoint及输出保留。第一轮与最终结果、MLflow曲线和Normal误差分析见`docs/research/service-20261019/gru-improvement-analysis.md`及关联机器可读摘要。

## 关键决策

- 历史上下文检验更长过去信息，hidden64检验容量；不把已见Fault结果用于组合二者或调整损失、阈值、特征。原校准Case仅作阈值校准，内部验证使用outer fit中的最新完整Normal Case，重新拟合所有预处理。
- Normal持久性诊断支持第三个因果残差候选：每字段从原4窗内选择最近有效历史值，无历史回退0，GRU学习修正。维度、初始化、优化和可用性先验保持原方案。只在Normal三seed逐对总损失和数值误差均改善后进入外层固定三seed评估。
- 所有外层seed预先固定1、2、42，不以单seed结果决定是否完成剩余seed。第三候选结束后停止本轮，不再扩展第四候选。
- 真实MLflow以只读SQLite读取，与声明训练历史逐epoch核对；产物校验与精确样本、prepared父引用及版本一致性分别核查。保留逐seed及q99误报/召回，单类分组AUROC/AP保持null。
- 推荐模型的保留要求为无泄漏且三个seed的AUROC和AP均改善。探索配置和输出作为可复现记录留存，不自动替换原模型。当前窗BERT及POOL/DNS不在本轮继续搜索。
- 三个候选均未满足该保留门槛：残差虽通过Normal验证，却在三个seed的检测指标上全部下降。完成本轮有界搜索并保留原GRU，不根据已见失败结果扩展第四候选。消化041的原GRU曲线审阅与有限候选验证TODO，不关闭历史Task10。

## 坑 / 已知问题

- 首轮已先看外层结果后补Normal验证，整个研究仍属已见测试集后的探索；不能把补验证说成盲测，三seed不代表独立故障重复或显著性。
- Normal预测误差下降不保证异常检出改善；残差跳连可能把持续异常电平当作参考。不得根据某故障退步反调跳连、窗口或评分。
- 残差和原GRU的参数形状相同，恢复时必须读取checkpoint的`config.training.value_prediction`；旧配置默认absolute，不能只按state_dict形状推断架构语义。
- 多文件Black曾挂起；改为逐文件完成，再单独运行其余pre-commit检查，不停止其他session进程。

## 遗留 TODO

- 未来任何新搜索需另立假设和独立Normal验证，优先新增盲测数据；本轮不承诺超参数最优。
- 039/040记录的历史Task10全矩阵及独立时序贡献消融不因本轮有界架构搜索而自动关闭。
