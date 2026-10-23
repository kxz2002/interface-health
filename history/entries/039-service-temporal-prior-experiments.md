# 039 · Service 时序先验、BERT 消融与 Normal-only 调参

- **日期**: 2026-10-23
- **PR**: 创建时登记于本分支 PR · **Commit**: 09bc287、bf5fdfd、a89bd1f、9271384
- **类型**: Model / Experiment
- **影响域**: src/service/modeling/、scripts/service_temporal*、configs/service/、tests/test_service*、docs/research/service-20261019/

## 做了什么

承接 038 的模型基础设施，对表示坍缩进行诊断，加入重建、可用性、固定尾概率融合及服务先验 GRU 对照；保留冻结 BERT 历史语义分支和有界 Normal-only 参数搜索。
真实结果、运行版本和 MLflow 标识见 model-collapse-investigation.md 与 temporal-tuning-protocol-20261023.md，不将大产物纳入 Git。

## 关键决策（不在 commit 里）

- 原先验 GRU 与 BERT 分支通过独立配置共存；未启用文本时不创建投影层，不修改原配置，不覆盖已有运行目录。
- 服务先验仅由 Normal fit 统计，严格过去上下文不跨 Case、Service 或划分边界；缺失目标仅按 mask 处理，Unknown 不补正常。
- 调参固定架构和候选集合，以完整 Normal Case 留出损失选择；内层重新拟合归一化、Drain、日志统计及服务先验，Fault 不参与选择。
- BERT 和参数搜索的负向发现完整保留，不能将验证损失改善表述成异常检测增益；已见测试集的后续工作均为探索。
- gpu-b 原仓库保留历史 dirty 状态；实验使用明确 commit 的独立目录，源码推送与计算服务器分开管理。

## 坑 / 已知问题

- 原 one-class 表示存在坍缩；新增复杂模块前应检查表示方差、梯度和独立分支分数。
- 当前 BERT 只给历史预测提供上下文，没有当前日志语义异常目标；不能称完整语义异常检测。
- 无文本与有文本 checkpoint 的结构不同，恢复必须匹配配置；分数比较必须核对精确 ID、校准集、版本与父 prepared。
- 全项目测试曾卡在旧 Endpoint selfref 冒烟流程；Service 范围测试通过不能冒充全项目通过。

## 遗留 TODO

- 原 Task10 九方法矩阵尚未全部完成，不因探索方法完成而关闭该项。
- 查新需结合静态先验、全局先验、自由服务偏置、时序和评分分支消融；不保证硕士创新评审通过。
- POOL/DNS 只读观测审计已完成，见 pool-dns-observability-audit-20261023.md；尚需核对干预路径与实际业务影响。注入关联标签不证明每窗都存在可检测业务影响，不据测试困难事件反调参数。
