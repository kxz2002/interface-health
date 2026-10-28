# 047 · 微服务领域神经基线调研与首轮适配

- **日期**: 2026-10-28
- **PR**: 待定 · **Commit**: fa6832e（修复后重跑）
- **类型**: Experiment / Model / Docs / Test
- **影响域**: scripts/service_trace_features.py、scripts/service_domain_baselines.py、tests/test_service_trace_features.py、docs/research/service-20261019/

## 做了什么

补充面向微服务/AIOps 的领域文献调研，核对 Triple、MSTGAD、AnoFusion、TraceGra、TraceVAE 和 TraceDAE 的输入粒度、公开代码和标签/阈值协议。对排除 POOL/DNS 后的 15 个 Case 扫描原始 SkyWalking Trace，确认服务调用图和请求级 span 结构完整；新增 Trace→Service 时间窗特征适配器，并实现 MSTGAD、AnoFusion、Triple 风格的纯 PyTorch 适配基线。

三个模型在 v3 `main_test`、Normal-only 训练和独立 Normal q99 校准下完成 seed 1/2/42 训练。修复 Trace 对齐、拓扑来源和 fit-only 标准化后，均值 AUROC/AP 分别为 MSTGAD 风格 0.8660/0.6764、AnoFusion 风格 0.7160/0.5799、Triple 风格 0.8906/0.5671；修复前数字作废。

## 关键决策

- **参考论文结构而不复制无许可证实现**：MSTGAD 仓库未发现许可证，代码只作为数据格式参考；实现采用独立纯 PyTorch 适配。
- **保留所有主评价 ID**：4 窗历史不足的 72 个窗口不删除，显式零分数并报告覆盖，避免通过删样本抬高结果。
- **不沿用论文阈值协议**：论文中的测试集阈值、EVT 或混合标签均不适用，统一改为独立 Normal calibration q99。

## 坑 / 已知问题

- **领域适配不等于完整复现**：当前输入是服务级聚合特征与 Trace 汇总，未完整复刻各论文的损失、在线流程和原始数据集。
- **Triple 风格误报高**：Normal FPR 均值约 0.173，不能只依据召回宣称其优于其他模型。
- **评价标签边界**：正类仍是受控关联证据，测试集已被多轮探索，不能称盲测或完整真实异常金标。

## 遗留 TODO

- 对三个模型生成逐样本 score、provenance、逐 Case 报告并完成独立重算审计。
- 若作为论文正式强基线，补充 Normal-only 内层验证和与官方实现的逐项差异说明。
