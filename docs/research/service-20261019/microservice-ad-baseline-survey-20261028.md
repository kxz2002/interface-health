# 微服务异常检测领域基线调研（第一轮）

本轮专门检索面向微服务或 AIOps 的异常检测论文，区别于此前的通用时间序列模型调研。检索重点是：输入是否包含服务依赖、Trace、Metric、Log；模型是否为神经网络；能否改造成当前 Service×15 秒、Normal-only 训练和独立 Normal 校准协议。

## 候选论文

| 论文 | 年份/ venue | 原始输入与任务 | 模型 | 对当前任务的适配判断 |
|---|---|---|---|---|
| **Triple: The Interpretable Deep Learning Anomaly Detection Framework based on Trace-Metric-Log of Microservice** | IWQoS 2023 | Trace、Metric、Log，多模态微服务异常检测 | STGCN + Deep SVDD，并提供模态解释 | 最贴近当前 Service 任务；可用固定服务依赖图、时间窗内多模态向量和 Normal-only SVDD 适配，但必须明确当前数据缺少完整请求级 Trace 图时的降级方式。 |
| **MSTGAD: Twin Graph-based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System** | 2023（arXiv 2310.04701） | 多模态微服务系统状态和服务图 | 空间/时间注意力 Transformer 图模型，半监督 | 适合服务依赖建模，但半监督设定和图级标签需要改成 Normal-only；代码公开，值得优先核对。 |
| **TraceGra: A trace-based anomaly detection for microservice using graph deep learning** | Computer Communications 2023 | Trace 调用图、响应时间和资源消耗 | GNN + LSTM encoder-decoder | 直接面向 Trace，但当前主任务是 Service 窗口而不是请求 Trace；只有在能稳定构造服务调用图时才适合复现。 |
| **Unsupervised Anomaly Detection on Microservice Traces through Graph VAE (TraceVAE)** | WWW 2023 | 请求级树状 Trace、节点耗时与类别 | 双变量 Graph VAE，重建和 NLL 异常分数 | 领域相关性强且是无监督模型，但需要请求级树结构；不能把当前聚合 Metric 行伪装成 TraceVAE 输入。 |
| **AnoFusion: Robust Multimodal Failure Detection for Microservice Systems** | KDD 2023 | 实例级 Metric、Log、Trace 等多模态观测 | 多模态深度融合与图结构 | 与当前多模态方向接近，但原任务偏实例故障检测；需核对公开代码、标签和是否可抽取服务级窗口分数。 |
| **TraceDAE: Trace-Based Anomaly Detection in Microservice Systems via Dual Autoencoder** | IEEE TNSM 2025 | 请求 Trace 及其结构/时延 | 双自编码器 | 时间较新，但主要是 Trace 粒度，当前数据是否有足够请求级结构是准入门槛。 |
| **ServiceAnomaly** | JSS 2024 | 分布式 Trace + 六类 profiling metrics | 标注 DAG 的正常行为建模 | 不是纯深度网络，但领域适配价值高，可作为非神经领域参照，不计入三种神经基线。 |

## 数据格式与公平性问题

领域论文常见三种输入格式：

1. **请求级 Trace 树/图**：节点是 Service/Span，边是调用关系，节点带 latency、status、资源字段。TraceVAE、TraceGra、TraceDAE 属于这一类。当前数据的 Service 15 秒聚合行不能直接替代请求级树；若没有稳定的 Trace ID 和完整父子关系，只能做明确降级适配。
2. **服务依赖图 + 时间窗特征**：节点是 Service，边来自部署或调用拓扑，每个时间窗附带 Metric/Log/Trace 特征。Triple、MSTGAD、AnoFusion 更接近这一类，也最可能映射到当前任务。
3. **实例级多模态事件序列**：以实例或故障事件为单位，将 Metric、Log、Trace 编码后做检测或诊断。其标签通常是 failure/event 级，不能直接与当前 Service×Window 标签混用。

当前协议要求：只用独立 Normal fit 训练，只用独立 Normal calibration 选阈值，故障仅做最终评价；Unknown 不转正常；POOL/DNS 排除；不使用 point adjustment 或测试集阈值。原论文若使用故障验证、混合训练、测试集阈值或事件级标签，必须重写评测部分并记录为适配，而非声称完整复现。

## 初步优先级

优先核查 **Triple、MSTGAD、AnoFusion**：它们最可能使用服务图和多模态窗口，能够在当前 Service 任务保留问题粒度。TraceGra/TraceVAE/TraceDAE 作为第二梯队，除非确认当前远端数据保留可用的请求级 Trace 图，否则不强行运行。ServiceAnomaly 作为领域经典非神经参照。

下一轮需要逐篇核对官方代码、许可证、训练标签、数据下载方式和输入张量，并在当前仓库检查能否从 prepared 数据生成同粒度输入。只有通过这些准入检查的模型才进入实验。

## 第一轮代码核查

- **MSTGAD** 官方仓库可获得，要求 MSDS 数据的 `metric.csv`、`log.csv`、`trace.csv`、`label.pkl` 和 `trace_path.pkl`。实现按服务实例节点构造时间窗，Metric 形状为 `window × num_nodes × metric_dim`，Log 形状为 `window × num_nodes × log_dim`，Trace 形状为 `window × caller × callee × trace_type`，并使用预先给定的实例图。它是半监督实现，原始标签 mask 和 MSDS 特定预处理不能直接带入当前协议；但数据张量结构与当前 Service 图适配方向最接近。
- **TraceVAE** 官方仓库可获得且为 MIT License，输入是请求级树状 Trace 的预处理 CSV，不是聚合 Service 表。其命令会把原始 Trace 转成图数据，再训练 Graph VAE；如果当前数据无法恢复稳定的 parent/child span 和 request ID，就不应把它改造成普通时间序列后宣称复现。
- **AnoFusion** 论文明确采用每分钟同步的 Metric、Log、Trace 通道，把每个通道作为异构图节点，再由 GTN、GAT 和 GRU 建模；它是无监督训练，但原文使用 EVT 选择 F1 阈值，必须替换为独立 Normal calibration。其输入时间粒度和当前 15 秒窗口存在可调差异，需保留在适配记录中。

因此，下一轮优先级暂定为 **MSTGAD → AnoFusion → Triple**；TraceVAE/TraceGra/TraceDAE 先做数据可用性检查。若当前 Service 数据不能构造服务依赖图，领域模型只能作为协议适配研究，不能声称复现原论文的 Trace 级任务。

## 来源

- [Triple（IEEE/ACM IWQoS 2023）](https://doi.org/10.1109/IWQoS57198.2023.10188773)
- [MSTGAD（arXiv:2310.04701）](https://arxiv.org/abs/2310.04701)
- [TraceGra（Computer Communications 2023）](https://doi.org/10.1016/j.comcom.2023.03.028)
- [TraceVAE（WWW 2023）](https://doi.org/10.1145/3543507.3583215)
- [AnoFusion（KDD 2023）](https://marvin233.github.io/paper/kdd23-AnoFusion.pdf)
- [TraceDAE（IEEE TNSM 2025）](https://doi.org/10.1109/TNSM.2025.3583213)
- [ServiceAnomaly（JSS 2024）](https://doi.org/10.1016/j.jss.2023.111917)
