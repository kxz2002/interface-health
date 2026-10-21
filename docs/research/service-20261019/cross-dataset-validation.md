# 外部数据验证：Eadro SocialNetwork

范围更新：共同特征方案已由用户调整为共同架构、允许各数据集字段不同。新执行证据发现Normal包Trace ID有交集、无注入包含error span；不得根据本文早期准入描述直接生成健康标签。以[实现记录](preparation-implementation.md)的最新边界为准。

更新时间：2026-10-20

## 目的

外部数据用于验证同一个 Service 异常检测模型能否迁移到另一套微服务系统。它不承担证明自有数据集更优秀的任务；自有数据和外部数据的绝对分数不能直接作为数据集优劣结论。

当前候选模型为：日志文本编码器、同一固定时间窗内的模态 Transformer、Deep SVDD 正常表示目标。各数据集分别用自己的 Normal 运行拟合和独立 Normal 校准；冻结权重跨系统推理另列为更强的跨域实验。

## 数据包核验

- 来源：Eadro Zenodo record `10.5281/zenodo.7615394`。
- 文件：`SN Dataset.zip`，`78,753,809` bytes。
- MD5：`4c523faf64770b1e9a709cdd610e07c4`，与发布元数据一致。
- 归档包含 `SN-all.tgz`（4 个故障运行）以及 3 个 `no fault` tar.xz。
- 每个完整运行目录实际包含 `spans.json`、`logs.json` 和 12 个服务 Metrics CSV。早期候选报告写成 14 个 CSV，是把目录内文件总数误读为服务数量，现以本核验为准。

## 输入可迁移性

Metrics 字段为 `timestamp`、CPU system/total/user、memory usage/working set、`rx_bytes`、`tx_bytes`，时间戳为 epoch 秒。它不提供当前数据同名的 `process_count` 或可直接确认的 memory limit，因此跨数据集应采用重新定义的共同资源契约，例如 CPU、内存使用、网络收发变化，并在每个数据集内部拟合归一化。

Logs 按 Service 组织为原始日志行，含文本和日志时间。当前核验未证明每条日志都带 TraceID，故日志应按 Service×Window 聚合，不假设逐请求关联。

Traces 为 Jaeger 风格 spans，包含 trace/span 标识、操作名、起止时间、耗时和引用关系，可聚合请求/Span数量、耗时及在字段确实存在时的错误统计。缺少 span 不自动等于零请求。

## Normal 与故障证据

3 个 `no fault` 包分别是 2022-04-20 与 2022-04-21 的不同时间段归档，文件内容哈希不同，未发现它们是同一个压缩内容的重复封装。包内各自包含约 23 分钟的 span、日志和 12 个服务指标观测。它们可作为 fit、calibration 和 Normal holdout 的候选；“时间不重叠”不等于统计独立，仍需记录工作负载连续性和划分依据。
故障包中 4 个运行分别覆盖 2022-04-17 约 02:12–02:36、02:37–03:00、03:02–03:25、03:26–03:50 UTC，指标每个服务约 1,411–1,413 行，约 1 秒采样。相邻运行存在分钟级间隔，但不能把相邻运行自动视为独立故障样本。

4 个故障运行带有单独 JSON metadata，字段包括 `start`、`faults`、`end`，故障记录包含目标容器、故障类型、开始时间和持续时间。它们支持构造故障暴露索引，但不自动提供完整 Service×Window 真值：目标容器不等于所有受影响服务，传播和恢复窗不能自动标正常，Unknown 需排除。

## 当前准入结论

- 数据形态：通过。具备 Service 级 Metrics、Logs、Traces 和故障时间元数据。
- 共同特征：有条件通过。必须使用跨数据集共同语义，不能直接复用当前 R3 名称。
- Normal-only 划分：候选通过，正式实验前记录运行段独立性和工作负载差异。
- 主标签：有条件通过。先做证据范围内的受控故障评价，不能把所有窗口硬标为真实 Service 状态。

详细机器审计及外部报告边界见 `/tmp/service-cross-dataset-validation/detailed/`。其中 `AUDIT.json` 已记录外层成员哈希、指标字段、行数和时间范围；审计脚本当前对嵌套 no-fault 包的成员级交集仍需补一次专门解析，因此 Normal 的“未发现重复”目前是归档哈希和时间范围层面的结论，不升级为统计独立证明。原始数据包不进入仓库。
