# 中期汇报逐页提纲（2026-10-14早期候选）

1. 研究对象：Service内部资源/执行/可用性与Endpoint客户端RED/Response分别输出；真实依赖关联是后续工作。数据集和旧小论文是基础，不计两个方法创新。
2. 证据与任务：S1状态、S2注入目标检索、S3完整RCA；本轮独立S1标签缺失，指标null。人工同源审核不构成独立金标。
3. Endpoint公平实验：引用[原9/12维报告](/tmp/endpoint-red-response-minimal-20261013/report.md)与其protocol.json。共同样本、参考期前视排除及非目标Unknown按原协议说明；Response增量集中PATCH，非PATCH退步，不预设多模态更强。不拼expanded18/21维数字。
4. Service观测工程：独立完整15s网格、25候选、身份完整counter实际dt、positive limit、稳定单实例。展示[覆盖与质量表](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/service_coverage.csv)。
5. Normal-only强基线：固定60/20/20、共享R3有效ID、CPU-only/R3×四方法；[尾段分位](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/normal_tail_summary.csv)仅描述性统计。
6. CPU事件：展示[目标时间线](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/event_1.png)及[24行排名区间](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/event_comparison.csv)。CPU弱暴露关联强，跨Service校准饱和造成并列；两任务不横比。
7. KILL：展示[order时间线](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/event_2.png)与[preserve时间线](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/event_3.png)，目标Unknown不代表健康。KILL操作传播仅引用此前/tmp/api-impact-review-20261013审计；本轮未重读raw、未验证可用性。
8. 双层差异：CPU资源变化与PATCH/DELAY接口表现是不同观测对象。当前8入口属8不同Service，不能证明同Service多入口差异；不宣称双层融合完成。
9. 方法准入：展示[五项门槛及止损](service-method-admission.md)与[失败模式](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/failure_modes.csv)。A停止扩展，B条件性，C尚未验证。
10. 后续：先补独立/限定证据、活实例观测与新确认事件，再获授权设计方法。10月26日冻结为未来计划，预算尚未确认；无法稳定补证据则放中期后。

全部图表和指标的正式入口为[只读报告](../../artifacts/service_minimal_v1_20261014T100628Z/report_v3/report.md)，其manifest记录保存输入SHA、配置、代码、环境和输出。没有已有PPT，本文件为可直接制作幻灯片的提纲。
