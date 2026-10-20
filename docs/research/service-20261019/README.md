# Service研究交接 — 2026-10-19

主任务为Service×固定时间窗正常/异常二分类；维持Normal-only训练，以独立校准阈值输出二分类。允许成熟方法上的适度改进，不以顶会级原创作为启动门槛。Endpoint模型独立开展，Operation/漂移/缺失等不代替主任务。

阅读顺序：

1. [当前研究目标](current-research-goal.md)：用户已对齐的任务、创新尺度及证据边界。
2. [数据与实验协议](dataset-and-protocol.md)：selected17、特征语义、标签及已执行划分。
3. [已完成基线与限制](baseline-findings.md)：保留强规则和失败结果，不等同完整Service准确率。
4. [研究档案与引用](archive-and-references.md)：完整复现原件、独立裁决、文献资料的位置与hash。
5. [下一轮启动说明](next-agent.md)：只在下一轮授权后实施adapter、标签评价表及模型。

数据根为 `data/tt_service_20261017_004546/`；唯一正式入口是其中 `batch_control/final_acceptance/selected_cases.json` 的17项allowlist。`_handoff/migration_summary.json` 是复制完成门禁，`_handoff/path_map.json` 解析原路径。未知窗口不补健康0；禁止目录glob混入失败attempt或旧new_merge。

旧 `service-method-admission.md`、`service-minimal-v1*.md` 和原/tmp中的Go/No-Go记录按各自旧批次/历史目标解读，**不作为当前轻量原型开发禁令**。保留其原件和负向结果，不以本页重新解释为当时已通过。

## 训练前数据准备（2026-10-20）

- [Spec](training-preparation-spec.md) 与 [Plan](training-preparation-plan.md)：TT/SN输入、标签、固定划分及用户批准的SN探索协议。
- [实施与验收记录](preparation-implementation.md)：五表、质量mask和来源清单已生成；全部评价标签Unknown，尚未训练。
- [来源核验](preparation-source-verification.md)：版本、时间与物理语义证据边界。
- [本次收尾与下次session入口](preparation-session-handoff.md)：数据准备到此结束，模型与训练配置留下一session。
