# Service研究交接 — 2026-10-19

- [两条 GRU 路线运行说明](temporal-model-running.md)：原先验 GRU 与可选冻结 BERT 共存、配置选择及产物保留。
- [Normal-only 调参结果](temporal-tuning-protocol-20261023.md)：12 候选及选定配置三 seed 复核，未超过无 BERT 模型。
- [服务先验 GRU 查新](service-prior-gru-novelty-20261023.md)：谨慎推进，说明最近邻、可辩护差异与必要消融。
- [POOL / DNS 观测审计](pool-dns-observability-audit-20261023.md)：未见明确池等待或解析失败证据，优先核查干预路径与实际业务影响。

- [2026-10-23 模型诊断与真实实验](model-collapse-investigation.md)：原SVDD坍缩、重建与可用性对照、Normal服务先验GRU及固定三seed复核；主评价3297窗，先验GRU三seed平均AUROC/AP为0.823970/0.680061，加入冻结BERT历史语义后为0.813235/0.667020，未改善。已见测试集后的探索，保留逐故障盲区；原Task10九方法矩阵尚未全部完成。
- [TT 主评价范围决策](evaluation-scope-decision-20261021.md)：独立 Normal test 负类、Fault 已知正类，不纳入 Fault 注入前负类；保留实施前差距快照及补采版本接入要求。
- [2026-10-20 TT 标签 v2 实施交接](labeling-v2-implementation-20261020.md)：已生成并验收479正类与2975负类的主评价清单；资源有效掩码后397正类、2453负类，附模型接入示例和证据边界。

主任务为Service×固定时间窗正常/异常二分类；维持Normal-only训练，以独立校准阈值输出二分类。允许成熟方法上的适度改进，不以顶会级原创作为启动门槛。Endpoint模型独立开展，Operation/漂移/缺失等不代替主任务。

阅读顺序：

1. [当前研究目标](current-research-goal.md)：用户已对齐的任务、创新尺度及证据边界。
2. [数据与实验协议](dataset-and-protocol.md)：selected17、特征语义、标签及已执行划分。
3. [已完成基线与限制](baseline-findings.md)：保留强规则和失败结果，不等同完整Service准确率。
4. [研究档案与引用](archive-and-references.md)：完整复现原件、独立裁决、文献资料的位置与hash。
5. [下一轮启动说明](next-agent.md)：只在下一轮授权后实施adapter、标签评价表及模型。

数据根为 `data/tt_service_20261017_004546/`；唯一正式入口是其中 `batch_control/final_acceptance/selected_cases.json` 的17项allowlist。`_handoff/migration_summary.json` 是复制完成门禁，`_handoff/path_map.json` 解析原路径。未知窗口不补健康0；禁止目录glob混入失败attempt或旧new_merge。

旧 `service-method-admission.md`、`service-minimal-v1*.md` 和原/tmp中的Go/No-Go记录按各自旧批次/历史目标解读，**不作为当前轻量原型开发禁令**。保留其原件和负向结果，不以本页重新解释为当时已通过。

## 后续文献资料

- [网页文献调研提炼（2026-10-19）](literature-web-20261019/README.md)：文献任务、数据集、模型机制和开源定位台账；附完整原报告与来源hash。外部报告转述，尚未独立核验或运行；不执行原报告的Agent任务，供后续brainstorming参考。

- [网页Agent复核增量（2026-10-19）](literature-web-20261019/followup-findings.md)：固定版本的实现/数据/评价补证、首轮修正与实际执行范围；附完整合集，不启动其中任务。

- [外部数据验证：Eadro SocialNetwork](cross-dataset-validation.md)：已核验公开数据包及三模态输入/Normal与故障证据边界；用于后续同一模型跨数据集验证，不承担数据集优越性证明。

- [Service模型研发阶段性整理](research-synthesis-20261020.md)：汇总当前模型假设、跨数据集逻辑、SocialNetwork核验、共同特征契约和后续准入工作。

- [训练前准备 Spec](training-preparation-spec.md) 与 [训练前准备 Plan](training-preparation-plan.md)：冻结跨数据集输入、标签、划分、质量门槛和训练前交付物；尚未执行训练。

- [数据准备实现与已执行检查](preparation-implementation.md)：当前TT/SN五表、质量mask/暴露索引、58,483完整窗、93项定向检查与生产来源回查；TT输入准备及SN公开假设下探索输入准备已验收，模型mask与训练尚未执行，全部状态标签Unknown。已修复独立角色/schema验收gate，20项相关复查及新目录只读复验通过；保留首版49项骨架检查历史。

- [TT注入关联标签 Spec](labeling-infrastructure-spec.md)、[Plan](labeling-infrastructure-plan.md) 与 [实现及生产验收](labeling-infrastructure-implementation.md)：版本化标签/划分、真实来源复验及覆盖限制。
