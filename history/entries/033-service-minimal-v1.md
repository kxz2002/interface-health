# 033 · Service独立资源基线与证据分层

- 日期：2026-10-14
- 分支：feature/service-minimal-v1；实现提交见 git log
- 类型：Feature / Docs
- 影响域：src/service/、configs/service/、scripts/service_experiment.py、tests/test_service_*、docs/research/

## 做了什么

建立独立Service资源观测、Normal-only经典基线、受控目标检索与保存产物报告。研究结果与准入决定保存在独立artifacts目录，历史Endpoint结果保持原协议。

## 关键决策

采用独立包，因为旧MetricPreprocessor丢失系列身份、固定dt、缺失补零等语义不能直接迁入Service。共享接口保持旧默认行为，Service不伪造Endpoint字段。

入口采用独立Hydra配置与command分支，符合项目现有调用习惯；不用Endpoint base配置驱动新粒度。数据、模型、评价、报告分别预留新目录，已存在拒绝覆盖。报告验证保存产物SHA后才加载，避免展示过程悄然重算模型或调整窗口。

S1只接受独立Service窗口证据；注入对象排序作为S2。生命周期事实不扩成资源或全Service可用性真值。KILL资源不可评分仍保留事件分母，Unknown竞争候选与并列保守评价。

方法准入五项须全部满足。资源输入的质量修复、现成模型组合和排序失败诊断不能自动成为新方法。原始数据与AnoMod保持只读，禁止用DVC重建替代局部开发；承接032的目录依赖漂移教训。

## 坑 / 已知问题

经验校准样本少时顶端饱和会造成跨Service并列；不能按测试结果翻方向或改用raw分数破并列。旧新系列同时存在时不代表多个活实例，保守身份规则会令目标资源Unknown。

首次CPU效果证据为counter差分区间起点，不能称精确生效瞬间。Normal尾段分位只是描述性参考，不构成独立误报率。手工读取模型同源输入不升级独立金标。

T6初次实现先于RED；已撤回新增报告代码，保留测试观察NotImplementedError，再重新实现并验证。该纠正不抹去初始流程偏差。

报告审查后补充：内容哈希正确不能替代表结构正确，保存score需验证8臂、稳定样本身份与评价候选关联再作图；已消费的admission目录同样须保护。图文件必须随正式报告交付。正确性修复写新report_v2，原报告保存冻结字节并单列版本说明。

## 遗留 TODO

1. 中期后按新授权补独立状态/活实例证据与新确认事件，另立Spec/Plan再讨论执行状态或可靠性机制。
2. 10月26日是未来冻结计划；本次保存的是10月14日早期候选，修复应另版本而不覆旧结果。
3. 临时Endpoint公平报告可能清理，正式论文引用前需保存其原协议与完整产物，不能拼接expanded历史数字。
