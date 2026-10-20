# 036 · TT与SN训练前数据准备及探索协议

- 日期：2026-10-20
- PR：#34；分支：feature/service-training-preparation
- 类型：Feature / Data / Docs / Test
- 影响域：src/service/、scripts/prepare_service_data.py、configs/service/、tests/test_service_*、docs/research/service-20261019/

## 做了什么

实现来源核验、三模态窗口适配、独立labels/splits及质量验收。生产准备和只读复验已执行，相关Service定向测试107项通过；未运行模型、编码器、归一化拟合、阈值校准或旧DVC管线。原始数据和大产物不提交。

## 关键决策

- 延续035的selected17、整Case Normal-only协议；消化035 TODO第1项中的adapter及标签/划分基础设施部分，模型与基线训练仍未实施。
- 各数据集保留自身字段，共享四元主键及完整15秒网格。TT资源保值；执行按结束时间归窗，日志保留消息、频次及原文引用。缺模态保留，不补健康零。
- 用户明确批准SN探索分支：日志与Trace固定减28800秒仅为公开的内部坐标对齐假设；资源保留单位/测量语义未知的raw数值，不作rate/diff。strict默认仍保留阻断状态。
- SN无声明注入运行只提供常态参考分布资格；labels保持Unknown，训练资格与状态金标分离。Fault不参与fit/cal，窗口真值不足不能报告独立状态AUROC/AP。
- 原生产与后续验收分目录保存；manifest区分production与validation源码hash，拒绝覆盖旧证据。

## 坑 / 已知问题

- 旧Entry明细按start裁剪且缺完整去重键，不能作为结束时间归窗的完整输入；本轮从绑定总包流式读取。
- 验收器须从inventory和冻结profile独立推导角色/许可，不能只信任待验splits自证；六表schema必须存在。审查发现后已修复。
- PR #34后台审查发现二元标签校验仍可放行误填0/1；当前profile增加全Unknown门槛，0/1污染fixture先红后绿，25项相关检查及现有数据新目录只读复验通过。未来独立状态标签需另行显式扩展协议，不能静默填充。
- 数据与历史缓存不随Git提交。CLI需要固定hash的SN inventory、原始SN包及TT迁移清单/实现快照；历史绝对路径必须在本地可解析。当前不是clean clone即能重跑生产的自包含数据发布。
- 本地outputs、.superpowers和/tmp是执行证据位置，不是Git内交付；使用指南记录结论和入口，不把这些路径当远端审查者已可访问的附件。

## 遗留 TODO

1. 下一session确认模型、编码器、融合、SVDD目标、模态mask及训练配置，再实现Loader、前向/梯度检查和小试跑；本PR不启动训练。
2. 缺模态与字段质量需由模型显式mask消费；参考池许可数量不是最终有效训练数。
3. SN真实时区、资源物理语义及完整Service状态标签仍未证明；探索结果必须沿用限定范围。
4. 若迁移机器，补充历史路径重定位和可分发库存生成/绑定工作；不能自动猜路径、重采或改raw。
