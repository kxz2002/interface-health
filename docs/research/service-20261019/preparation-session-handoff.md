# 数据准备收尾与下次session入口

本次任务到数据准备结束；未实现模型/Loader、文本编码、归一化拟合或训练。后续先确认模型与训练配置，不因本页自动启动实验。

## 已完成

- TT selected17与SN七运行分别输出resource_windows、execution_windows、log_windows、labels、splits及schema，保留全网格、质量mask、暴露索引和来源引用。
- 本轮生产为TT 50,575窗、SN 7,908窗；全部评价标签Unknown。参考池与质量mask分开，最终模型有效样本数未确定。
- 最后相关检查为107项Service定向测试通过。未执行全仓测试；数据、Parquet、执行日志及本地审查工作区不随Git提交。
- Spec/Plan记录用户批准的SN探索定义：固定-28800秒公开假设、unknown-unit raw资源、无声明注入常态参考分布；不宣称时区或健康金标已证。

## 本地验收入口

生产目录：`outputs/service/preparation-exploratory-20261020-task4/`。

最新只读复验：`outputs/service/preparation-exploratory-20261020-task4-acceptance-final/`。

PR #34审查追加：当前准备profile的验收器明确拒绝任何非空0/1标签，避免误把参考资格写成状态金标。污染fixture先复现后修复，25项相关测试通过；最新只读复验为 `outputs/service/preparation-exploratory-20261020-pr-review-acceptance/`，TT/SN仍全Unknown，原产物未覆盖。新validator SHA256为 `f03c8f76b3de1bdee7ce34bb0b972cfcaefe3d9fbc4ab39419df0503c1d19e56`；此前final目录保留为历史验收。

这些是本机被忽略的数据与证据，不是PR附件。完整字段/命令/限制见[实施记录](preparation-implementation.md)、[Spec](training-preparation-spec.md)、[Plan](training-preparation-plan.md)、[来源核验](preparation-source-verification.md)和history entry 036。

复跑需要本地TT快照及历史path_map/实现快照、固定MD5的SN包和经SHA256绑定的历史SN inventory；当前未交付clean-clone自包含生产数据。机器迁移时先核对路径可解析性，不猜偏移或自动重采。

## 下次session

1. 读取AGENTS、当前研究索引、本文及Spec/Plan；检查Git与本地验收manifest。
2. 与用户确认编码器、融合方式、SVDD目标、缺模态mask及训练配置。
3. 按授权实现Loader/模型，检查前向和梯度、小规模试跑；再安排TT/SN各自Normal-only训练和独立校准。
4. 评价标签仍Unknown；不能把参考池或注入暴露代理转换成完整状态真值。
