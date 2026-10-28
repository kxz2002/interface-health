# 微服务领域神经基线首轮结果

运行入口：`scripts/service_domain_baselines.py`。代码提交 `fa6832e` 修复了 Trace bucket 对齐、Normal-fit 图拓扑泄漏和 Trace 特征 fit-only 标准化；远端训练在 gpu-b 专用 worktree `v3-baselines-8413b1a` 完成。输入为 prepared Service 数据、v3 已验收标签包和原始 SkyWalking Trace；训练只消费 Normal fit，q99 只消费独立 Normal calibration。

机器可读的逐 seed 汇总见 [`microservice-domain-baselines-results-20261028.json`](microservice-domain-baselines-results-20261028.json)。

主评价严格使用 v3 `main_test`，共 3141 个 Service×15 秒窗口（289 正、2852 负），POOL/DNS 排除。模型历史长度为 4 个窗口；无法形成完整历史的 72 个主评价窗口保留在评价中，分数填为 0 并单独计数。

当前每个 Normal Case 全为负类、每个故障 Case 全为正类，因此 per-case AUROC/AP 均无定义（`n_cases_with_both_classes=0`）。本轮不填充伪造的逐 Case 指标；标签体系修改列为后续独立分支，当前结果只报告 pooled 指标。

| 模型适配 | seed | AUROC | AP | q99 recall | Normal FPR |
|---|---:|---:|---:|---:|---:|
| MSTGAD 风格图时空重建 | 1 | 0.8665 | 0.6729 | 0.5813 | 0.0929 |
| MSTGAD 风格图时空重建 | 2 | 0.8566 | 0.6716 | 0.5779 | 0.0926 |
| MSTGAD 风格图时空重建 | 42 | 0.8750 | 0.6848 | 0.5779 | 0.0929 |
| AnoFusion 风格多模态 GRU | 1 | 0.6975 | 0.5765 | 0.5398 | 0.0810 |
| AnoFusion 风格多模态 GRU | 2 | 0.7402 | 0.5877 | 0.5433 | 0.0929 |
| AnoFusion 风格多模态 GRU | 42 | 0.7104 | 0.5755 | 0.5398 | 0.0947 |
| Triple 风格图 SVDD | 1 | 0.9037 | 0.5849 | 0.8097 | 0.1655 |
| Triple 风格图 SVDD | 2 | 0.8783 | 0.5280 | 0.6990 | 0.1522 |
| Triple 风格图 SVDD | 42 | 0.8900 | 0.5885 | 0.7509 | 0.1417 |

修复后三个 seed 均值为：MSTGAD 风格 AUROC/AP `0.8660/0.6764`，AnoFusion 风格 `0.7160/0.5799`，Triple 风格 `0.8906/0.5671`。Triple 风格的 FPR 均值约 `0.1531`，仍然偏高，不能只看召回宣称优越。

## 适配边界

- 独立 review 发现并修复了三个问题：Trace bucket 的绝对/相对索引错位、从故障 Case 取全局图拓扑、Trace 特征未使用 Normal fit 统计量。修复后已在 gpu-b 重新运行全部三个 seed；修复前结果不再使用。
- 这些是论文结构的可追溯适配，不是官方代码的逐行复现；没有复制无许可证的 MSTGAD 代码。
- 服务依赖图由原始 Trace 的显式父子 span 汇总，节点输入由 prepared 的 Metric/Execution/Log 统计和 Trace 聚合特征组成。
- 当前结果尚未完成逐样本重放、逐 Case 审计和独立外部复核；不能把三种方法写成已通过正式强基线验收。
- 当前测试集已被多轮探索，结果不是盲测；正类是受控关联证据，不是完整 Service 异常金标。
