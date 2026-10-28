# 微服务领域神经基线首轮结果

运行入口：`scripts/service_domain_baselines.py`。代码提交 `a238d17`，远端训练在 gpu-b 专用 worktree `v3-baselines-8413b1a` 完成。输入为 prepared Service 数据、v3 已验收标签包和原始 SkyWalking Trace；训练只消费 Normal fit，q99 只消费独立 Normal calibration。

主评价严格使用 v3 `main_test`，共 3141 个 Service×15 秒窗口（289 正、2852 负），POOL/DNS 排除。模型历史长度为 4 个窗口；无法形成完整历史的 72 个主评价窗口保留在评价中，分数填为 0 并单独计数。

| 模型适配 | seed | AUROC | AP | q99 recall | Normal FPR |
|---|---:|---:|---:|---:|---:|
| MSTGAD 风格图时空重建 | 1 | 0.7878 | 0.5670 | 0.5329 | 0.1203 |
| MSTGAD 风格图时空重建 | 2 | 0.7949 | 0.5729 | 0.5190 | 0.1199 |
| MSTGAD 风格图时空重建 | 42 | 0.8042 | 0.5774 | 0.4844 | 0.0870 |
| AnoFusion 风格多模态 GRU | 1 | 0.7005 | 0.5279 | 0.4740 | 0.0782 |
| AnoFusion 风格多模态 GRU | 2 | 0.7125 | 0.5263 | 0.4844 | 0.0936 |
| AnoFusion 风格多模态 GRU | 42 | 0.7376 | 0.5275 | 0.4740 | 0.0894 |
| Triple 风格图 SVDD | 1 | 0.8332 | 0.4606 | 0.6990 | 0.1683 |
| Triple 风格图 SVDD | 2 | 0.8010 | 0.3977 | 0.7059 | 0.1999 |
| Triple 风格图 SVDD | 42 | 0.8808 | 0.4906 | 0.7543 | 0.1504 |

三个 seed 均值为：MSTGAD 风格 AUROC/AP `0.7956/0.5724`，AnoFusion 风格 `0.7169/0.5272`，Triple 风格 `0.8383/0.4496`。Triple 风格的 FPR 明显偏高，不能只看召回宣称优越。

## 适配边界

- 这些是论文结构的可追溯适配，不是官方代码的逐行复现；没有复制无许可证的 MSTGAD 代码。
- 服务依赖图由原始 Trace 的显式父子 span 汇总，节点输入由 prepared 的 Metric/Execution/Log 统计和 Trace 聚合特征组成。
- 当前结果尚未完成逐样本重放、逐 Case 审计和独立外部复核；不能把三种方法写成已通过正式强基线验收。
- 当前测试集已被多轮探索，结果不是盲测；正类是受控关联证据，不是完整 Service 异常金标。
