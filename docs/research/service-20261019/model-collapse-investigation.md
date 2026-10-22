# SVDD 坍缩排查与重建实验

2026-10-22，基于已合并的 `0fc5055`，实验分支 `exp/service-svdd-collapse`。

## 已验证事实

gpu-b 原仓库仍在 `21fdc3e`，存在未提交修改和历史同步产生的根目录副本。
完整已提交源码已部署到 `~/Code/Repos/interface-health-runs/0fc5055`，不覆盖旧目录。
training.py、model.py、service_model.py 两端 SHA256 一致；GPU 检查时显存使用 1 MiB。

首轮训练完成 20 epoch、12700 optimizer steps。原始诊断来自
`model-task10-transformer-cls-seed42-20261022-a/run/model/training_result.json`。

|epoch|训练均值损失|fit诊断表示总方差|退化标记|
|---|---:|---:|---|
|0|—|2.05295984|false|
|1|0.15283451|0.00040958|false|
|2|0.00756638|0.00014149|true|
|20|0.00002413|0.00000011479|true|

证据更支持表示坍缩，而不是训练轮数不够。目标函数持续下降并不代表异常检测能力提高。
可学习模态常量、summary token、LayerNorm 平移及 BERT 偏置存在输入无关输出路径；
具体模块贡献仍需消融，不能仅凭源码断言唯一原因。

## 本轮预先固定的实验

采用独立配置 `tt_model_reconstruction_gpu_b_v1.json`，保持相同 Normal-only fit、
calibration、主测试样本及数值预处理。三模态编码器和融合结构不变，增加从 32 维
表示重建 8 项数值特征与 4 项日志统计的 decoder。仅在可观测字段计算均方误差；
缺失文本时不将日志统计当有效监督。训练目标及异常分数均为重建误差，
不是原 SVDD 距离，也不是使用测试标签训练的分类器。旧配置保持原行为。

固定 seed42、20 epoch、原学习率；不利用测试结果选 checkpoint、反转分数或调整超参数。
这是一项探索性方法变更，不能保证胜过 IF/OCSVM。既有测试集结果已经可见，后续结果
必须披露探索性质，不能当作全新盲测。尚未获得本轮训练/评价结果。
