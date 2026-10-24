# 041 · 当前窗口冻结 BERT 语义评分对照

- **日期**: 2026-10-23
- **类型**: Experiment / Model
- **影响域**: scripts/service_current_semantic_model.py、configs/service/、tests/test_service_current_semantic.py、docs/research/service-20261019/

## 做了什么

新增独立当前窗口冻结 BERT 语义距离分支，并与原服务先验 GRU 分数做 Normal calibration robust-z 融合；semantic-only 与 fusion 各完成三 seed，全部产物递归核验通过。

## 关键决策

- 当前窗口 BERT 不参与重建，而是相对每服务 Normal fit 语义中心计算 cosine distance。
- 语义分支与融合分支使用独立配置、输出和缓存父链，不覆盖原 GRU/BERT 运行。
- 两种新路线均显著退步，停止继续搜索；后续回到原 GRU 的训练过程和结构改进。

## 遗留 TODO

- 审阅 MLflow 训练曲线与分量，提出并验证原 GRU 的有限改进候选。
