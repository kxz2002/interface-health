# 050 · Eadro SN 先验 GRU 跨数据集最小闭环

- **日期**: 2026-10-29
- **类型**: Experiment / Model / Docs / Test
- **影响域**: `src/service/modeling/sn_adapter.py`、`scripts/run_sn_prior_gru.py`、`scripts/audit_sn_prior_gru.py`、`configs/service/sn_temporal_prior_gpu_v1.json`

## 做了什么

下载并校验 Zenodo Eadro SocialNetwork 数据包，生成新的 SN exploratory preparation 和独立 acceptance 产物。新增 SN model-facing adapter，把 SN 资源字段映射到现有先验 GRU 的固定张量接口，保留原始 labels 全部 Unknown；训练/测试清单依据固定 Normal/Fault run role，并把 Fault 正类明确标记为 controlled-exposure proxy。

修复 SN 资源缺失字段和日志模板统计语义后，复用本项目现有 `temporal_inputs`、`train_temporal`、`score_temporal` 和 Normal-fit availability prior，完成 seed 1/2/42 的最小闭环。正式 v2 三 seed proxy AUROC/AP 均值为 0.8146/0.9460；独立审计重新从 scores 计算并确认三次运行的 fit/calibration/test ID 清单一致。旧 v1 结果作废。

补充冻结 TT 共享先验 GRU 到 SN 的诊断迁移，proxy AUROC/AP 为 0.6523/0.8712；该结果独立保存，不与 SN 内训结果混排。

## 边界

- 这不是 Eadro 原论文模型复现，目标是验证本项目自己的先验 GRU 是否能在另一数据集输入上运行。
- SN 源 labels 仍全部 Unknown，proxy 指标不是独立 Service 金标性能。
- SN 资源单位和固定减 28,800 秒的时间对齐仍是 exploratory 假设；结果不能解释为已证明的真实跨系统泛化。
