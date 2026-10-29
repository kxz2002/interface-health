# SN 先验 GRU 最小跨数据集闭环（2026-10-29）

## 数据与协议

- 数据包：`data/external/eadro_sn/SN Dataset.zip`。
- 完整性：`78,753,809` bytes，MD5 `4c523faf64770b1e9a709cdd610e07c4`，ZIP 测试通过。
- preparation：`outputs/service/sn-preparation-20261029-task1/`；独立验收：`outputs/service/sn-preparation-20261029-task1-acceptance/`。
- SN 全网格 7,908 个 Service×15 秒窗；固定 Normal fit 1,140 窗，独立 Normal calibration 1,140 窗，测试/诊断 5,628 窗。
- 模型：复用本项目 `temporal_inputs`、`train_temporal`、`score_temporal` 和服务可用率先验；context=4、hidden=32、50 epochs、q99 calibration。
- SN 原始 labels 仍全部 Unknown。Fault-run 正类和 Normal-role 负类只作为 `controlled_exposure_proxy`，不是真实 Service 金标。

## 结果

| seed | proxy AUROC | proxy AP | test rows | proxy positives | proxy negatives |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8168 | 0.9468 | 5,628 | 4,488 | 1,140 |
| 2 | 0.8129 | 0.9454 | 5,628 | 4,488 | 1,140 |
| 42 | 0.8141 | 0.9459 | 5,628 | 4,488 | 1,140 |
| mean | 0.8146 | 0.9460 | — | — | — |

测试 proxy 正类比例为 `4,488 / 5,628 = 0.7974`，因此只输出 AP 不足以说明模型排序能力；常数分数 AP 基线为 `0.7974`。AUROC/AP 仍只适用于受控暴露 proxy，不是 Service 金标性能。

另外使用现有 TT 共享先验 GRU checkpoint 做了冻结权重诊断迁移：`outputs/service/sn-frozen-tt-prior-gru-20261029/`，SN 侧仍使用 fit-only 归一化和可用率先验，未使用 SN Fault 训练。proxy AUROC/AP 为 `0.6523/0.8712`，仅作为冻结迁移诊断，不能与 SN 内训三 seed 结果直接排名。

## q99 阈值下的分类指标

从独立 Normal calibration 的 q99 阈值重算分类指标：

| 实验 | F1 | Recall | Precision |
|---|---:|---:|---:|
| SN 内训先验 GRU，seed 均值 | 0.5109 | 0.3445 | 0.9887 |
| TT→SN 冻结迁移 | 0.2572 | 0.1493 | 0.9280 |

这些指标仍使用 controlled-exposure proxy 标签，未使用测试集调阈值。论文原文 SN 检测结果为 F1=0.986、Recall=0.996、Precision=0.977；论文采用监督训练和注入窗口标签，因此只能作协议参照，不能作为严格同口径排名。

## 8 窗与训练过程

8 窗 GRU 使用同一输入、split、seed 和 q99 协议，并通过 MLflow 记录逐 epoch loss。seed42 的 50→100 epoch 训练中，fit loss 从 `0.4223` 降到 `0.4059`，proxy AUROC 从 `0.8253` 升到 `0.8341`；100 epoch 三 seed 的均值为 AUROC `0.8382`、AP `0.9547`、F1 `0.5216`、Recall `0.3540`、Precision `0.9909`，相较 4 窗 50 epoch 的 AUROC `0.8146`、AP `0.9460` 有改善。

本机 CPU 实测（5628 个测试窗）如下：4 窗 50 epoch 单 seed fit 约 `7.3–8.4 s`，8 窗 100 epoch 约 `5.1–7.9 s`；单次测试评分约 `0.02–0.04 s`。模型参数量为 `6,447`。

论文报告了 GPU 型号、层宽和训练超参数，但没有报告可直接比较的训练/推理耗时；论文模型还包含模态专用编码器、DCC、4-head GAT 和联合检测/定位头，因此目前只能说我们的 GRU 更轻量，不能据此声称相对论文有已验证的速度倍数。

## 训练网格与候选阈值

固定 8 窗、200 epoch、三 seed 的训练网格通过 MLflow 完成：

| learning rate | q95 F1 | q97.5 F1 | q99 F1 | q99.5 F1 |
|---:|---:|---:|---:|---:|
| 1e-3 | 0.7870 | 0.7262 | 0.5371 | 0.4569 |
| 5e-4 | **0.8057** | **0.7536** | 0.5273 | 0.4541 |

在当前 proxy 协议和预先固定的候选阈值中，最佳平均 F1 为 `0.8057`，仍低于论文 SN 的 F1 `0.986`。这说明增加历史和训练预算能改善结果，但无法消除监督标签、任务粒度和评价协议之间的差异。

补充的 12 窗×15 秒、200 epoch、lr=5e-4 三 seed 结果：proxy AUROC 均值 `0.8394`、AP `0.9550`；q95/q97.5/q99/q99.5 的 F1 均值分别为 `0.8012/0.7403/0.5099/0.4554`。12 窗 AUROC 略高于 8 窗，但 q95 F1 略低于 8 窗的 `0.8057`，因此当前 12 窗可运行但没有稳定优势；暂不启动 5 秒窗口重建。

修复语义后的正式产物目录为 `outputs/service/sn-prior-gru-seed{1,2,42}-20261029-v3/`，每个目录包含 checkpoint、calibration/test 分数、q95/q97.5/q99/q99.5 阈值、训练历史和输入绑定 manifest。旧目录使用了已废弃的 `process_count ← rx_bytes_mean` 和伪模板统计映射，不纳入结论。

## 边界

这轮证明了本项目自己的先验 GRU 可以在 SN 的 exploratory preparation 上完成 Normal-only fit、独立 Normal calibration 和固定测试评分。它没有证明跨数据集泛化性能，也没有把 SN 的故障暴露映射成完整 Service 状态真值。SN 资源单位和固定减 28,800 秒的时间对齐仍是公开记录的 exploratory 假设。
