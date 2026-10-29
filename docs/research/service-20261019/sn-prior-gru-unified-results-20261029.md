# SN 先验 GRU 统一实验汇总（2026-10-29）

## 固定协议

- 数据：Eadro SocialNetwork，7,908 个 Service×15 秒窗口。
- fit：1,140 个 Normal 窗口；calibration：1,140 个独立 Normal 窗口；测试：5,628 个 proxy 窗口。
- 训练：Normal-only；阈值候选预先固定为 q95、q97.5、q99、q99.5；故障窗口不用于训练、epoch 或阈值选择。
- 指标：AUROC/AP 使用连续分数；F1/Recall/Precision 使用独立 Normal calibration 的对应分位数阈值。标签是 controlled-exposure proxy，不是独立 Service 金标。

## 多窗和训练网格

下表的 F1/Recall/Precision 使用 q95；q99 结果见原始 run 的 `report.json`。

| 配置 | AUROC | AP | F1 | Recall | Precision | seed |
|---|---:|---:|---:|---:|---:|---:|
| 4 窗 / 50 epoch / lr=1e-3 | 0.8146 | 0.9460 | 0.7504 | 0.6245 | 0.9400 | 1/2/42 |
| 8 窗 / 100 epoch / lr=1e-3 | 0.8382 | 0.9547 | 0.8051 | 0.6980 | 0.9509 | 1/2/42 |
| 8 窗 / 200 epoch / lr=1e-3 | 0.8373 | 0.9534 | 0.7870 | 0.6762 | 0.9413 | 1/2/42 |
| 8 窗 / 200 epoch / lr=5e-4 | **0.8385** | **0.9546** | **0.8057** | **0.7023** | 0.9448 | 1/2/42 |
| 12 窗 / 200 epoch / lr=5e-4 | 0.8394 | 0.9550 | 0.8012 | 0.6952 | 0.9454 | 1/2/42 |

q99 下，4 窗 50 epoch 的 F1/Recall/Precision 为 `0.5109/0.3445/0.9887`；8 窗 100 epoch 为 `0.5216/0.3540/0.9909`。q95 明显更适合 F1/Recall，q99 更适合低误报报警。

## 训练和推理耗时

- 8 窗 50 epoch：单 seed fit 约 2.8 秒（seed42，MLflow）。
- 8 窗 100 epoch：约 5.1–7.9 秒；8 窗 200 epoch 的并发网格运行受 CPU 争用影响，约 565–579 秒，不能当作无争用硬件基准。
- 5628 个测试窗口单次评分约 0.02–0.04 秒。
- GRU 参数量：6,447。

论文只报告了 GPU/层宽和分类指标，没有报告可直接比较的训练/推理耗时；因此这里报告本项目实测速度，不宣称相对论文的倍数加速。

## 结论

8 窗比 4 窗稳定提升；12 窗 AUROC/AP 略高，但 q95 F1 略低，没有稳定优势。当前最佳 q95 proxy F1 为 `0.8057`，仍低于 Eadro 论文 SN 的监督式 F1 `0.986`，两者任务、标签和训练协议不同，不能直接排名。

## 训练方案与架构对照

| 项目 | 本项目 SN 先验 GRU | Eadro 原论文 |
|---|---|---|
| 训练范式 | Normal-only fit；独立 Normal calibration；Fault 仅测试 | 监督训练；60% train / 40% test；注入窗口直接作为异常标签 |
| 主要模型 | 单个共享 GRU，输入 12 个数值/统计字段、mask 和可用性，hidden=32，输出重建与可用性 logits | 日志/KPI/Trace 模态专用编码器，DCC 时序模块，4-head GAT 依赖建模，联合 detector/localizer |
| 优化 | AdamW，batch=128，lr=1e-3 或 5e-4，50/100/200 epoch，gradient clip=1 | Adam，batch=256，lr=0.001，50 epoch |
| 评价阈值 | Normal calibration q95/q97.5/q99/q99.5，主协议不看 Fault 选阈值 | 二分类监督输出，论文报告 F1/Recall/Precision |
| 任务粒度 | Service×15 秒窗口异常分数 | 系统级异常检测并联合根因服务定位 |

## 未达到论文结果的可能原因

1. **标签协议不同**：论文把注入操作覆盖的窗口直接作为异常/正常真值；本项目保留 SN 原始 labels 为 Unknown，只使用 fault-run exposure proxy，正类范围更宽且不等价于真实 Service 受影响状态。
2. **监督信息不同**：论文训练期间使用异常样本和根因服务标签；本项目 fit 只看 Normal，模型只能学习正常分布，不能直接优化异常分类边界。
3. **模型容量和结构不同**：Eadro 使用模态专用网络、DCC、GAT 和联合检测/定位任务；本项目刻意保持轻量共享 GRU，没有显式建模服务依赖传播，也没有根因定位辅助损失。
4. **输入语义不对称**：SN 没有 `process_count`，该通道被 mask；日志 adapter 只使用真实 event count，未引入论文中的 Drain 事件语义；资源单位和测量语义仍是 exploratory unknown。
5. **时间和评价粒度不同**：本项目固定 15 秒 Service 窗口并用 q 分位数阈值；论文使用其监督检测窗口和分类阈值。即便使用相同原始数据，样本集合和决策边界也不相同。
6. **阈值目标不同**：q99 优先控制 Normal FPR，会牺牲 Recall；q95 已明显提高 F1，但仍受 proxy 标签和输入缺失限制。

因此，当前结果更适合支持“轻量先验 GRU 在另一套多模态微服务数据上可运行，并能进行冻结迁移诊断”的结论；不能把它写成对 Eadro 监督式结果的严格复现或性能超越。
