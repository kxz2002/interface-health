# 012 · L1/L2 融合消融：独立编码器 + 门控条件融合实测

- **日期**: 2026-07-15
- **PR**: N/A（同分支多 commit，未开 PR）· **Commit**: 21ff215..351c5c3（本 entry 对应实测结果 commit）
- **类型**: Experiment
- **影响域**: `src/fusion/`, `configs/fusion/`, `tests/`

## 做了什么

在 011 补齐的基础设施（Hydra 可插拔 fusion + Contract v1 时序切分）之上，实现并跑通了 L0-L3 消融的前两级：

- **L1 `IndependentConcatFusion`**（`src/fusion/independent_concat.py`）：endpoint/service 两路各过独立单层 `Linear(bias=False)+ReLU` 编码器后直接拼接，无门控交互。
- **L2 `GatedFusion`**（`src/fusion/gated.py`）：同样两路编码器，额外加 `gate`/`value` 两个 `Linear` 层，`g=sigmoid(gate([e_ep;e_svc]))`，`z=e_ep+g⊙value(e_svc)`。
- 对应 Hydra config `configs/fusion/independent_concat.yaml`、`configs/fusion/gated.yaml`，新增 12 个单元测试（编码器独立性、门控公式对称性、`output_dim`、`modality_dims` 校验等）。

在 Contract v1（时序切分，13632 eval 样本）上用 `training.epochs=50` 完整跑通 L1、L2 训练+评估，全量测试套件 193 passed / 2 skipped，与 L0（entry 011 记录的 AUROC=0.617/AUPRC=0.327）三组对比：

| 变体 | 机制 | AUROC | AUPRC |
|------|------|-------|-------|
| L0 | 裸拼接（`EarlyConcatFusion`，零参数） | 0.617 | 0.327 |
| L1 | 独立 encoder（无门控，本轮实测） | 0.6082 | 0.3226 |
| L2 | 门控条件融合（本轮实测） | 0.6316 | 0.3369 |

## 关键决策（不在 commit 里）

- **本轮定位是"收集数据"，不是"证明创新点"**：entry 010 的 novelty-check 结论已经确定——门控机制本身（`sigmoid` 门 + 仿射调制）结构上与 FiLM（Perez et al. 2018）/ GS-Fuse 属于同一机制家族，是十年历史的成熟机制类别，不能作为论文的核心创新点宣称。本轮 L1/L2 的角色是**强基线 / 组件候选**：如果 L2 相对 L0/L1 有明显收益，这是"content-adaptive modulation of broadcast-duplicated service-level features"这个方向有实际价值的证据，用来指导下一轮"真正的新融合模型"往哪个方向设计增量；如果收益很小，这也是一个有用的负向结果（说明简单门控吃不到多少油水，得往别处找增量）。不应把这篇 entry 或后续论文写成"我们的贡献是门控融合"。
- **L1 vs L2 不做参数量对齐**：010 记录的 L0→L1 confounding variable 问题（裸拼接→独立 encoder 本身引入了新的非线性容量，性能提升不能干净归因于"解决了粒度错配"）在 L1→L2 这一步不成立，原因不同——L2 的 `gate`/`value` 两层**是门控架构本身的组成部分**，不是与门控设计无关的额外容量。去掉 `gate`/`value` 就不再是门控机制，无法在"同参数量但去掉门控"的条件下做对照；参数量对齐 baseline 只在"想证明某个机制的贡献独立于其带来的容量增长"时才有意义，而 L1→L2 的实验问题本来就是"门控这整个机制（含其必然带来的参数量）值不值"，因此参数量差异是设计变量的一部分，不是混淆变量，不需要（也不应该）对齐。

## 坑 / 已知问题

- **本轮未遇到新坑**。两次训练均在几秒内收敛完 50 epoch（838 训练样本，规模符合预期），loss 曲线单调下降，两次评估均正常写出 `metrics.json`，`by_anomaly_type`/`by_anomaly_level`/`by_endpoint` 分层字段结构与 L0/v1 一致。
- **观察到但判断为正常现象、非 bug 的一点**：L2 最终 epoch loss（≈0.00047）比 L1（≈0.0113）低了一个数量级。这是 Deep SVDD one-class 目标下常见的现象——L2 多出的 `gate`/`value` 两层给了模型更大的容量把训练集（838 个 Normal 样本）里的正常表征收缩到超球心附近，训练 loss 更低是预期中的容量效应，不代表数值异常或训练不稳定；结合 eval AUROC/AUPRC 均正常（非 0.5、非 1.0、非 NaN）且分布合理，判断不是超球坍缩（hypersphere collapse）导致的退化。这一点留给后续设计参数量对齐实验时留意：容量越大，训练 loss 会越低，不能拿训练 loss 高低直接当作机制优劣的证据，必须看 eval 指标。

## 遗留 TODO

- **L3（全局标量门控）是否值得做**：本轮 L2 相对 L1 有正向但不算悬殊的提升（AUROC +0.0234，AUPRC +0.0143），相对 L0 则是 L1 略降、L2 略升（L0 AUROC=0.617 高于 L1 的 0.6082，但低于 L2 的 0.6316）。三组数字说明"引入非线性容量"（L0→L1）本身没有带来提升，而"引入门控"（L1→L2）才有；这支持"门控机制本身有一定增量价值，值得继续往这个方向摸"的判断，但增量幅度不大，不足以支撑"门控就是答案"的强结论。是否继续做 L3 取决于下一轮讨论：如果目标是把消融矩阵做完整（L0-L3 全跑一遍是原计划），可以顺手补上；如果目标已经转向"设计真正的新融合模型"，L3 可能不再必要，直接拿 L1/L2 的数据去支撑新设计的动机即可。
- **"真正的新融合模型"该往哪个方向设计增量**：本轮数据显示门控收益存在但有限，暗示单纯的 sigmoid 门控调制已经接近这类简单机制在当前特征/数据规模下的天花板。下一轮设计新融合模型时，应该重点考虑 endpoint×time-window 粒度的时序结构或跨 case 的分布差异等门控机制没有利用到的信息，而不是在门控公式本身做更复杂的变体（如多头门控、多层门控），后者大概率边际收益更小且进一步逼近 FiLM 系机制家族，novelty 风险更高。
