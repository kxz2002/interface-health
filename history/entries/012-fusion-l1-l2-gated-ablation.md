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

在 Contract v1（时序切分，13632 eval 样本）上用 `training.epochs=50` 完整跑通 L1、L2 训练+评估，全量测试套件 193 passed / 2 skipped，与 L0（`artifacts/baseline_v1/metrics.json`，entry 011 记录时取 3 位小数 AUROC=0.617/AUPRC=0.327，本表统一取 4 位小数重新读取）三组对比：

| 变体 | 机制 | AUROC | AUPRC |
|------|------|-------|-------|
| L0 | 裸拼接（`EarlyConcatFusion`，零参数） | 0.6169 | 0.3275 |
| L1 | 独立 encoder（无门控） | 0.6065 | 0.3238 |
| L2 | 门控条件融合 | 0.6390 | 0.3398 |

**注意：以上是 `scripts/train_baseline_v0.py` optimizer bug 修复后的数字，不是本 entry 最初提交时记录的数字**，详见下方"坑 / 已知问题"第一条。修复前的错误数字曾是 L1 AUROC=0.6082/AUPRC=0.3226、L2 AUROC=0.6316/AUPRC=0.3369——已作废，不应再被引用。

### 多 seed 复测（seed=1/2/3，叠加原有 seed=42）

optimizer bug 修复后，为确认 L1→L2 的增量是真实信号而不是单次训练的随机性噪声，额外用 `seed=1/2/3` 各跑一遍 L1/L2 训练+评估（复用同一份 `artifacts/contract_v1`，Contract v1 的时序切分本身不吃 seed，只有训练阶段的权重初始化/DataLoader shuffle 受 seed 影响）：

| 变体 | seed=1 | seed=2 | seed=3 | seed=42 | 均值 | 标准差 |
|------|--------|--------|--------|---------|------|--------|
| L1 AUROC | 0.6100 | 0.6072 | 0.6174 | 0.6065 | 0.6103 | 0.0050 |
| L2 AUROC | 0.6388 | 0.6218 | 0.6190 | 0.6390 | 0.6297 | 0.0107 |
| L1 AUPRC | 0.3271 | 0.3164 | 0.3293 | 0.3238 | 0.3241 | 0.0056 |
| L2 AUPRC | 0.3398 | 0.3312 | 0.3309 | 0.3398 | 0.3354 | 0.0051 |

L0 目前仍只有 seed=42 单次结果（AUROC=0.6169/AUPRC=0.3275），未补多 seed，不纳入本节的统计对比，只作单点参考。

**判断：L1→L2 的均值差距（AUROC +0.0194，AUPRC +0.0113）明显大于两组各自的标准差（L1 std=0.0050，L2 std=0.0107），4 个 seed 里 L2 全部高于对应 L1（L2 最低值 0.6190 仍高于 L1 最高值 0.6174）——门控相对独立编码器的增量是真实信号，不是单次训练随机性造成的假象**，之前只用 seed=42 单次跑得出的方向性结论在多 seed 下站得住。

同时发现一个新信息：**L2 的方差（0.0107）比 L1（0.0050）大一倍以上**。门控层引入的 `gate`/`value` 两组参数带来更大的随机初始化空间，不同 seed 下收敛到的解更分散——这提示门控机制不仅要看均值表现，训练稳定性也是评估维度，纳入下面"遗留 TODO"。

## 关键决策（不在 commit 里）

- **本轮定位是"收集数据"，不是"证明创新点"**：entry 010 的 novelty-check 结论已经确定——门控机制本身（`sigmoid` 门 + 仿射调制）结构上与 FiLM（Perez et al. 2018）/ GS-Fuse 属于同一机制家族，是十年历史的成熟机制类别，不能作为论文的核心创新点宣称。本轮 L1/L2 的角色是**强基线 / 组件候选**：如果 L2 相对 L0/L1 有明显收益，这是"content-adaptive modulation of broadcast-duplicated service-level features"这个方向有实际价值的证据，用来指导下一轮"真正的新融合模型"往哪个方向设计增量；如果收益很小，这也是一个有用的负向结果（说明简单门控吃不到多少油水，得往别处找增量）。不应把这篇 entry 或后续论文写成"我们的贡献是门控融合"。
- **L1 vs L2 不做参数量对齐**：010 记录的 L0→L1 confounding variable 问题（裸拼接→独立 encoder 本身引入了新的非线性容量，性能提升不能干净归因于"解决了粒度错配"）在 L1→L2 这一步不成立，原因不同——L2 的 `gate`/`value` 两层**是门控架构本身的组成部分**，不是与门控设计无关的额外容量。去掉 `gate`/`value` 就不再是门控机制，无法在"同参数量但去掉门控"的条件下做对照；参数量对齐 baseline 只在"想证明某个机制的贡献独立于其带来的容量增长"时才有意义，而 L1→L2 的实验问题本来就是"门控这整个机制（含其必然带来的参数量）值不值"，因此参数量差异是设计变量的一部分，不是混淆变量，不需要（也不应该）对齐。

## 坑 / 已知问题

- **严重坑（PR review 抓到，已修复）：`train_baseline_v0.py` 的 optimizer 从未包含 fusion 参数，L1/L2 首次提交的数字测的是随机初始化投影，不是训练后的融合模块**。`scripts/train_baseline_v0.py:135`（修复前）：
  ```python
  optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=svdd.parameters())
  ```
  这行代码配的注释是"fusion 无可训练参数，只传 svdd"——这个前提只对 L0（`EarlyConcatFusion`，纯 `torch.cat`，零参数）成立。L1 的 `ep_encoder`/`svc_encoder`（288 参数）、L2 额外的 `gate`/`value`（合计 1088 参数）都是真实的 `nn.Linear` 层，但从未被塞进 `optimizer` 的参数组——`loss.backward()` 会给这些层填上 `.grad`，但 `optimizer.step()` 只更新 SVDD，融合层全程停留在随机初始化。这意味着本 entry 最初提交的 L1/L2 结果实际测的是"随机投影 + 训练好的 SVDD"，不是"训练好的融合模块"，L1→L2 的对比结论没有实验支撑。
  修复：把 `fusion.parameters()` 并入 optimizer 的参数列表，同时在 `_train`/`_infer` 里补上 `fusion.train()`/`fusion.eval()`（此前只调了 `svdd.train()`/`svdd.eval()`，fusion 的训练/推理模式从未显式设置——L1/L2 目前没有 dropout/batchnorm 所以此前不影响数值，但补上是正确的做法，为未来引入这类层的融合机制兜底）。修复后重新跑了 L1/L2 训练+评估，上表数字已经是修复后的版本。
  **教训**：任何新增可训练参数的 `FusionModule` 子类接入这套训练脚本时，都要显式验证 `fusion.parameters()` 确实进了 optimizer——不能只看"跑起来不报错、loss 在降"就认为训练是对的，L1/L2 修复前的 loss 曲线看起来完全正常（单调下降），因为 SVDD 自己的参数仍在正常训练，只是融合层没有跟着动，这种失败模式不会在训练日志里留下任何异常信号，必须专门检查 optimizer 的参数组构成。
- **观察到但判断为正常现象、非 bug 的一点**：修复后 L2 最终 epoch loss（≈0.0003）比 L1（≈0.0075~0.013）低了一个数量级以上。这是 Deep SVDD one-class 目标下常见的现象——L2 多出的 `gate`/`value` 两层给了模型更大的容量把训练集（838 个 Normal 样本）里的正常表征收缩到超球心附近，训练 loss 更低是预期中的容量效应，不代表数值异常或训练不稳定；结合 eval AUROC/AUPRC 均正常（非 0.5、非 1.0、非 NaN）且分布合理，判断不是超球坍缩（hypersphere collapse）导致的退化。这一点留给后续设计参数量对齐实验时留意：容量越大，训练 loss 会越低，不能拿训练 loss 高低直接当作机制优劣的证据，必须看 eval 指标。

## 遗留 TODO

- **L3（全局标量门控）是否值得做**：optimizer bug 修复 + 多 seed 复测后，L2 相对 L1 的提升（均值 AUROC +0.0194，AUPRC +0.0113）在 4 个 seed 上是一致方向，不是单次训练的噪声。三组数字说明"引入非线性容量"（L0→L1）本身没有带来提升，而"引入门控"（L1→L2）才有，且这个结论现在有多 seed 支撑，比单 seed 阶段的结论更可信。是否继续做 L3 取决于下一轮讨论：如果目标是把消融矩阵做完整（L0-L3 全跑一遍是原计划），可以顺手补上（同样需要配多 seed，不能只跑单 seed 就下结论）；如果目标已经转向"设计真正的新融合模型"，L3 可能不再必要，直接拿 L1/L2 的数据去支撑新设计的动机即可。
- **"真正的新融合模型"该往哪个方向设计增量**：本轮数据显示门控收益存在但有限，暗示单纯的 sigmoid 门控调制已经接近这类简单机制在当前特征/数据规模下的天花板。下一轮设计新融合模型时，应该重点考虑 endpoint×time-window 粒度的时序结构或跨 case 的分布差异等门控机制没有利用到的信息，而不是在门控公式本身做更复杂的变体（如多头门控、多层门控），后者大概率边际收益更小且进一步逼近 FiLM 系机制家族，novelty 风险更高。
- **L2 训练稳定性比 L1 差，需要在后续设计中一并评估**：多 seed 复测显示 L2 的 AUROC 标准差（0.0107）是 L1（0.0050）的两倍以上——门控层的 `gate`/`value` 参数带来更大的随机初始化空间，不同 seed 收敛到的解更分散。后续如果门控或其变体要往前推进，除了看均值指标，也要把"同一架构在不同 seed 下的方差"作为评估维度之一，方差过大的机制即使均值好看，实际部署时的可靠性也存疑。
- **L0 尚未补多 seed**：目前 L0 只有 seed=42 单次结果，L1/L2 已经补了 seed=1/2/3。如果后续要严格论证"L0→L1 无提升"这个方向性结论，也需要给 L0 补上同样的多 seed 复测，目前的"L0 与 L1 相近"判断只是单点参考，还没有方差支撑。
