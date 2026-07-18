# 014 · Reliability Gate Fusion：偏离量门控路由实测与可解释性分析

- **日期**: 2026-07-18
- **PR**: N/A（同分支多 commit，未开 PR）· **Commit**: 013dc40..（本 entry 对应实测结果的最新 commit）
- **类型**: Experiment
- **影响域**: `src/fusion/`, `src/data/endpoint_baseline_stats.py`, `src/contracts/`（`endpoint_id`/训练池扩容）, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `scripts/analyze_gate_weights.py`, `configs/fusion/`

## 做了什么

在 011/012 的融合消融基础设施之上，实现了 `ReliabilityGatedFusion`（`src/fusion/reliability_gate.py`）：门控输入不是 L2 GatedFusion 的原始特征拼接，而是两分支各自相对自身 endpoint normal 基线的偏离摘要（`dev_ep`/`dev_svc`，来自 `EndpointBaselineStats` sidecar），归一化用 softmax（竞争性二选一）而非逐维 sigmoid，融合形式对称加权 `w_ep*value_ep + w_svc*value_svc`。配套改动：`EndpointBaselineStats` fit-only-on-train_fit 的 sidecar（含 shrinkage-to-global 稀疏 endpoint 收缩）、`endpoint_id` 列打通 contract→dataloader→fusion 全链路、训练池扩容（838→8221 行，吸收故障 case 的 `baseline` 阶段行、不吸收 `recover`）、`ReliabilityGatedFusion` 的三个消融开关（`deviation_mode`、`gate_normalization`）。

多 seed（1/2/3/42）在扩容后的 Contract v1（n_samples=8221）上跑通 L0/L1/L2/RG 四组主对比、2×2 训练池归因、三个路由消融：

| 变体 | seed=1 | seed=2 | seed=3 | seed=42 | 均值 | 标准差 |
|------|--------|--------|--------|---------|------|--------|
| L0 AUROC | 0.5573 | 0.5567 | 0.5769 | 0.5374 | 0.5571 | 0.0161 |
| L1 AUROC | 0.5489 | 0.6197 | 0.6050 | 0.6184 | 0.5980 | 0.0334 |
| L2 AUROC | 0.5745 | 0.6391 | 0.5911 | 0.5744 | 0.5948 | 0.0306 |
| RG AUROC | 0.4481 | 0.6130 | 0.5060 | 0.6024 | 0.5424 | 0.0792 |
| L0 AUPRC | 0.4921 | 0.4881 | 0.5126 | 0.4831 | 0.4940 | 0.0130 |
| L1 AUPRC | 0.5053 | 0.5700 | 0.5477 | 0.5583 | 0.5453 | 0.0282 |
| L2 AUPRC | 0.4960 | 0.5825 | 0.5219 | 0.5218 | 0.5306 | 0.0367 |
| RG AUPRC | 0.3515 | 0.5270 | 0.4467 | 0.5278 | 0.4632 | 0.0837 |

2×2 训练池归因（838 行原始 vs 8221 行扩容 × {L2, RG}，seed=42）：

| | 838 行 | 扩容后 | Δ AUROC |
|---|---|---|---|
| L2 | 0.6074 | 0.5744 | -0.0330 |
| RG | 0.5542 | 0.6024 | +0.0482 |

路由消融（seed=42，参照组 = RG 默认配置 seed42，AUROC=0.6024）：

| 变体 | AUROC | AUPRC |
|------|-------|-------|
| `scalar` deviation（去分支细分） | 0.4286 | 0.3855 |
| `independent_sigmoid`（去竞争归一化） | 0.6598 | 0.5872 |
| `fixed_uniform`（禁用门控，下界对照） | 0.5201 | 0.4244 |

## 关键决策（不在 commit 里）

- **softmax 而非 sigmoid**：数据信号上 HTTP 类故障与资源类故障对 endpoint/service 两分支的偏离模式此消彼长（一个分支显著偏离时另一个分支往往正常），竞争性二选一（softmax，权重和恒为1）比独立开关（sigmoid，两个开关互不相关）更贴合这个假设的机制形状。但见下方"坑/已知问题"——本轮实测的可解释性分析显示这个假设在训练后的模型上并未体现出来。
- **偏离量表示而非原始特征作为门控输入**：一是机制约束（reliability-aware 路由的核心卖点是"信哪个分支更可靠"，天然活在异常度量空间而非特征空间，这是与 L2 GatedFusion 的本质区分点，不是同一机制的变体）；二是防过拟合（`branch_aware` 模式下门控输入只有 6 维，比 L2 直接吃拼接特征的输入维度小得多，缓解 838 行小样本下的过拟合风险）。
- **只吸收 baseline 不吸收 recover**：`recover` 阶段系统尚未验证稳定回到正常态，分布未知，保守排除，只把已确认"系统仍处于故障发生前正常运行"的 `baseline` 阶段行吸收进训练池（源头见 009/011 遗留问题，本轮 Task 6 落地）。
- **2×2 归因用 `train_fit.parquet` 替换 `train.parquet` 而非重新切分**：`eval_all.parquet`/`schema.json`/`endpoint_baseline_stats.json`/`normalization_stats.json` 的 fit 范围本来就固定在 `train_fit`（跟训练池扩不扩容无关），直接复用整份，只替换 `train.parquet` 的内容，避免为一次性归因实验重新走一遍完整 build_contract 流程。
- **`artifacts/contract_v1` 全量重建（l0/l1/l2 的 metrics.json 因此改变）是必要代价，不是意外副作用**：Task 4（`endpoint_id`/sidecar）+ Task 6（训练池扩容）改变了 `contract_v1` 的内容和 eval 集合大小（13632→8221），若不重建就跑 RG，会让 RG 和 L0/L1/L2 分别评估在不同的 eval 集合上，主对比失去意义。重建后 l0/l1/l2 的数字随之变化，属预期，不是回归。**entry 012 发布的 L1/L2 数字（n_samples=13632，L1 均值 AUROC=0.6103/AUPRC=0.3241，L2 均值 AUROC=0.6297/AUPRC=0.3354）现已被本 entry 的新数字（n_samples=8221，见上表）取代**——entry 012 保留不改，作为历史快照，但读者若拿 entry 012 的表格去对照当前 `artifacts/l1/metrics.json`/`artifacts/l2/metrics.json` 会发现数字不一致，这是 Task 4/6 的 contract 重建导致，不是文件损坏或误操作。`history/index.md` 第 101 行"横切主题"里引用的 L0/L1/L2 数字（AUROC 0.6169/0.6082/0.6316）另有出处（既不匹配 entry 012 单 seed 数字也不匹配其多 seed 均值，疑似 entry 012 早期草稿或口误遗留），本次一并更正为当前实测值。

## 坑 / 已知问题

- **RG 的方差远大于 L0/L1/L2，主对比的"RG 更好"结论目前站不住**：RG 4 个 seed 的 AUROC 标准差（0.0792）是 L0（0.0161）的近 5 倍、L1（0.0334）/L2（0.0306）的约 2.5 倍。RG 的 seed 范围（0.4481~0.6130）跨越了 L0/L1/L2 各自的均值区间，4 个 seed 里 RG 只有 2 个（seed=2/42）优于 L2 均值，另 2 个（seed=1/3）明显劣于全部三个 baseline。**如实报告：本轮数据不支持"reliability gate 路由优于门控/独立编码器/裸拼接"这个方向性结论，RG 的高方差本身就是需要解决的问题，不是可以忽略的噪声**。
- **可解释性分析发现门控已坍缩为几乎恒定的 `w_svc≈1.0`，未体现设计假设的 HTTP/资源故障路由差异**：用 `scripts/analyze_gate_weights.py` 对 `rg_seed42`（真实训练权重，非随机初始化）按 `anomaly_type` 分层统计发现，**全部 27 种故障类型 + Normal 的 `w_svc` 均在 0.9994~1.0 之间**（`w_ep` 对应在 1e-13~1e-4 量级），包括预期 `w_ep` 该占主导的 `Lv_E_HTTPABORT_*`/`Lv_E_HTTPPATCH_*` 等 endpoint 级 HTTP 故障类型。这说明：(1) 门控机制学到的是"几乎总是信 service 分支"，而非"按故障类型动态路由"；(2) softmax 竞争性归一化 + `branch_aware` 偏离量输入这套设计在当前数据/训练配置下没有产生预期的可解释性效果，"HTTP 类故障 w_ep 高、资源类故障 w_svc 高"的假设未被证实。**如实报告，不掩盖**——这与"路由消融里 `independent_sigmoid` 反而比默认 `softmax` 更优（AUROC 0.6598 vs 0.6024）"这一发现相互印证：可能 softmax 的竞争性归一化在训练动态上更容易坍缩到某一端主导，而独立 sigmoid 允许两个权重独立变化，更不容易坍缩。
- **`fixed_uniform` 消融模式会静默绕过 `endpoint_id` 未知时的 `KeyError` 契约**：`gate_weights` 在 `gate_normalization="fixed_uniform"` 时提前返回常数 `[0.5, 0.5]`，不会走到 `endpoint_id` 查表那一步，因此传入一个不在 `id_to_endpoint_key` 映射里的 `endpoint_id` 不会触发 Task 2 建立的 `test_unknown_endpoint_id_in_batch_raises` 契约。判断这是可接受的行为而非 bug：`fixed_uniform` 的设计前提就是"完全不看输入"，没有发生任何 endpoint 特定的计算，不存在数据被错误归因的风险；但这是当前唯一一个"非法 endpoint_id 不报错"的模式，若未来在生产邻近代码里复用该模式需要额外加一道校验，本轮只是一次性消融实验，不需要。
- **2×2 归因显示训练池扩容对 L2 和 RG 的影响方向相反**：扩容后 L2 AUROC 从 0.6074 降到 0.5744（-0.0330），RG 从 0.5542 升到 0.6024（+0.0482）。这提示"扩容训练池"这个数据侵入本身不是单向利好——对 L2 这种直接吃特征拼接的机制，扩容引入的 `fault_baseline` 行可能带来分布噪声（故障 case 的 baseline 阶段特征分布未必与纯 Normal case 完全一致）；对 RG，由于门控输入是相对 endpoint 基线的偏离量（基线仍固定在 838 行纯 Normal 上，不受扩容影响），扩容只增加了训练信号量而不改变门控的参照系，因此看到正向增益。这个不对称性值得在后续设计里留意：训练池扩容的收益/风险取决于下游机制是直接消费扩容后的原始特征，还是只用扩容部分增加梯度信号量。
- **稳健性检查：shrinkage 对稀疏 endpoint（login，仅 17 行）的 std 估计有实质性收缩**：`shrinkage=True` vs `False` 对比，`login` 的 `endpoint_red` 分支部分列 std 收缩幅度达 12%~24%（如 `trace_latency_p50` 从 0.2462 收缩到 0.1980，-19.6%；`client_request_count` 从 0.4573 收缩到 0.3960，-13.4%），`service` 分支部分列收缩 7%~20%（`event_rate`/`template_diversity` 均超过 19%）；作为对照，密集 endpoint（`order/refresh`，131 行）的收缩幅度小得多且方向不总一致（部分列甚至轻微反向，±0.2%~2%），符合"shrinkage-to-global 只在样本量不足时发挥作用，样本充足时趋于中性"的设计预期。这验证了 shrinkage 机制本身按设计工作，但没有回答"shrinkage 开/关是否影响下游 AUROC/AUPRC"——本轮未跑这个下游对比实验，留作遗留 TODO。

## 遗留 TODO

- spec §7 明确排除项原样搬入：不做 sibling cross-attention（endpoint↔service 是 1:1 映射，无同伴信号）；不做 endpoint 时序序列建模（数据量不足，稀疏 endpoint 单 case 仅个位数窗口）；不做 session-conditional 数值校正作为独立创新（审稿判定强度不足）；不做逐特征 z-score 全维门控输入（会退化为特征门控，novelty 故事受损）；不把 mixed-effects/fixed-effect/random-effect 术语写进论文。
- 目标耦合式融合（SVDD 超球体距离反馈门控）作为 plan B 留痕：本 spec 不涵盖，仅当 reliability gate 实证效果持续不佳时再另立设计（且需约束式设计规避循环论证风险）。鉴于本轮 RG 的高方差和门控坍缩问题，这个 plan B 触发条件可能已经成立，值得下一轮讨论是否启动。
- **门控坍缩问题需要专项调查，而不是直接宣称路由机制失败**：本轮只观察到坍缩现象（`w_svc≈1.0` 恒定）和一个侧面证据（`independent_sigmoid` 优于 `softmax`），未定位根因。可能方向：(1) softmax 温度/初始化导致训练早期就走向某一端主导后无法逃逸的鞍点；(2) 偏离量输入（`dev_ep`/`dev_svc`）两分支的数值尺度差异过大，导致 `gate_mlp` 学到的实际是"看哪个分支的偏离范数更大的数值量级"而非"哪个分支更可信"这个语义；(3) SVDD one-class loss 本身没有直接监督门控权重的信号，门控只能通过间接梯度学习，收敛慢或容易卡在局部解。需要专项实验（如可视化训练过程中 `w_ep`/`w_svc` 的演化曲线、检查 `dev_ep`/`dev_svc` 各自的数值分布范围）才能下结论。
- **shrinkage 开/关对下游检测指标（AUROC/AUPRC）的影响未测**：本轮只验证了 shrinkage 机制本身按设计工作（收缩幅度符合预期），未跑"shrinkage=True vs False 训练出的模型在稀疏 endpoint 上的检测效果差异"这个下游对比。
- **RG 高方差的根因未定位**：是否与门控坍缩问题同源（不同 seed 收敛到不同的坍缩终点）、还是独立的训练不稳定性来源，需要下一轮结合上一条的训练过程可视化一并排查。
