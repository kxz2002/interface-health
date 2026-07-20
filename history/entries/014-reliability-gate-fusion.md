# 014 · Reliability Gate Fusion：偏离量门控路由实测、可解释性分析与坍缩根因定位

- **日期**: 2026-07-18 ~ 2026-07-19
- **PR**: N/A（同分支多 commit，未开 PR）· **Commit**: 013dc40..598ef7a（本 entry 对应实测结果与坍缩根因排查的最新状态）
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
- **门控坍缩根因二轮排查后定位为"初始尺度温和不对称 + softmax 竞争性归一化持续放大"，不是鞍点、不是数值污染主导**（承接下方"坑/已知问题"里的坍缩现象）：逐 epoch 追踪 50 轮 `w_svc` 分布，全程单调爬升（epoch0 mean=0.69 → epoch50 mean=0.97），`frac(w_svc<0.01)` 全程恒为 0——模型从未探索过"信 ep 分支"，排除"早期坍缩后卡死在鞍点"的形态。进一步对比 `dev_ep`/`dev_svc` 原始 z-score：均值层面 dev_svc 是 dev_ep 的 ~92 万倍（`process_count` 等退化列的极端值污染），但**中位数层面 dev_ep=2.21 vs dev_svc=2.70，只有 1.2 倍温和差距**——这个温和差距才是驱动 epoch0 起 w_svc 就 >0.5 的真实原因，数值污染只拉高了 mean/max，不是中位数情形的主因。与本 entry 路由消融里 `independent_sigmoid` 优于 `softmax` 的证据相互印证。
- **严格 A/B 对比确认训练池扩容（838→6249 行）是坍缩的实质性放大因素，不是无关旁枝改动**：固定 `eval_all`/`endpoint_baseline_stats.json` 不变，只替换 `train.parquet`，梯度更新步数对齐（838 行版本跑 320 epoch≈1280 步，对齐 6249 行版本 50 epoch≈1250 步，排除"跑得不够久"的混杂因素）。结果：838 行版本 `frac(w_svc>0.99)` 全程稳定在 ~2.9%；6249 行版本从 epoch1 的 3.1% 持续单调爬升到 epoch50 的 50.7%。具体因果链未定位到底层特征分布差异（推测故障 case 的 `fault_baseline` 行在 svc 分支给出了更一致的"信 svc"梯度信号），需要后续单独验证。
- **A/B 实验第一次直接用 `contract_v1_838/endpoint_baseline_stats.json` 产出一组作废数据，已发现并重跑修正**：该 sidecar 与其对应的 `train.parquet`（同为 838 行、逐行数值 allclose 一致）本地重新 `fit()` 后结果不一致（部分列 std 相差 5 倍），判定为过时快照，未随代码/流程同步更新。**教训**：`artifacts/` 下长期复用的对照快照目录若无内容一致性校验，容易在事后复用时静默产出误导性结果，使用前必须验证，不能假设"存在于 `artifacts/` 下的产物互相一致"。
- **训练池扩容对既有实验（baseline_v1/l1*/l2*）的影响需拆成两层独立原因**：（1）master 自身遗留——`normalization.py` 零方差 bugfix 改了源码但未配套 `dvc repro` 更新 `dvc.lock`，本分支的 `093216e` 是第一个补跑全链路的 commit，`baseline_v0` 的 AUROC 变化（0.634→0.666，n_samples 不变）纯属这层，与 reliability_gate 特性代码无关；（2）本分支真正引入——`_write_v1()` 训练池扩容把故障 case 的 `baseline` 阶段行从 `eval_all` 移入训练池，`contract_v1` 系 n_samples 从 13632→8221，`baseline_v1`/`l1*`/`l2*` 的 AUROC/AUPRC 因此系统性变化。两层原因性质不同：前者是既有 bug 修复的副作用（跟本分支合不合并都会发生），后者是需要决策"是否接受"的实质性交叉污染。**已通过 `expand_train_pool` 开关解决（见下方"遗留 TODO"，24cbdee 落地）**：`configs/contract/v1.yaml` 默认 `false` 恢复原始 838/13632 规模（与 entry 012 可比），RG 专属实验改用 `configs/contract/v1_expanded_pool.yaml`（`true`）+ 独立产物目录 `artifacts/contract_v1_expanded/`，两条链路的 eval 集合不再互相污染。
- **决定合入 master，但 RG 保持非默认/不推荐使用状态，耦合债务留给下一个 PR 专项收束**：RG 的核心机制（softmax 竞争性门控）已确认效果不达标（见下方"坑/已知问题"），不构成"reliability gate 路由优于 baseline"的可用结论，但本分支沉淀的负面结果记录（坍缩根因定位、2×2 归因、路由消融）和基础设施（`EndpointBaselineStats`、`endpoint_id` 全链路打通、`expand_train_pool` 开关）有独立价值，值得进 master 供后续分支复用/参考，不必等 RG 本身达标才合并。合入前确认了默认路径不受影响：`configs/base.yaml` 默认 `fusion=concat`，`configs/contract/v1.yaml` 默认 `expand_train_pool=false`，L0/L1/L2 数值路径不因 RG 存在而改变。但合入确实往公共代码里留了几个 RG 专属的耦合点（清单见下方"遗留 TODO"），这些不是零成本隔离，是明确的技术债务，留给后续专项 PR 处理，不在本次范围内解决。

## 坑 / 已知问题

- **RG 的方差远大于 L0/L1/L2，主对比的"RG 更好"结论目前站不住**：RG 4 个 seed 的 AUROC 标准差（0.0792）是 L0（0.0161）的近 5 倍、L1（0.0334）/L2（0.0306）的约 2.5 倍。RG 的 seed 范围（0.4481~0.6130）跨越了 L0/L1/L2 各自的均值区间，4 个 seed 里 RG 只有 2 个（seed=2/42）优于 L2 均值，另 2 个（seed=1/3）明显劣于全部三个 baseline。**如实报告：本轮数据不支持"reliability gate 路由优于门控/独立编码器/裸拼接"这个方向性结论，RG 的高方差本身就是需要解决的问题，不是可以忽略的噪声**。
- **可解释性分析发现门控已坍缩为几乎恒定的 `w_svc≈1.0`，未体现设计假设的 HTTP/资源故障路由差异**：用 `scripts/analyze_gate_weights.py` 对 `rg_seed42`（真实训练权重，非随机初始化）按 `anomaly_type` 分层统计发现，**全部 27 种故障类型 + Normal 的 `w_svc` 均在 0.9994~1.0 之间**（`w_ep` 对应在 1e-13~1e-4 量级），包括预期 `w_ep` 该占主导的 `Lv_E_HTTPABORT_*`/`Lv_E_HTTPPATCH_*` 等 endpoint 级 HTTP 故障类型。这说明：(1) 门控机制学到的是"几乎总是信 service 分支"，而非"按故障类型动态路由"；(2) softmax 竞争性归一化 + `branch_aware` 偏离量输入这套设计在当前数据/训练配置下没有产生预期的可解释性效果，"HTTP 类故障 w_ep 高、资源类故障 w_svc 高"的假设未被证实。**根因已在后续排查中定位，见上方"关键决策"**：不是鞍点、不是数值污染主导，是初始尺度 1.2 倍温和不对称被 softmax 竞争性归一化持续放大，训练池扩容会显著加速这个放大过程。
- **发现 `eval_all` 的正负样本比例被本分支训练池扩容逻辑实质性改变，且违反项目既有约定**：实测 `eval_all.parquet`（8221 行）按 `phase` 拆分：`inject`=6875（正样本）、`recover`=1110（负样本）、`normal`=236（负样本），负样本总量从旧版 6757（=5411 baseline+1110 recover+236 holdout）骤降到 1346，正负比从约 50/50 变成约 84/16。`CLAUDE.md`/`docs/agent-docs/dataset-guide.md` 明确写着"负样本（训练+评估）：Normal case 全程 + 异常 case 的 baseline/recover 阶段"，现在的实现把 baseline 阶段整段移出 eval，与文档规定的评估协议本身不一致。AUPRC 对类别比例极敏感，上表记录的 AUPRC 普涨（如 baseline_v1 从 0.327→0.483，见分支影响排查的对照记录）很可能主要是类别比例漂移的假象，不是模型质量提升的真实效果。
- **门控坍缩的梯度来源与 eval 类别比例是两个独立问题，不要混为一谈**：`gate_mlp` 只从 `train.parquet` 上的 SVDD loss 反向传播，`eval_all` 从不参与训练，"eval 类别失衡导致坍缩"这条因果链机制上不成立；但反过来，训练池扩容改 `eval_all` 摘除逻辑时，容易在无意中破坏 eval 集合本身的类别平衡（即上一条问题），这是独立于训练动态之外必须单独核查的正确性问题。
- **`artifacts/contract_v1_838/` 快照目录内部曾发现不一致**：`endpoint_baseline_stats.json` 与同目录的 `train.parquet` 不匹配（本地重新 fit 结果不同，部分列 std 相差 5 倍），推测是早期生成后未随后续代码改动（如 shrinkage 参数调整）重新落盘。**教训**：使用 `artifacts/` 下长期复用的对照快照目录前必须验证内容一致性，不能假设产物互相同步。
- **`fixed_uniform` 消融模式会静默绕过 `endpoint_id` 未知时的 `KeyError` 契约**：`gate_weights` 在 `gate_normalization="fixed_uniform"` 时提前返回常数 `[0.5, 0.5]`，不会走到 `endpoint_id` 查表那一步，因此传入一个不在 `id_to_endpoint_key` 映射里的 `endpoint_id` 不会触发 Task 2 建立的 `test_unknown_endpoint_id_in_batch_raises` 契约。判断这是可接受的行为而非 bug：`fixed_uniform` 的设计前提就是"完全不看输入"，没有发生任何 endpoint 特定的计算，不存在数据被错误归因的风险；但这是当前唯一一个"非法 endpoint_id 不报错"的模式，若未来在生产邻近代码里复用该模式需要额外加一道校验，本轮只是一次性消融实验，不需要。
- **2×2 归因显示训练池扩容对 L2 和 RG 的影响方向相反**：扩容后 L2 AUROC 从 0.6074 降到 0.5744（-0.0330），RG 从 0.5542 升到 0.6024（+0.0482）。这提示"扩容训练池"这个数据侵入本身不是单向利好——对 L2 这种直接吃特征拼接的机制，扩容引入的 `fault_baseline` 行可能带来分布噪声（故障 case 的 baseline 阶段特征分布未必与纯 Normal case 完全一致）；对 RG，由于门控输入是相对 endpoint 基线的偏离量（基线仍固定在 838 行纯 Normal 上，不受扩容影响），扩容只增加了训练信号量而不改变门控的参照系，因此看到正向增益。这个不对称性值得在后续设计里留意：训练池扩容的收益/风险取决于下游机制是直接消费扩容后的原始特征，还是只用扩容部分增加梯度信号量。
- **稳健性检查：shrinkage 对稀疏 endpoint（login，仅 17 行）的 std 估计有实质性收缩**：`shrinkage=True` vs `False` 对比，`login` 的 `endpoint_red` 分支部分列 std 收缩幅度达 12%~24%（如 `trace_latency_p50` 从 0.2462 收缩到 0.1980，-19.6%；`client_request_count` 从 0.4573 收缩到 0.3960，-13.4%），`service` 分支部分列收缩 7%~20%（`event_rate`/`template_diversity` 均超过 19%）；作为对照，密集 endpoint（`order/refresh`，131 行）的收缩幅度小得多且方向不总一致（部分列甚至轻微反向，±0.2%~2%），符合"shrinkage-to-global 只在样本量不足时发挥作用，样本充足时趋于中性"的设计预期。这验证了 shrinkage 机制本身按设计工作，但没有回答"shrinkage 开/关是否影响下游 AUROC/AUPRC"——本轮未跑这个下游对比实验，留作遗留 TODO。

## 遗留 TODO

- spec §7 明确排除项原样搬入：不做 sibling cross-attention（endpoint↔service 是 1:1 映射，无同伴信号）；不做 endpoint 时序序列建模（数据量不足，稀疏 endpoint 单 case 仅个位数窗口）；不做 session-conditional 数值校正作为独立创新（审稿判定强度不足）；不做逐特征 z-score 全维门控输入（会退化为特征门控，novelty 故事受损）；不把 mixed-effects/fixed-effect/random-effect 术语写进论文。
- 目标耦合式融合（SVDD 超球体距离反馈门控）作为 plan B 留痕：本 spec 不涵盖，仅当 reliability gate 实证效果持续不佳时再另立设计（且需约束式设计规避循环论证风险）。鉴于本轮 RG 的高方差和门控坍缩问题，这个 plan B 触发条件可能已经成立，值得下一轮讨论是否启动。
- ~~门控坍缩问题需要专项调查~~ **已定位，见上方"关键决策"**：根因是初始尺度 1.2 倍温和不对称被 softmax 竞争性归一化持续放大，训练池扩容会加速这个过程；不是鞍点、不是数值污染主导。
- **训练池扩容改变坍缩动态的底层机制未定位**：只确认了"扩容会加速/加剧坍缩"这个因果关系（A/B 对比，已排除梯度步数混杂因素），未定位到具体是 `fault_baseline` 行在 svc 分支上的偏离量分布相比 `normal_case` 行有何不同（更集中？均值偏移？）导致梯度更倾向于"信 svc"。需要单独对比两类行的 `dev_svc`/`dev_ep` 分布。
- ~~`_write_v1()` 训练池扩容建议做成 config 可控开关~~ **已实现（24cbdee）**：`expand_train_pool: bool`，默认 `false` 保持旧行为，新增 `configs/contract/v1_expanded_pool.yaml` 供 reliability_gate 专属实验用，产物目录隔离（`contract_v1` / `contract_v1_expanded`）。实测验证：`contract_v1` 恢复 train_fit=838/eval_all=13632，`contract_v1_expanded` 为 train_pool=6249/eval_all=8221。**该切分仍是"进训练池就从 eval 摘除"的整批吸收，未采用下方 eval_all 类别比例问题条目里讨论的"按时间窗切分、部分留在 eval"方案**——这两个 TODO 独立，开关只解决了"能否跟旧实验比"，没有解决"eval 类别比例失衡"。
- **`eval_all` 类别比例问题需要独立修复，优先级判断应先于训练池开关**：因为它已经不是"新旧实验可比性"层面的问题，是当前 `contract_v1` 评估协议本身违反项目文档既有约定。倾向的修法是"baseline 阶段行应同时进训练池、也保留在 eval_all 里"（只要不是同一行同时出现在两处，不构成行级泄漏），而非现在的"进训练池就从 eval 摘除"。具体方案未定，需要下一轮讨论。
- **shrinkage 开/关对下游检测指标（AUROC/AUPRC）的影响未测**：本轮只验证了 shrinkage 机制本身按设计工作（收缩幅度符合预期），未跑"shrinkage=True vs False 训练出的模型在稀疏 endpoint 上的检测效果差异"这个下游对比。
- **RG 高方差的根因未定位**：是否与门控坍缩问题同源（不同 seed 收敛到不同的坍缩终点）、还是独立的训练不稳定性来源，本轮排查集中在坍缩机制本身，未回答这个问题。
- **RG 耦合点收束（下一个专项 PR 范围）**：合入 master 时未清理，留在此处作为下一个 PR 的范围清单——
  - `scripts/train_baseline_v0.py:142-156` 按 `cfg.fusion._target_` 字符串特判 `ReliabilityGatedFusion`，命中时才注入 `endpoint_baseline_stats`/`id_to_endpoint_key`/`endpoint_id` 等 kwargs，写在所有 fusion 共享的训练脚本里，不是隔离在 RG 自己文件内。收束方向：抽象成 fusion 自己声明"是否需要额外运行时 kwargs"的接口（如 `FusionModule` 加一个可选的 `extra_kwargs_provider`），而非在训练脚本里按类名分支。
  - `scripts/build_contract.py:342` 只要 `contract_version=="v1"` 就无条件派生 `endpoint_id_map` 并 fit/落盘 `endpoint_baseline_stats.json`，不管当前实验实际用不用 RG——L0/L1/L2 从不消费这两个产物，但每次 `build_contract_v1` 都会白算一遍。收束方向：视情况改为按需生成（如 contract config 里加一个显式开关），或确认这个开销可接受、保留现状但补一句注释说明"这是为 RG 铺路，L0/L1/L2 不用"。
  - `dvc.yaml` 的 `build_contract_v1_expanded`/`train_v1_reliability_gate`/`eval_v1_reliability_gate` 三个 stage 进了默认 DAG，任何人跑不带 stage 名的全量 `dvc repro` 都会连带跑这三个 RG 专属 stage（build_contract 实测耗时约 20 分钟）。收束方向：评估是否应该从默认 `dvc repro` 的隐式全量目标里摘除，改为必须显式指定 stage 名才跑。
  - `configs/fusion/reliability_gate.yaml` + 3 个 ablation 变体、`configs/contract/v1_expanded_pool.yaml` 会一直留在对应目录下，浏览者需要靠本 entry 或代码注释才知道"已验证效果不佳、不建议默认使用"，没有代码层面的强制力（如 deprecation warning）。收束方向：视需要决定是否要加运行时警告，或接受"文档层面说明已足够"。
