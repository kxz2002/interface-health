# DeviationWeightedFusion — Design Spec

**Date:** 2026-07-27
**Branch:** `feature/deviation-weighted-fusion`
**Status:** Design approved, pending spec self-review + user review before implementation plan.

## 1. 背景与动机

### 1.1 研究定位的转折

原"per-endpoint shortcut"假设已被 Phase 0 数据诊断证伪（`history/entries/017-phase0-shortcut-refuted-reframe.md`）：HTTP 层故障（Lv_E）下 service 级 metric/log 几乎不动（0.02σ/0.15σ），没有被污染的共享协变量可传染给同 service 无辜邻居，shortcut 没有触发条件。外部审稿人（gpt-5.6-sol）裁定继续用"shortcut"框架是误导，论文脊柱转向 **"Reliability ≠ Observability"**：一个模态可以在 Normal 数据上低方差、稳定（看起来"可靠"），却对特定故障完全不变（实际无用）；把"稳定"当"可信"的 reliability gate 会压制唯一对故障敏感的分支。这直接解释了 `2026-07-16-reliability-gate-fusion-design.md`（entry 014）中 `ReliabilityGatedFusion` softmax 门控坍缩到 `w_svc≈1.0` 的负结果。

### 1.2 C 实测确认的精确靶心：信号淹没

entry 017 的 C 实测（`scripts/analyze_modality_observability.py` → `artifacts/modality_observability/report.md`）把"service 全盲"这个粗糙说法修正为更精确的诊断——**信号淹没（signal drowning）**：

- ABORT/REPLACE 故障下，单特征 `client_error_rate` 的 oracle AUROC 达到 0.993/1.000（几乎完美可分）。
- 但朴素等权融合（`EarlyConcatFusion`/`IndependentConcatFusion` 等权 concat + Deep SVDD 等权 L2 距离）把这一维和其余 9~17 维噪声/弱信号维度平均掉，AUROC 降到 0.278~0.499（endpoint_only 10 维 / all 18 维），部分甚至反相关。
- 根因：ABORT 类故障快速失败，导致 latency/request_count 等维度反而更接近 normal 中心，`error_rate` 这一维稀疏但极强的信号被 L2 距离对全部维度的等权求和平均稀释。

这是"Reliability≠Observability"的**特征级版本**：`client_error_rate` 这一维在故障发生时高度不稳定（对该故障"可观测"），但融合层和打分公式都把它当作和其他稳定维度等权重的普通一维处理，等价于把"这一维现在极不像 normal"的信息直接丢弃。

### 1.3 本设计的定位：最小改动验证，非最终方法

用户明确要求先做**最小改动**验证"逐特征可信度加权能否救回这个 gap"，如果验证不达标，不做后续深化（可学习阈值、动 SVDD 打分公式）。本设计因此有意排除多个"更完整"的候选机制（分支级门控、可学习阈值、Mahalanobis 距离改造 SVDD），把它们记录为条件触发的未来工作，而不是本轮实现范围。

## 2. 核心机制：DeviationWeightedFusion

### 2.1 数据流（单样本）

```
raw row (endpoint_red 10维 + service_metric/service_log 合并 8维 + endpoint_id)
  → 拆分 raw_ep (endpoint_red) / raw_svc (service_metric + service_log 合并列)
  → 逐特征 z-score：z_ep = (raw_ep - mean_ep) / std_ep   [EndpointBaselineStats.branch_stats(eid, "ep")]
              z_svc = (raw_svc - mean_svc) / std_svc  [EndpointBaselineStats.branch_stats(eid, "svc")]
  → 逐特征权重：w = sigmoid(|z| - 2.0)   （阈值固定复用 RG 的 _DEVIATION_THRESHOLD）
  → 退化列（EndpointBaselineStats.degenerate_columns(branch) 命中）权重强制置 1，不参与上式
  → weighted_ep = raw_ep * w_ep;  weighted_svc = raw_svc * w_svc
  → concat([weighted_ep, weighted_svc])   ← 无分支级门控，无编码器，直接拼接
  → 交给现有 DeepSVDD（打分公式不变）
```

### 2.2 与现有 L0/L1/L2/RG 的关键区别

| 维度 | L0 EarlyConcat | RG ReliabilityGate | DeviationWeightedFusion（本设计） |
|------|---------------|---------------------|-----------------------------------|
| 加权粒度 | 无（等权原样拼接） | 分支级（w_ep vs w_svc，2 个标量） | **逐特征**（每一维一个权重） |
| 门控输入 | — | 分支偏离摘要（norm/max_abs/frac_exceed，3维/分支） | 该特征自身的 `\|z\|`（标量，逐维独立） |
| 归一化 | — | softmax（竞争性二选一，坍缩已知风险） | 逐维独立 sigmoid（无竞争，不存在 softmax 坍缩机制） |
| 是否有编码器 | 无 | 有（ep_encoder/svc_encoder + value_ep/value_svc） | **无**——只做逐元素重加权，不引入 Linear 层 |
| 可学习参数量 | 0 | 中（gate_mlp + 编码器） | **0**（阈值固定，无网络） |
| output_dim | `ep_dim + svc_dim` | `branch_dim` | `ep_dim + svc_dim`（与 L0 严格一致） |

设计选择"和 L0 只差一个变量（是否逐特征加权）"是有意为之：这是单变量验证，不是完整方法提案。选择"不做分支级门控"是为了避免复现 RG 已诊断的 softmax 竞争归一化正反馈坍缩机制（entry 014），并保证结果可以干净归因到"逐特征加权本身有没有用"，不被叠加的分支竞争机制混淆。

### 2.3 加权公式选择理由

`weighted = feature * sigmoid(|z| - threshold)`：

- `sigmoid` 把权重限制在 (0,1)，数值稳定（不同于 `z * feature` 直接相乘可能因为大 `|z|` 把某一维数值炸到远超其余维度）。
- `|z|` 小于阈值时权重趋近 0（该维和 normal 基线接近，判定为当前不携带异常信息，压低）；`|z|` 大于阈值时权重趋近 1（该维显著偏离，判定为携带信息，原样保留）。
- 阈值固定为 2.0，直接复用 `ReliabilityGatedFusion._DEVIATION_THRESHOLD` 的既有语义（"|z|>2.0 算显著偏离"），避免同一项目内出现两套不一致的偏离显著性标准。

### 2.4 退化列处理

`EndpointBaselineStats.fit()` 对全局零方差/全 NaN 列（如 Normal 期间恒为 0 的 `trace_5xx_rate`）会把 `std` 兜底到 `_FALLBACK_STD_EPSILON=1e-9`。若不特殊处理，eval 阶段该列一旦出现非零值，`z=(raw-mean)/std` 会除以 1e-9 炸到 1e6~1e9 量级——数值上 `sigmoid` 不会溢出，但语义上这是被除零放大的假信号（`std` 是人为兜底值不是真实统计量），不是"淹没"问题要救的那种真实信号。

处理策略：`EndpointBaselineStats.degenerate_columns(branch)` 命中的列，权重直接固定为 1（不调制，原样通过），不参与 `sigmoid(|z|-threshold)` 计算。这是最保守的处理——没有额外证据支持"该维当前异常"，就不特殊处理，回退到原始特征原样通过。

### 2.5 endpoint_id 缺失兜底

若 `endpoint_id is None`（接口兼容路径，如被 L0/L1/L2 的兼容调用方式意外调用），退化为全部特征权重为 1，等价于纯 concat（即退化成 L0 EarlyConcatFusion 的行为）。实际训练/评估流程中 `endpoint_id` 总有值，这条路径只是接口层防御。

### 2.6 明确排除（本轮不做，记录为条件触发的未来工作）

- **可学习阈值**：`threshold` 不做成 `nn.Parameter`。若本轮验证达标（见第 4 节判定标准）且需要进一步提升，下一步候选是把阈值改为逐特征可学习参数——先验证"暴露逐维偏离信息"这个方向本身有没有用，再决定是否值得多引入一组可学习自由度（多参数=多一层"是否调参凑出来"的审稿风险，应该在有收益证据后才引入）。
- **分支级门控**：不叠加 RG 式 `w_ep`/`w_svc` softmax 竞争。理由见 2.2。
- **改造 Deep SVDD 打分公式**：`DeepSVDD.score()` 目前是等权 L2 距离（`((z-center)**2).sum(dim=1)`），本身也是"淹没"机制的一部分——即使融合层完美放大了信号维度，等权 L2 距离仍可能被 `rep_dim=32` 里的噪声维度平均稀释。但本轮设计只改融合层，不改 SVDD 打分公式，理由是最小改动优先验证"改融合层是否已经够"；只有本方案验证不达标，才需要考虑改打分公式（如 Mahalanobis 距离/逐维可学习权重）。
- **三分支拆分（service_metric vs service_log 独立门控）**：`EndpointBaselineStats` 目前只有 `ep`/`svc` 两个分支（`svc` 内部混合 service_metric + service_log 列），本设计不新增第三分支，逐特征加权本身已经能覆盖 `svc` 分支内部 service_log 被 service_metric 平均掉的问题（因为是逐列而非分支级摘要）。

## 3. 组件划分与接入

### 3.1 技术前提（已核实，无需新增统计基建）

`EndpointBaselineStats`（`src/data/endpoint_baseline_stats.py`）已经是逐列拟合的：`fit()` 对 `red_cols`（→`ep` 分支）和 `svc_cols`（→`svc` 分支，本身就是 service_metric + service_log 列拼接）分别算逐列 mean/std，`branch_stats(endpoint_key, branch)` 返回整组列各自的 mean/std 向量。本设计需要的逐特征 z-score 所需数据已经存在，不需要修改 `EndpointBaselineStats`。

### 3.2 新增：`DeviationWeightedFusion`

- **位置**：`src/fusion/deviation_weighted.py`
- **接口**：继承 `FusionModule`（`src/fusion/base.py`），实现 `forward(modality_dict, endpoint_id=None)` 和 `output_dim`。
- **构造**：需要 `EndpointBaselineStats` 实例（同 RG 的构造模式），通过覆写 `from_contract()` 从 `contract_dir/endpoint_baseline_stats.json` 加载，复用 RG 已有的 `_derive_id_to_endpoint_key` 反查逻辑（`src/contracts/endpoint_id_mapping.py`）。
- **配置**：新增 `configs/fusion/deviation_weighted.yaml`（`_target_` 指向新类，`threshold: 2.0` 作为唯一超参，默认不暴露为消融维度但保留字段方便后续实验）。
- **依赖前提**：需要 `fit_endpoint_baseline_stats=true` 的 contract（`configs/contract/v1_expanded_pool.yaml`），与 RG 相同，不能用 `v1.yaml`。

### 3.3 无需改动的部分

- `EndpointBaselineStats`：数据结构和拟合逻辑已满足需求，不改。
- `DeepSVDD`：打分公式不变（见 2.6 排除项）。
- `scripts/build_contract.py` / `scripts/train_baseline_v0.py`：无需新增列或 collate 逻辑，`endpoint_id` 已由 RG 的既有基础设施提供。
- `scripts/eval_baseline_v0.py`：不改代码，但 headline 指标解读方式改变（见第 4 节）。

## 4. 训练与评估协议

- **Contract 依赖**：与 RG 用同一 contract（`v1_expanded_pool.yaml`，`fit_endpoint_baseline_stats=true`），保证和 RG 直接可比（同一 eval_all 样本集、同一 `fault_baseline_train_fraction`）。不与 L0/L1/L2 在 `v1.yaml` 上的历史数字混比——两者 eval_all 行数和类别比例不同，横向比较需注明口径差异（CLAUDE.md Known Gotchas 已有此约定）。
- **Headline 指标**：用 `eval_baseline_v0.py::compute_stratified_metrics` 已有的 `by_anomaly_type`（`eval_baseline_v0.py:52-58`）宏平均，不用 `overall`（`eval_baseline_v0.py:50`）。
  - **原因存档**：`overall` 把所有 case 的所有行揦成一个二分类问题，Normal case（`normal_v2`，采集于 07-11）与故障 case（`new_ep1`/历史 anomod_v1/endpoint_raw2，不同天/不同 run）混合作正负样本对比，检测器可以靠"这是哪个 run"的绝对噪声水平分开样本，不需要学会真实故障语义，AUROC 因此虚高（entry 017 实测：ABORT 0.822→0.922，DELAY 0.437→0.696，REPLACE 0.469→0.717，跨 run 虚高 0.1~0.25）。entry 012/014 报的 headline 数字用的正是这个被污染的 `overall`。
  - `by_anomaly_type` 天然是 within-case 比较（每个 anomaly_type 恰好对应 1 个 case，组内正负样本——inject vs baseline/recover——都来自同一个 case 自己），不受跨 run 污染，是当前代码里已经存在、但历史上没被当 headline 用的干净指标。
  - **本设计不修复切分逻辑本身**（即不改 `src/contracts/` 让 eval 阶段负样本来源从"跨天 Normal case"换成"故障 case 自己的 baseline/recover"）。这是报告侧最小修复，代码侧的切分逻辑改造留作独立工作项，不纳入本设计范围。
- **对比对象**：
  - L0（`EarlyConcatFusion`）：结构对照，唯一变量差异是"是否逐特征加权"。
  - RG（`ReliabilityGatedFusion`，`softmax` 归一化，当前最优 baseline）：同 contract 直接对比。
  - oracle（单特征 `client_error_rate` 的 AUROC）：上界参考，不是要打败的对象。

## 5. 成功判定标准

以 ABORT/REPLACE 两个 anomaly_type 的 `by_anomaly_type` AUROC 为准（DELAY/PATCH 信号本弱，entry 017 已确认，不作为达标判据，只看是否被明显拖累）：

- **达标**：AUROC 从现有 0.3~0.5 拉到 **≥0.8**，且 DELAY/PATCH 无明显退步（允许持平/小幅波动）→ 说明逐特征加权方向对，值得深化（下一步候选：可学习阈值、逐维可学习权重网络，或改造 SVDD 打分公式）。
- **部分达标**：有改善但离 oracle 0.99 明显有距离（如 0.4→0.6）→ 需诊断卡在哪一层（融合层还是打分层），类比 `scripts/analyze_gate_weights.py` 的思路做逐特征权重分布分析。
- **不达标**：与 L0/RG 基本无差异甚至更差 → 不做后续深化任务（不引入可学习阈值，不改 SVDD 打分公式），回头重新审视机制假设本身。

## 6. 测试

遵循项目 `tests/`约定 + CLAUDE.md「`src/fusion/` 必须有单元测试」要求。测试文件：`tests/test_fusion_deviation_weighted.py`，风格对齐现有 RG 测试（`tests/test_fusion_reliability_gate.py`，若存在则直接参考其 fixture 构造方式）。

- **加权公式数值正确性**：构造已知 `mean`/`std`/`feature` 的小样本，验证 `z` 和 `sigmoid(|z|-2.0)` 计算结果与手算一致。
- **退化列权重固定为 1**：构造一个命中 `degenerate_columns` 的列，验证该列输出等于输入（未被调制），不受该列人为兜底 `std` 影响。
- **`endpoint_id=None` 兜底**：验证退化为全 1 权重，输出等价于纯 concat（等价 L0 行为）。
- **`output_dim` 正确性**：等于 `ep_dim + svc_dim`。
- **`modality_dims` 校验**：复用现有 L0/L1/L2 的 `MODALITY_ORDER` 缺失/多余 key 报错逻辑。

## 7. 已识别风险

1. **验证结果可能不达标**：这是本设计明确接受的可能性——用户已确认"如果第一步效果就不达标，后续深化任务不需要做"，不达标不代表实现有 bug，可能代表"逐特征加权"这一层机制本身不足以对抗 SVDD 打分公式的等权稀释效应。
2. **困难样本无解**：DELAY/PATCH 信号弱/不存在于 RED 特征，逐特征加权不能创造不存在的信号，如实报告不掩盖（沿用 RG spec 已有的诚实边界原则）。
3. **评估口径与历史数字不可直接横向比较**：`by_anomaly_type`/`v1_expanded_pool` 口径与 entry 012/014 的 `overall`/`v1.yaml` 口径不同，论文/报告中需明确注明,避免读者误以为是同一基准下的直接提升。

## 8. 明确排除项（YAGNI，本轮不做）

- 不做分支级门控（RG 式 softmax 竞争）——已诊断的坍缩风险，且会混淆单变量归因。
- 不做可学习阈值/可学习加权网络——记入条件触发的未来工作（仅当本方案验证达标才考虑）。
- 不改造 Deep SVDD 打分公式——同上，仅当本方案验证不达标才考虑。
- 不修复跨 run 污染的切分逻辑代码——本轮用 `by_anomaly_type` 作报告侧最小修复即可满足本设计的评估需求，代码侧改造留作独立工作项。
- 不新增 `service_metric`/`service_log` 第三分支——逐特征加权已经覆盖 `svc` 分支内部的淹没问题，不需要拆分支。
