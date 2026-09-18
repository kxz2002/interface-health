# Contract v2：per-case 自适应归一化 + DWF 自参照 + RG 退役 + 平凡基线参照线

- **日期**: 2026-09-18
- **对应 history entry**: 027（P0 系列）
- **状态**: 设计已确认，待实施
- **基线 commit**: `56c4561`（分支 `exp/new-merge-rerun-pr24-pr25`）

## 1. 问题

entry 027 的诊断给出三个实测事实（`new_merge`，per-case 宏平均 AUROC）：

| 打分方式 | per-case macro AUROC | 是否需要训练 |
|---|---|---|
| `rel_pos`（只看窗口在 case 内的相对时间位置，不看任何特征） | 0.9656 | 否 |
| per-case z-score `L2‖z‖`（用每个 (case, endpoint) 自己 baseline 阶段的 mean/std） | 0.9415 | 否 |
| `max|z|` | 0.9403 | 否 |
| **最好的学习模型 DWF** | **0.9238** | 是 |

两个零参数基线打败全部六种融合机制。归因已完成且明确：

- 模型输入的 21 维特征**不含任何时间戳字段**，从特征回归 `rel_pos` 的 GroupKFold R² 仅 0.1026 —— 模型在结构上无法利用时间位置，**平凡基线赢是数据集与评估协议的缺陷，不是模型作弊**。
- `Normalizer` 与 `EndpointBaselineStats` 的统计量全部来自唯一一个 Normal case（478 行），而它位于 17 小时采集窗口的第 0 位。用第一小时的统计量去衡量 17 小时后采集的 case，**系统漂移被当作异常偏离量**计入特征。

结论：当前主 pipeline 的参照系是坏的。在修好它之前，任何新融合机制的 AUROC 排名都不可信 —— 无法分辨 +0.02 是模型进步还是又一次泄漏/漂移。**本轮迭代的目标是修参照系，不是提出新模型。**

## 2. 关键决策

### D1：归一化改为 per-case z-score（不是 min-max）

最初倾向 min-max，理由是"可以不动 contract 的 `[0,1]` 校验"。该理由在两处被推翻：

1. `build_contract.py:529` 的 `clip(0,1)` 会把所有越界值压成恰好 `1.0`，使 DWF 的 excess 恒为 0 —— DWF 对 rate 列直接失明，而 signal drowning 的头号受害者恰恰是 rate 列（`client_error_rate`/`client_5xx_rate`，oracle AUROC ≈0.99）。所以 per-case 口径下 rate clip 无论如何必须退役，"不动校验"的前提不存在。
2. 实测 fit 子集规模（`contract_new_merge_expanded`）：per-(case, endpoint) baseline 共 208 组，窗口数 min 5 / median 81 / max 82；取前 20% 作 fit 后 **median 仅 16 行，min 1 行，26/208（12.5%）不足 5 行**。min-max 只用 2 个极值序统计量，n=16 时单个离群窗口即可扭曲 scale；且 n=1 时 `hi-lo=0` 触发 entry 013 的退化跳过 → 原始量纲透传，而 per-case 下原始值跨 case 不可比，会直接主导 SVDD 距离。

第三个理由是科学上的：打败我们的基线本身就是 per-case mean/std。**用同一种参照系才能 apples-to-apples 回答"SVDD 比一个 L2 范数强在哪"**；换成 min-max，输赢都说不清是参照系差异还是模型差异。

### D2：泄漏红线（不可妥协）

**不变量：任何参照统计量的 fit 集合必须 ⊆ 训练池行。**

| case 类型 | fit 源 |
|---|---|
| Normal case | `train_fit`（前 60%，现状不变） |
| 故障 case | 被吸入训练池的 baseline 前 20% 窗口（即 `source_phase == 'fault_baseline'` 对应的切分结果） |

eval 侧留作负样本的 80% baseline 行**不参与**任何统计量计算。平凡基线参照线同样遵守此约束（**lean 版**，作主口径）；entry 027 诊断脚本里的 transductive 版（用全 baseline）留在一次性脚本中作附注，不进常驻 stage —— 否则等于拿一个偷看过 eval 的参照来审判模型。

### D3：升 contract v2，不在 v1 加开关

per-case 口径破坏三处字段口径：rate 列不再落在 `[0,1]`、clip 语义变化、`EndpointBaselineStats` sidecar 消失。按 entry 003 立下的接口墙规矩（破坏字段口径必须升版），这必须是 v2 而非 v1 的 config 开关，否则下游消费者无法从版本号判断语义。

v1 保持原样与 v2 **并列**作为消融维度，不替换 —— per-case 归一化会丢弃绝对水平信息，而资源型故障的信号恰在绝对水平上（见 §7 canary）。

### D4：DWF 退化为无状态纯函数

per-case z-score 之下特征值本身就是偏离量，excess 无需查任何外部表：

```
w = sigmoid((|x| - τ) / s)      # τ=2.0, s=1.0，进 config
out = w ⊙ x
```

这恰好是 entry 018 的原始公式 `sigmoid(|z| - 2.0)`，只是不再查 sidecar 而直读特征。**结论因此可以写成：DWF 的公式一开始就是对的，错的是参照系。** 它同时成为 `max|z|` 平凡基线（0.9403）的可微版本 —— SVDD 学表征、DWF 做软 top-k 偏离聚合，这正是 entry 027 P2 想要的"平凡基线 → 学习模型"的桥。

连带删除：stats 索引 buffer、`case_idx` 管线、`from_contract` 覆写钩子、列顺序强校验（entry 018 的脆弱耦合）、per-sample Python 循环查表。

### D5：RG 退役，不重定义

RG 与 DWF 的病不在同一层。DWF 的机制对准了已证实的问题（signal drowning），只是参照系错，修参照系能救。RG 的缺陷是结构性的：**one-class loss 下没有任何梯度信号把门控权重与检测质量绑定，门控学无可学**。两次独立塌陷（softmax 竞争放大坍缩 → `w_svc≈1.0`；independent_sigmoid 退化列 z-score 爆 1e9 → sigmoid 饱和冻结）是同一病根的两种表象，改参照系救不了"学无可学"。

处置：两个 RG stage 从 `dvc_new_merge/dvc.yaml` 移除，类代码留在 git 历史备查，两次塌陷根因写进 entry 作为负面结果。这也符合 entry 027 "不要做"清单中"停止增加融合机制变体"的精神 —— branch 级路由这一假设让位给已证实的 feature 级偏离聚合。

### D6：`EndpointBaselineStats` 整体退役

它实现的是"分组 mean/std + 向全局 shrinkage（k=10）"，正是 per-case z-score 退化回退所需的数学。但 per-case z-score 之后**它就是归一化本身**，不再是旁挂参照表 —— 所以 shrinkage 应住进 `Normalizer`，让后者成为归一化统计量的唯一事实源，同时消掉 sidecar 产物与 entry 015 收束的那批耦合债。代价是 ~15 行 shrinkage 逻辑在新家重写（有单测护航，旧实现留在 git 历史，旧测试可改写复用）。

## 3. PR 切分

| PR | 内容 | 依赖 |
|---|---|---|
| PR-1 | 落地 `exp/new-merge-rerun-pr24-pr25`（rebase + push + PR）：entry 027 定稿 + 补写 entry 028（PR #26 欠的 history entry）+ 本设计文档。零代码改动 | — |
| PR-2 | eval 参照线：per-case macro 升主指标 + `score_trivial_baselines.py` + dvc stage。**v1 口径立即生效**，现有数字马上有参照 | PR-1 |
| PR-3 | `fault_inject_nontarget_train_fraction` 退回 0.0 的对照实验（v1 口径，L0/DWF × 4 seed） | PR-2 |
| PR-4 | 主线：contract v2 + DWF 自参照 + RG 退役 + 四臂 × 4 seed 重跑 | PR-2 |

分支现状核实：`56c4561` 领先 master 1 commit、落后 1 commit（#26），**与 #26 的改动零文件交集**，rebase 无冲突。

PR-3 的 comparability 陷阱：fraction 1.0→0.0 会让 5380 行从训练池回到 eval_all，**两个口径的 eval 集不同**（CLAUDE.md 已记录的坑）。对照必须在**两版 eval_all 的共有 sample_id 子集**上算 AUROC，以隔离训练效应。

## 4. Contract v2 规格

### 4.1 Normalizer

新增 scope：`per_case_endpoint`（group 键 `(case_id, endpoint_key)`）、`per_case_service`（`(case_id, service_name)`）。method 从 `min_max` 扩到 `z_score`。

退化回退为层级 shrinkage：

```
per-(case, endpoint)  →  该 endpoint 跨 case 汇总  →  全局
                      ↑ n < k 或 std ≈ 0 时按 k=10 收缩
```

这替代 entry 013 的"跳过归一化保留原值"用于 per-case scope —— 后者在 per-case 下不可接受（原始值跨 case 不可比）。global/per_endpoint/per_service 三个既有 scope 的行为**逐行不变**。

记录性事实：当前数据集上 `per_case_endpoint` 与 `per_case_service` 恰好同为 208 组（entry 017 记录的 inner join 把每个 service 塌成 1 个 endpoint）。两个 scope 都实现（未来数据集可能不同），但 entry 中须写明本数据集上二者等价，**不围绕该区分设计消融实验**。

### 4.2 build_contract 重排

`split_fault_phase_temporal` 的调用**提到归一化之前**，得出每 case 的 baseline-train 行 mask → Normalizer 只在这些行 + Normal `train_fit` 上 fit → `_write_v1` 复用同一次切分结果（单一事实源，杜绝两处参数漂移）。

### 4.3 校验与 clip 变更

- rate 列 `[0,1]` 校验退役，替换为量级 sanity（`|x| ≤ 1e3`，防 entry 013 的 1e9 爆值重演）
- rate `clip(0,1)`（`:529`）与 content_length `±5` clip（`:544-551`）在 v2 分支退役 —— 后者治的是跨批次参照系错位的症状，per-case 消除病因后 clip 反而可能误伤
- **保留不变**：sample_id 唯一、`phase ≡ is_anomaly`、`label_granularity` 枚举、`is_endpoint_anomaly ⇒ is_anomaly`、行守恒
- `fit_endpoint_baseline_stats = false`，不产 sidecar

### 4.4 行等价保证（最强回归断言）

v2 的 split / 标签 / 吸收闸门逻辑与 v1 **逐行一致**（fractions 沿用 0.2 / 1.0 / 1.0），因此 **v1 与 v2 的 `eval_all` 行集与全部标签列必须逐行相等**。这条断言把"归一化改动"与"划分改动"彻底隔离，使 v1↔v2 的 AUROC 差异只可能来自特征值本身。

### 4.5 退化审计

group 数从 ~10 涨到 ~416（208 endpoint 组 + 208 service 组）。构建时产出每列的 shrinkage 回退占比报告；回退率 >50% 的列必须在 entry 中解释。

### 4.6 产物

`configs/contract/v2_new_merge.yaml` + `artifacts/contract_new_merge_v2/`（目录隔离），`_write_v1` 的落盘逻辑复用。

## 5. 平凡基线与 eval 升级

`scripts/score_trivial_baselines.py`：读 contract_dir，产出 3 个契约合规的 scores.parquet —— `rel_pos`、`zscore_l2`、`zscore_max`。z-score 统计量只来自训练池行（lean 版，守 D2）。走现有接口墙，由 `eval_baseline_v0.py` 原样消费，**eval 脚本不需要读 contract**（接口墙不打洞）。注册为 dvc stage，v1/v2 两个口径都带。

`eval_baseline_v0.py`：新增 `per_case_auroc_macro` / `per_case_auprc_macro`（仅对同时含正负类的 case 宏平均）升为主指标，pooled 降为附注。`metrics_v0` **不需要改** —— 契约显式允许任意额外字段（`src/contracts/metrics_v0.py:19-22`）。

该脚本自 PR #4 引入后只被改过一次（PR #8），#24 的标签三档化大修完全没有触及它 —— 这正是接口墙的设计意图，也是此处改动风险低的原因。

## 6. 测试策略

- `Normalizer` per-case scope 全套单测：fit / transform / shrinkage 三级回退 / 未知 group 透传 / save-load roundtrip
- **泄漏断言**：mock split，断言实际传入 `fit()` 的行集 ≡ baseline-train 行集
- **v1↔v2 行等价**集成断言（sample_id 多重集 + 标签列逐行相等）
- DWF：权重公式单测、无 sidecar 构造、梯度通畅（无冻结参数 —— entry 012 optimizer 漏接 fusion 参数的教训）
- 平凡基线：toy 数据正确性 + **z-score 只用允许行**（泄漏测试）+ scores 契约合规
- 回归门：现有测试全绿 + **v1 口径 repro 数字逐位不变**（证明 v1 代码路径零改动）

## 7. 验收标准

主表：v2 四臂（L0 / L1 / L2 / DWF）× 4 seed（mean±std）vs v1 同臂 vs 3 条平凡基线，**per-case macro 为主指标**。

Canary：`Lv_P_CPU_preserve` 的分层 AUROC ≥ 0.9（当前组合口径实测 **0.9995**，L0/seed42，`artifacts/baseline_new_merge_concat/metrics.json`）。CPU 注入相对自身 baseline 仍是巨幅偏离，理论上信号应保留；若显著下滑，说明参照系被污染或实现有误。

**三条必须写入 entry 的预期，避免数字被误读：**

1. **`rel_pos` 基线仍会是 ~0.9656。** per-case 归一化治的是漂移与参照系，对"inject 起点固定在 `rel_pos` 0.586±0.015"零作用 —— 那只能靠重采（entry 027 P1）。
2. **最可能的结果是 SVDD ≈ z-score 基线。** 同参照系下两者输入信息量基本一致。这是 entry 027 P2 已预演的合法科学结论。
3. **本轮的成功定义是"拿到参照线齐全、泄漏可证的可信排名"**，不是"打赢基线"。若 P0 做完 SVDD 仍输给 lean z-score，按 P2 预案诚实转向"benchmark 缺陷 + 规范协议"的论文形态。

## 8. 非目标（本轮不做）

- 重采数据集（entry 027 P1）—— AnoMod 正在筹划投稿，不便重采
- 新增任何融合机制变体（entry 027 "不要做"清单）
- DANN / gradient reversal 消除 case 身份 —— entry 027 已证伪 case 身份是 shortcut
- 修 RG independent_sigmoid 的塌陷 —— 直接退役
- 通用极端值裁剪框架（entry 026 已定案不做）
