# 031 · contract v2：per-case z-score 参照系修复达成，四臂全改善但 SVDD 仍未跑赢 lean z-score（D1–D6 全落地，contract v2 迭代总收尾）

- **日期**: 2026-09-19
- **PR**: 本 PR（contract v2 计划 PR-4，栈式 PR：base = PR #28 的分支 `feature/eval-percase-trivial-baselines`，#28 合并后自动 retarget master；与 PR #29 平级，merge 顺序无关，index.md 可能有一行追加冲突） · **Commit**: 见本 PR merge（代码主线 `69efa5f`→`996937f` 共 14 个 commit）
- **类型**: Feature + Experiment + Refactor
- **影响域**: `src/data/normalization.py`, `src/contracts/contract_config.py`, `src/contracts/contract_v0.py`, `scripts/build_contract.py`, `src/fusion/deviation_weighted_selfref.py`, `configs/contract/v2_new_merge.yaml`, `configs/fusion/deviation_weighted_selfref.yaml`, `dvc_new_merge/dvc.yaml`, `dvc_new_merge/dvc.lock`, `tests/`（4 个新测试文件）, `artifacts/contract_new_merge_v2/`, `artifacts/baseline_new_merge_v2_*/`, `artifacts/trivial_baseline_v2_*/`, `artifacts/contract_v2_acceptance/main_table.md`, 归一化参照系 / 平凡基线口径 / 多模态融合

## 做了什么

entry 027 P0 系列的主线收尾：把主 pipeline 的归一化参照系从"单一 Normal case（478 行、位于 17h 采集窗口第 0 位）全局 min-max"换成 **per-case z-score + 三级层级 shrinkage**，以 contract **v2** 与 v1 并列落地（不替换、不加 config 开关，v1 代码路径冻结做消融维度）。规格见 `docs/plans/2026-09-18-contract-v2-percase-design.md`（D1–D6），18 任务拆分见同目录 implementation 计划。

六件事：① `Normalizer` 新增 `per_case_endpoint` / `per_case_service` 两个 scope 与 `z_score` method，退化走"组 → 跨 case 汇总 → 全局"shrinkage（k=10），全局兜底 1.0 哨兵；② `build_contract.py` v2 分支把时序切分提到归一化之前，fit 行集严格等于 `train_fit ∪ 各故障 case baseline 前 20%`；③ rate 列 `[0,1]` 校验与两处 clip 在 v2 退役，换全特征列量级 sanity；④ DWF 以**新类** `SelfReferentialDeviationFusion` 落地自参照版（公式不动、参照系修正），旧类一行不改；⑤ RG 两个 stage 从 pipeline 移除（类代码留 git 历史），`EndpointBaselineStats` sidecar 整体退役、shrinkage 数学收进 Normalizer；⑥ v2 四臂（concat / independent_concat / gated / selfref）× 4 seed + v2 三条平凡基线全部跑通，主表见 `artifacts/contract_v2_acceptance/main_table.md`。

## D1–D6 六项决策：依据与实际结果

### D1：per-case z-score（不是 min-max）——四臂全部改善

否定 min-max 的实测依据：fit 子集（per-(case,endpoint) baseline 共 208 组，取前 20%）窗口数 **median 仅 16 行、min 1 行、26/208 组（12.5%）不足 5 行**。min-max 只用 2 个极值序统计量，n=16 时单个离群窗即可扭曲 scale，n=1 时零区间触发退化跳过、原始量纲透传，而 per-case 下原始值跨 case 不可比。另两条理由：rate clip（`clip(0,1)`）在 per-case 口径无论如何必须退役（DWF 对 rate 列会失明，而 signal drowning 头号受害者恰是 rate 列），"不动 `[0,1]` 校验"的前提不存在；打败我们的基线本身就是 per-case mean/std，同参照系才能 apples-to-apples 回答"SVDD 比 L2 范数强在哪"。

**实际结果**：v2 四臂 per-case macro AUROC 相对 v1 同臂**全部改善且方差全部收窄**（见主表），D1 方向被证实。

### D2：泄漏红线——fit = train_fit ∪ baseline 前 20%，6 个承载测试 + 变异双向打红

不变量：任何参照统计量的 fit 集合必须**恰好**是「Normal 的 `train_fit` ∪ 各故障 case 被吸入训练池的 baseline 前 20% 窗口」；eval 侧剩余 80% baseline 负样本、Normal holdout、inject/recover 行（含被吸收的非目标行）一律不进 fit。实现上把 `split_fault_phase_temporal` 提到归一化之前，切分结果单一事实源传给归一化与 `_write_v1`，杜绝两处独立切分的参数漂移。

证据是 `tests/test_build_contract_v2_fit_scope.py` 的 **6 个 in-process 承载测试**（monkeypatch 截获 `Normalizer.fit` 实参做行集断言；fixture 单 endpoint、每 case 10 个 baseline 窗，预期 fit 行按窗序号手算）。commit `fa3377e` 补测试前做过变异验证：故意把 fit 范围扩大/缩小时红、扩大到 inject/recover 时红——双向变异都打红后才认定测试有判别力（该 commit 同时发现"落盘尺度"与"inject/recover 不进 fit"两处此前零覆盖）。平凡基线同样守此红线（lean 口径，fit 只读 `train.parquet`）。

### D3：升 v2 不加开关——行等价红线 + 反向断言防 vacuous

per-case 口径破坏三处字段口径（rate 不再落 `[0,1]`、clip 语义变、sidecar 消失），按 entry 003 的接口墙规矩必须升版，v1 与 v2 并列做消融。最强回归断言是**行等价**：v2 的 split / 标签 / 吸收闸门逐行复用 v1 逻辑（fractions 沿用 0.2/1.0/1.0），实测 **train 9513 行、eval_all 14186 行、正样本 916 个**与 v1 完全一致，8 个标签/切分列逐行相等（`tests/test_contract_v1_v2_row_equivalence.py`，4 条断言覆盖 eval_all、train、Normal 三切分帧、sidecar 缺席）。

为防"v2 什么都没改、行等价平凡成立"的 vacuous 通过，另有**反向断言**：v2 特征值必须真的变——实测 17 个信息量特征列中 14 列实质变化、全体 **67.1% 的特征单元改变**。v1↔v2 的 AUROC 差异因此被隔离为只能来自特征值本身。三条 v1 红线：PR-4 的 14 个 commit 不触碰任何 v1 代码路径与 v1 artifacts（v1 scores/metrics 的 sha256 不变、`git diff` 为空）。

### D4：DWF 自参照新类（不是原地重写）——公式不变，参照系修正

实施计划相对设计文档的唯一结构偏离在此：设计 D4 写"原地重写为无状态纯函数"，实现改为新建 `SelfReferentialDeviationFusion`、旧 `DeviationWeightedFusion` 一行不改。理由：原地重写会改变 v1 DWF arm 的数字，回归门失效、v1↔v2 不再单变量。新类零参数零状态，`w = sigmoid((|x|−τ)/s)`（τ=2.0、s=1.0，进 config），直读特征——v2 的 per-case z-score 已使特征值本身就是"相对自身 baseline 的偏离"。**结论：DWF 的公式一开始就是对的，错的是参照系。** 连带消除 entry 015 的 from_contract 钩子、entry 018 的列顺序强校验与 per-sample Python 循环查表。

**实际结果**：selfref **0.9495 ± 0.0033** vs v1 DWF 0.8932 ± 0.0180（+0.0563，std 收窄 5.5 倍），是四臂中方差最小的。

### D5：RG 退役——one-class loss 下门控学无可学，两次塌陷归档为负面结果

RG 与 DWF 的病不在同一层：DWF 机制对准了已证实的问题（signal drowning）、只是参照系错；RG 的缺陷是结构性的——one-class loss 下没有任何梯度信号把门控权重与检测质量绑定。两次独立塌陷是同一病根的两种表象，归档如下：

1. **softmax 竞争放大坍缩**：两分支初始偏差尺度仅 ~1.2× 不对称，softmax 强制竞争归一化在无监督 SVDD loss 下形成正反馈，`w_svc` 单调爬升到 ≈1.0（entry 014 逐 epoch 追踪，非鞍点）；训练池扩容会显著加速放大。
2. **independent_sigmoid 退化列爆值冻结**：去掉竞争归一化后，退化列 z-score 爆到 1e9 量级（entry 013 同根因），sigmoid 输入 1e8 直接饱和冻结（entry 027 实测完全塌陷）。

entry 025 修复标签后 RG 还是六机制里唯一退化、且方差远超稳定机制（softmax std 0.1352，约 L0 的 3.8 倍）；entry 014 的教训"单 seed 连方向都会判错"在这条机制上完整应验过。修参照系救不了"学无可学"，故两个 RG stage 从 `dvc_new_merge/dvc.yaml` 移除，类代码留 git 历史备查，不做第三种归一化的重试。branch 级路由让位给已证实有效的 feature 级偏离聚合。

### D6：EndpointBaselineStats 退役——shrinkage 数学住进 Normalizer

EBS 实现的"分组 mean/std + 向全局 shrinkage（k=10）"正是 per-case z-score 退化回退所需的数学；per-case z-score 之后它就是归一化本身，不该再是旁挂参照表。处置：shrinkage 搬进 `Normalizer`（k=10 直接复用 EBS 的校准值，不重新调参），Normalizer 成为归一化统计量的唯一事实源；v2 config `fit_endpoint_baseline_stats=false`，sidecar 产物、entry 015 收束的那批耦合债一并消失。旧 EBS 代码保留（v1 arm 仍依赖）。

## 主表（per-case macro AUROC，4 seed mean±std，ddof=0）

| 方法 | v1 | v2 | v2−v1 | v2 pooled（附注） |
|---|---|---|---|---|
| concat (L0) | 0.8964 ± 0.0110 | **0.9608 ± 0.0040** | +0.0644 | 0.9598 ± 0.0047 |
| independent_concat (L1) | 0.8566 ± 0.0198 | 0.9513 ± 0.0097 | +0.0947 | 0.9517 ± 0.0101 |
| gated (L2) | 0.8995 ± 0.0102 | 0.9350 ± 0.0061 | +0.0355 | 0.9343 ± 0.0060 |
| deviation_weighted_selfref | 0.8932 ± 0.0180（v1 DWF） | 0.9495 ± 0.0033 | +0.0563 | 0.9514 ± 0.0038 |
| 平凡 rel_pos | 0.9656 | 0.9656 | 0 | 0.9303 |
| 平凡 zscore_l2 | 0.9539 | **0.9853** | +0.0314 | 0.9630 |
| 平凡 zscore_max | 0.9441 | 0.9810 | +0.0369 | 0.9548 |

四臂一致改善（+0.036～+0.095）、std 全部收窄（最大 std v1 0.0198 → v2 0.0097）。但**没有一个臂打赢平凡基线**：最强 concat 0.9608 落后 v2 zscore_l2 0.9853（−0.0245）、zscore_max 0.9810（−0.0202），也略低于不消费归一化特征的 rel_pos 0.9656（−0.0048）；四臂均值 0.9492 落后两条 zscore 基线 0.03 以上。数据集 27 case 中 23 个同时含正负类（单类 case 不进宏平均）。

macro AUPRC 口径排名不同：concat 0.8647 ± 0.0105（v1 0.7194）、independent 0.8543 ± 0.0203、selfref 0.8413 ± 0.0126 **反超全部平凡基线**（rel_pos 0.6008、zscore_l2 0.8223、zscore_max 0.8168），gated 0.7846 ± 0.0266 仍略低于两条 zscore 基线。宏平均与 pooled 排名差异源于平凡基线在少数 case 上的极端输出。逐 seed 明细、AUPRC 全表与训练健康度附注见 `main_table.md`。

## 三条预注册预期的落地（设计文档 §7，逐条对照）

1. **rel_pos 仍 ~0.9656 —— 成立**：v2 实测 0.9656，与 v1 逐位一致（它不消费归一化特征，行等价的旁证）。per-case 归一化治漂移，治不了"inject 起点固定在 rel_pos 0.586±0.015"；该问题只能靠 P1 重采（随机化 inject 起始位置 / Normal 交错采集），AnoMod 投稿期继续延期。
2. **"SVDD ≈ z-score 基线"——方向比预期更不利于 SVDD，且结论随指标口径分叉**：AUROC 主指标下不是"持平"而是**全臂输给 lean z-score**（最强 concat 0.9608 vs zscore_l2 0.9853，差 0.0245，四臂无一例外）；但 **macro AUPRC 口径下 concat 0.8647 反超** zscore_l2 0.8223、rel_pos 仅 0.6008。这是 entry 027 P2 预演的合法科学结论：**同参照系下 Deep SVDD + 现有融合没有挣到它的复杂度**——z-score 范数直接聚合分散在多特征上的弱信号，压进低维球心距离反而损失信息。按 P2 预案，论文形态转向"benchmark 缺陷 + 规范评估协议"的论据进一步增强：v1 transductive（entry 027）、v1 lean（entry 029）、v2（本 entry）三轮、两套归一化参照系下平凡基线获胜的证据链已齐。
3. **本轮成功定义——达成**：参照线齐全（3 平凡基线 × 2 口径常驻 stage）、泄漏可证（6 承载测试 + 双向变异验证）、v1↔v2 单变量（行等价 + 67.1% 特征单元反向断言 + v1 sha256 不变）。这份排名可信，包括"学习模型排第三"这件事本身。

## Canary：`Lv_P_CPU_preserve` 分层 AUROC（阈值 ≥0.9）

四臂 × 4 seed = **16 个 cell 全部 ≥ 0.9**：concat 0.9969 ± 0.0003、independent 0.9949 ± 0.0024、selfref 0.9946 ± 0.0027、gated 0.9313 ± 0.0223。v1 concat 0.9996 ± 0.0002（seed42=0.9995）未掉头向下，参照系未被污染——资源型故障的绝对水平信号经层级 shrinkage 保留（per-case 丢弃绝对水平信息的设计风险没有在 canary 上兑现）。**唯一边际项：gated seed2 = 0.9026，压线通过，引用该臂 canary 时必须保留这个不确定性**，与 gated 臂整体垫底一致。rel_pos 在该 case 仅 0.8895（排名特征对资源型故障不敏感），非训练臂、仅记录不作失败项。

## 退化审计（Task 14）

- **规模**：`per_case_endpoint` 与 `per_case_service` 在本数据集上**同为 216 组/scope**（27 case × 8；inner join 把每个 service 塌成 1 个 endpoint，entry 017），合计 432 组。设计文档预估 ~416 是按 26 case、208 组/scope 估的，偏差仅来自计数基数。两个 scope 都实现（未来数据集 endpoint↔service 可能不 1:1），但**本数据集上二者等价，不围绕该区分设计任何消融**。
- **std 回退率 >50% 的列共 11 个**：全部是错误率/计数类近常数或稀疏特征（如 `*_5xx_rate`、窗口内事件个位列）。正常工况下这些列在 fit 段本就恒为 0/单值，回退到跨 case 汇总或全局 std 是三级回退的设计内行为，不是 bug。
- **`trace_5xx_rate` 有 198 个组 n=0**：上游采集稀疏——该特征只在 preserve endpoint 上有观测，其余组 fit 段无任何行，走未知/空组回退链，非归一化错误。
- **n<10 小样本收缩率无列超过 50%**（最高 44.9%），shrinkage 主要被零方差而非小样本触发，说明 median 16 行的 fit 子集规模在大多数列上够用，D1 对 min-max 的担忧（小 n + 极值统计量）不构成 z-score 的实际损害。

## 执行期的计划外决策（均有 commit 与裁决记录）

1. **量级 sanity 阈值 1e3 → 1e6（用户决策，commit `3d0e031`）**：按设计 4.3 实现的 `|x| ≤ 1e3` 在 v2 构建时确定性失败——`Lv_E_HTTPDELAY_assurance` 目标 endpoint 的 **44 行**真实信号 z≈1366–1995（注入 3s 延迟，实测 3004–3323ms vs 极稳定 baseline，std_eff≈2.2，**SNR 约 1500:1**），z-score 数学完全正确，且 44 行全部在 eval 侧、`is_anomaly=True`（train 侧 0 行，无泄漏）。1e3 的自述用途是兜 entry 013 的 1e9 退化除零爆值，z≈2000 距 1e9 差六个数量级——字面阈值与意图冲突。1e6 对最大真实信号留 500× 余量、仍比 1e9 低三个数量级，爆炸必被拦。事后抽查全特征 |z|：p50=0.09、p99=3.73、max=1994.6，佐证 1e6 之上的区间确实空。
2. **std 退化不加权、mean 恒收缩（Task 11，有意偏离 EBS）**：std 侧沿用 EBS 哲学——未定义的量不参与加权，零方差 std 直接取回退层/1.0 哨兵，不做 n 加权；mean 一律按 `w=n/(n+k)` 向回退层收缩。偏离理由：小 n 时 mean 噪声大、收缩是 shrinkage 的本职；大 n 时常数列的 (1−w) 偏移有界（w→1），且保留了绝对水平的残余信号（canary 依赖的就是这部分）。
3. **对称守卫 + 全列 sanity + v0/v1 配对禁止（Task 13，commit `04e301f`/`49660ca`）**：`v0`/`v1` contract 配 per-case z-score 在加载期即显式拒绝（非法 scope×method 组合提前报错）。裁决前实跑验证过该组合的真实失败形态：它抛的是**裸 `KeyError: 'case_id'`**（旧归一化路径不产出 case 分组键），review 阶段"会静默跑通、被 clip 截断成隐性错误"的判断经实验证伪——但裸 KeyError 对用户不可读，守卫的价值是把失败变成带正确信息的显式报错，裁决记录在 commit。量级 sanity 从 rate 列扩到**全部特征列**：entry 013 实际爆值的 latency 列并不是 rate 列，只守 rate 列会漏。
4. **v2 构建第三次触发 entry 029 的既有数据漂移**：裸 `dvc repro` 又被 2026-08-24 起未 `dvc commit` 的 189 个 `_pipeline_out` 漂移文件拖去重跑 build，并删掉 v1 产物。仍按既定方法恢复（锁定对象从本地 cache 还原，md5 与 dvc.lock 逐位一致，未提交任何数据 hash 变更）。这是同一漂移**第三次**产生恢复成本（entry 029、本 entry 两次），处置选项（`dvc commit` 承认上游重算 vs 备份恢复）不能再挂着，紧迫度升级，见遗留 TODO。

## 坑 / 已知问题（健康度观察）

- **gated 臂的门近乎常数 ~0.51**：seed42 checkpoint 在 eval_all 上复算 16 维 sigmoid 门，无 0/1 饱和（<0.05 与 >0.95 各 ≤1.7%），但 p5=0.478、p95=0.573，16 维门均值跨度仅 0.098、行间均值 std=0.0096——调制作用很弱，与该臂四臂垫底（0.9350）一致。形态不同于 RG 式 softmax 塌缩（不归一、不竞争、不饱和），但同样没学出有效选择性，是**branch/feature 级路由假设的又一不利证据**；是否随 RG 一并退役留待下轮决策。
- **selfref sigmoid 健康**：固定 τ/s 不可学习，正样本行门均值 0.315、开启（>0.95）率 5.9%，负样本行 0.163、开启率 0.2%——有选择性、无病理饱和（w>0.95 仅 0.6%），过渡带（|x| 在 τ 附近）占 16.4%，与输入 |x| p99=3.73 的分布自洽。
- **v2 平凡 zscore 基线（0.9853）比 v1 lean（0.9539）更高，且高过所有学习模型**：v2 特征已 per-case 标准化，平凡基线在 v2 train 上再按 `(case,endpoint)` 做一次 z-score，相当于方差稳定化的双重校正。现象记录在案，**机制未分解**（两次标准化可能是纯益、也可能在训练池小样本上引入新偏差），列入遗留 TODO；在分解完成前，不应把"v2 zscore_l2 0.9853"直接当作数据集可分性上限引用。
- 12 次训练（4 臂 × 4 seed，seed42 走 dvc）全部 50 epoch 正常收敛，scores.parquet 均 14186 行、0 NaN/inf；loss 从 0.13～0.36 降到 7e-5（gated）/ 0.001～0.012（其余）量级。
- v1/v2 行等价断言依赖两份产物同时在磁盘上，产物缺失时测试 **skip** 而非 fail——在干净 clone 上该红线不生效，需先按 dvc stage 构建两份 contract；引用"行等价已验证"时隐含这个前提。

## 遗留 TODO

- **P1 重采（最高优先的数据集侧事项）**：rel_pos 0.9656 的时间混淆只能靠随机化 inject 起始位置 / Normal 交错采集消除，归一化侧已无可用手段。
- **P2 论文形态转向素材已齐**：v1 transductive（027）/ v1 lean（029）/ v2（031）三轮平凡基线获胜证据链 + AUROC 输 / macro AUPRC 赢的口径分叉，支撑"benchmark 缺陷 + 规范评估协议"立论；正式写作前可补的只有 v2 侧 n=4 的统计加固（bootstrap CI）。
- **数据漂移处置（紧迫度升级）**：189 个 `_pipeline_out` 漂移文件的 `dvc commit` vs 备份恢复裁决已三次产生恢复成本，须在下个任何人需要裸 repro 的工作之前决断。
- **v2 zscore 双重校正机制分解**：两次标准化的增益/偏差未拆开，分解前 0.9853 不读成上限。
- **gated 常数门处置**：下轮决策是否随 RG 一并退役 branch/feature 级可学习路由（当前证据：RG 两种归一化皆塌陷、gated 在 v2 退化为常数门且垫底）。
- **inject fraction 1.0→0.0 退回的独立决策**：entry 030 已给证据（5380 行吸收对 L0 无稳健净收益、对 DWF 净损害 +0.04 方向），v2 行等价基准现已立住；但现在改 fraction 的代价是**重建 v2 contract 并重跑全部 v2 stage**（行集合一变，行等价断言失效），故不搭车，按 entry 030 的析因对照预案独立推进，届时同步重算三条平凡基线。
