# 025 · service 级标签粒度补齐：三档 `label_granularity` + `label_target_observable` 吸收闸门

- **日期**: 2026-08-22
- **PR**: #24（工作分支 `feature/service-level-label-granularity`）· **Commit**: `b7dd22a`（不含本次占位符补全 commit）
- **类型**: Bugfix + Experiment
- **影响域**: `scripts/build_contract.py`（`_attach_label_columns` 三档路由 + `_write_v1` 吸收闸门与 eval_all 七段拼接）, `src/contracts/contract_v0.py`（新增必需列 + 两条校验）, `tests/test_build_contract_endpoint_labels.py`, `tests/test_contract_v1_train_pool.py`, `tests/test_contract_v0_schema.py`, `tests/fixtures/nontarget_split_mini/`, `artifacts/contract_new_merge_expanded/`, `artifacts/baseline_new_merge_*`, `data/new_merge`（恢复落盘）, `CLAUDE.md`

## 做了什么

落实 entry 024 的两条遗留 TODO。`label_granularity` 从二档（`"endpoint"` / `"case"`）扩到三档，中间补上 `"service"`：有 `target_endpoint` 走 endpoint 档，只有 `target_service` 走 service 档（正样本收窄为 `is_anomaly & (service_name == target_service)`），两者皆无才退到 case 档。同时新增一列 `label_target_observable`（per-case 常量布尔），标记"该 case 声明的注入目标是否落在 `endpoint_to_service.yaml` 的可观测范围内"，并把它作为 `_write_v1` inject 侧吸收的第二道闸门。

在 `new_merge`（27 case，当前主数据集）上重建 contract 并重跑 6 种融合机制（seed=42），service/performance 档的判别力大幅回升——`Lv_P_CPU_preserve` 从 0.5428 到 0.9993，整体 concat(L0) AUROC 0.7416→0.9021。entry 024 那次 0.534→0.906 的手工抽样验证在全量重跑下被证实，且幅度更大。

## 关键决策（不在 commit 里）

- **新增 `label_target_observable` 列，而不是复用"正样本数为 0"做隐式判据**：`Lv_D_*`（target=`tsdb-mysql`）这类 case 在 service 档下正样本恒为 0。若不显式标记，`_write_v1` 的 inject 吸收判据 `~is_endpoint_anomaly` 会对这些 case 的**整个 inject 阶段**成立（正样本 0 → 取反恒 True），在 `fault_inject_nontarget_train_fraction=1.0` 下把"数据库挂掉时 8 个 service 全受影响"的故障数据整段吸进 One-Class 训练池当正常数据。这是本次改动里最危险的一条连锁反应，且**不会报任何错**——训练照跑、指标照出，只是模型学到的"正常边界"被污染。用一个显式列把"评估诚实性"（AUROC 为 null）与"训练安全性"（不吸收）分开表达，比让两件事都隐式依赖 `n_pos == 0` 更难被后续改动破坏。
- **mysql/gateway 类 target 选择"放行 `n_pos=0` + warning"，不扩展 `endpoint_to_service.yaml`**（entry 024 遗留的未决项，本次定案）：扩映射表意味着给 `tsdb-mysql` 编造一个客户端 endpoint，而它本来就不对应任何客户端入口——那会造出假 endpoint 行，污染的是特征侧，比 `n_pos=0` 更坏。产出 null AUROC 与 entry 016/023 对 `Lv_S_KILLPOD_gateway` 的既有处理惯例一致（单一类别 → 数学上无定义 → 如实报 null）。
- **service 档的 recover 行不吸收进训练池**：`_write_v1` 的 recover 侧判据保持 entry 022 的原样（`is_target_endpoint == False` 且仅 `label_granularity == "endpoint"` 生效），没有为新的 service 档开口子。理由是 service 档的 `is_target_endpoint` 全为 `0.0`/`NaN`，无法区分目标/非目标（与 case 档同样的问题），硬用 `service_name != target_service` 筛虽然技术上可行，但 recover 阶段的分布本身未经验证（entry 014 起就一直保守对待），本次改动已经足够大，不叠加未验证的口径变化。
- **`_attach_label_columns` 用 `fillna(False)`，`_write_v1` 用 `fillna(True)`，方向相反且都正确**：前者在**判定正样本**（未知默认"不是目标"→ 宁可漏判正样本，不虚报）；后者在**筛非目标行以吸收**（未知默认"是目标"→ 宁可少吸收，不误吸收）。两处都朝"宁漏勿误"倾斜，但因为语义相反，`fillna` 的方向也必须相反。entry 022 曾在 `_write_v1` 处先写成 `fillna(False)` 后改正，本次没有重犯，但两处并列存在时极易被后来者"统一"成同一个值，故在两处都留了注释。
- **先重跑 multi-seed 再写文档，调整了原定顺序**：原计划先写文档后补 seed{1,2,3}。发现 `artifacts/baseline_new_merge_*_seed{1,2,3}/metrics.json`（18 个）是 git 追踪文件且都是旧口径产物，若不重跑就提交，仓库里会同时存在 6 个新口径 + 18 个旧口径数字，**且文件名完全看不出区别**。这不是"验证完整性"问题而是"避免留下静默失效文件"问题，优先级高于文档。

## 坑 / 已知问题

- **加吸收闸门会在 `_write_v1` 里凿出一个行守恒漏洞，必须同步把 eval_all 从六段扩到七段**：不可观测 case 的 inject 行既不进训练池（被新闸门挡住），又不是正样本（`is_endpoint_anomaly` 全 False），也不落在原有任何一个 eval_all 分段的条件里——`new_merge` 上是 2105 行会**静默消失**。这个漏洞是加闸门直接造成的，不是既有 bug；发现它靠的是 `_write_v1` 里原有的行守恒断言，而不是任何显式测试。已补第三段 `eval_inject_unobservable` 并纳入守恒检查，同时加 `test_unobservable_target_inject_rows_never_absorbed` 钉住。**任何未来给吸收侧加新判据的改动，都要同步检查"被新判据挡下的行有没有归宿"**。
- **`is_endpoint_anomaly` 必须是 `is_anomaly` 的子集，这个不变量之前没有任何地方校验**：三档判据都形如 `is_anomaly & <target 条件>`，蕴含关系恒成立，所以一直没出问题。但它是隐式的——若未来某档判据被改成不带 `is_anomaly` 的形式，baseline/recover 行会被标成正样本，评估协议（CLAUDE.md：正样本 = inject 阶段内的 endpoint × 时间窗）直接失真且不报错。本次在 `contract_v0.py` 补了显式校验（`test_positive_label_outside_inject_window_raises`）。同时补了 `label_granularity` 的枚举校验——拼错成 `"svc"` 之类不会抛异常，只会让那批 case 悄悄走进 `else` 分支，行为与预期相反且无提示。
- **旧测试 `test_fallback_case_is_endpoint_anomaly_matches_is_anomaly` 断言的正是被修掉的 bug**：它断言 `is_endpoint_anomaly` 严格等于 `is_anomaly`，即"case 级 fallback 对整个 case 一视同仁标正"。已删除并换成 `test_service_level_case_narrows_positives_to_target_service`。这类"锁住错误行为"的测试是修 bug 时的隐性阻力——看到测试失败的第一反应容易是"我改错了"，需要先判断测试本身是否在保护 bug。
- **`tests/fixtures/mini_dataset.yaml` 验证不了 service 级收窄，必须用 `nontarget_split_mini`**：前者的故障 case（`Lv_P_DISKIO_preserve`）CSV 里虽有 2 个 endpoint，但 `routeservice` 不在 v0 白名单内，过滤后只剩 1 个 endpoint，非目标 service 候选池恒为空——测试会以"fixture 前提变了"失败而非断言失败，容易误判为实现问题。改用 `nontarget_split_mini`，其 `Lv_D_CASELVL_travel` 有真实的 service 级 fan-out（travel 5 正样本 / travel2 0）。本次另给该 fixture 新增了 `Lv_D_UNOBSERVABLE_mysql`（target=`tsdb-mysql`）端到端覆盖不可观测路径。
- **`data/new_merge` 的 DVC cache 与 remote 都是空的，恢复后不要跑 `dvc checkout`**：本次从另一台设备 SCP 回一份备份（124GB，27 case）重命名为 `data/new_merge`，字节数与 `data/new_merge.dvc` 的 `size: 132155637198` 完全一致。但 cache 和 remote 都没有内容，此时 `dvc checkout` 会按 `.dvc` 记录去 cache 找、找不到就可能清掉工作区这 124GB。正确方向是 `dvc commit`（工作区 → cache），本次**故意没跑**（等用户决定）。`dvc status` 显示 "not in cache" 是预期状态，不影响实验——`build_contract.py` 直接读文件，不经 DVC。
- **备份与原始 `AnoMod/2026_0729` 的差异已核对：只多 3 列，标签完全一致**：备份是较新版本（07-31，与 entry 023 接入日期吻合），`tt_endpoint_health_*`/`tt_fused_*` 末尾多了 `content_length_mean`/`content_length_rel_shift`/`body_hash_mismatch_rate` 三列（共 4,799,137 字节）。27/27 case 的旧列逐字节一致、`tt_traces_red_*` md5 一致、全部 `case_metadata.json` md5 一致（**故 `target_service` 标签未变，两版本标签口径可比**）。3 列未被消费（`api_preprocessor.py:70` 按显式白名单选列），本次不接入。
- **`Lv_E_HTTPPATCH_travel2` AUROC=0.2551，是反信号而非无信号**：显著低于 0.5 意味着分数方向系统性反了（模型给异常打的分比正常还低）。PATCH 类故障对现有 endpoint RED 特征隐形是 entry 017 的已知问题，但"隐形"应表现为 ≈0.5，0.255 是另一回事。旧口径 0.2712 也低，本次改动没有引入它、也没有改善它。不在本次范围内。
- **单 seed 报告 RG 类高方差机制会得出方向完全相反的结论（本次实际踩中）**：起草本 entry 时只有 seed=42 的结果，据此写下"RG 是六种机制里唯一退化的"（softmax 0.6510→0.6015，endpoint 档 0.7567→0.5307），并用 gate collapse 机制解释得很自洽。补齐 seed{1,2,3} 后发现 seed42 恰是塌陷点，其余三个 seed 都在 0.858~0.878，**均值实为 0.6575→0.8038 的改善**。教训不是"应该多跑 seed"这种泛泛之谈，而是：**当某机制已被历史记录标注为高方差（entry 014 的 std=0.0792）时，单 seed 结果连"方向"都不能报**，哪怕能给出一个听起来完全合理的机制性解释——那个解释本身会让人更相信错误结论。RG 的真实问题是方差变大（softmax std 0.0352→0.1352）而非均值变差。

## 数字变化：哪些历史数字作废

### `new_merge` contract 口径变化（entry 023 → 本次）

| | entry 023 旧口径 | 本次 |
|---|---|---|
| `eval_all` 行数 | 15809 | **14186** |
| 正样本 | 3113（19.7%） | **916（6.5%）** |
| 训练池 | — | **9513** |
| `label_granularity` 分布 | endpoint / case | endpoint 8132 / **service 5884** / case 170 |

训练池构成 `{fault_inject_nontarget: 5380, fault_baseline: 2879, fault_recover_nontarget: 776, normal_case: 478}`。其中 `fault_inject_nontarget=5380` 是本次新增的实质变化——旧口径下 service/case 档 case 在这一路贡献 0 行（entry 024 指出的那个"靠 bug 撑住的不变量"）。

3 个不可观测 case（`Lv_D_CONNECTION_POOL_exhaustion` / `Lv_D_TRANSACTION_timeout` / `Lv_D_cachelimit`）共 2105 行留在 eval、正样本 0、**inject 行吸收 0 行**（进训练池的 333 行全是 `fault_baseline`，那是另一路、本来就该吸收）——闸门生效。

正样本率 6.5% 比 entry 016 当年判定为"失衡"的 16% 更低，但**成因相反、不是同类问题**：entry 016 的失衡是负样本被掏空（baseline 整段搬进训练池），本次是正样本收窄回其真实身份，负样本绝对数反而从 12696 涨到 13270。6.5% 更接近真实类别先验——一次故障注入只打 1 个 service，8 个 service 里 1 个受影响。**没有任何 fraction 配置能把正样本率"调回" 19.7%**，那个数字里有 2197 行是误标的。

### 六种融合机制（seed{1,2,3,42} 四组，`contract_new_merge_expanded`）

**均值口径（结论以此为准）**：

| fusion | 旧均值 (std) | 新均值 (std) | Δ均值 |
|---|---|---|---|
| gated (L2) | 0.6814 (0.0230) | **0.8892 (0.0053)** | +0.2078 |
| concat (L0) | 0.7109 (0.0259) | **0.8887 (0.0115)** | +0.1778 |
| DWF | 0.7047 (0.0071) | **0.8803 (0.0069)** | +0.1756 |
| indep_concat (L1) | 0.6975 (0.0094) | 0.8575 (0.0239) | +0.1601 |
| RG softmax | 0.6575 (0.0352) | 0.8038 (**0.1352**) | +0.1462 |
| RG indep_sigmoid | 0.6570 (0.0554) | 0.7459 (**0.0977**) | +0.0889 |

**逐 seed 明细**（暴露 RG 的塌陷点）：

| fusion | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|
| L0 | 0.9021 | 0.8744 | 0.8915 | 0.8869 |
| L1 | 0.8767 | 0.8564 | 0.8728 | 0.8243 |
| L2 | 0.8863 | 0.8853 | 0.8885 | 0.8969 |
| DWF | 0.8870 | 0.8812 | 0.8822 | 0.8707 |
| RG softmax | **0.6015** ← 塌陷 | 0.8782 | 0.8772 | 0.8583 |
| RG indep_sigmoid | 0.6896 | **0.6372** ← 塌陷 | 0.8272 | 0.8298 |

**六种机制的均值全部提升，无一退化。** 新旧严格不可比（评估集 15809 vs 14186、正样本率 19.7% vs 6.5%），引用任一数字必须连口径一起报；但位移幅度（+0.09~0.21）远超各自的 seed 内波动，方向性结论成立。

机制排序：**L2 (0.8892) 与 L0 (0.8887) 实际不可分辨**（差 0.0005，远小于各自 std），DWF (0.8803) 紧随，L1 (0.8575) 落后，RG 两变体仍最差且方差极大。修复前 L0 单独最优（0.7109），修复后 L2 追平——但这个"追平"落在噪声范围内，**不足以支撑"L2 优于 L0"的新结论**。

**⚠️ 本 entry 起草时曾基于 seed=42 单点得出"RG 是唯一退化的机制"，该结论错误，已更正。** seed42 恰好是 RG softmax 的塌陷点（0.6015），其余三个 seed 都在 0.858~0.878；按均值算 RG softmax 是 0.6575→0.8038，**改善而非退化**。当时据此写的"RG 的 endpoint 档从 0.7567 崩到 0.5307、与 gate collapse 自洽"那段推理，逻辑本身没错，但它解释的是**一个 seed 的偶发坍缩**，不是 RG 在新标签下的一般行为——把偶发现象当成了系统性结论。这正是 entry 014 "多 seed 方差本身就是证据，不是可忽略的噪声"那条横切主题的又一次实证：**单 seed 报告 RG 类高方差机制会得出方向完全相反的结论**。

**RG 的真实问题是方差变大，不是均值变差**：softmax std 0.0352→0.1352（约 3.8 倍），indep_sigmoid 0.0554→0.0977（约 1.8 倍），而四种稳定机制的 std 全部 ≤0.024（L2 仅 0.0053）。标签变准后 RG 的不稳定性反而更突出。

**AUPRC 上升是本次最反直觉的一点**（seed=42 口径：L0 0.5469→0.7322、L2 0.5201→0.7281、DWF 0.5495→0.7312）。正样本率减半通常让 AUPRC 机械性下降（随机基线 = 正样本率），这里反而普遍上升，说明旧口径那 2197 行误标正样本是**纯噪声**：模型无法把它们与负样本分开，它们只在压低天花板。结论是旧 AUROC/AUPRC **双双虚低**，不是"旧的虚高被打回"——这与修 bug 前的直觉预期（"正样本变少 → AUPRC 会掉"）相反，记录在此以免后续复盘时误判。

### 判别力恢复的分层证据

`by_anomaly_level`（concat）：performance 0.6137→**0.9698**，service 0.7351→**0.9596**，endpoint 0.8894→0.8820。**endpoint 档几乎不动是关键对照**——这些 case 本来就有 `target_endpoint`，走的是未改动的 endpoint 分支；改动只动了该动的档。gated/DWF/L1 同一模式。

单 case（concat）：`Lv_P_CPU_preserve` 0.5428→**0.9993**、`Lv_S_HTTPABORT_preserve` 0.7951→**0.9998**、`Lv_S_KILLPOD_preserve` 0.7307→**0.9978**、`Lv_P_DISKIO_preserve` 0.4948→**0.9187**、`Lv_P_NETLOSS_preserve` 0.7932→0.9930、`Lv_S_DNSFAIL_preserve_no_order` 0.5026→0.8695。**六个里五个原本在 0.5~0.55 区间（等同掷硬币）。这些故障从来不是"信号弱"，是标签错了。**

3 个 `Lv_D_*` case 的分层 AUROC 从有值变 `null`（旧值 0.4642/0.8900/0.5045 是**假数字**——彼时其"正样本"是 8 个 service 的全部 inject 行，与 `tsdb-mysql` 无关）。`by_endpoint` 有 3 组同理变 null。**旧的非 null 值比 null 更有害**：null 会引起注意，0.8900 会被当成真实结论引用。

### 永久无法刷新的历史数字

entry **014 / 016 / 018 / 022** 的 AUROC/AUPRC/gate 权重数字**只能标记作废，不能重跑刷新**。它们都依赖 `configs/data/merged_v2.yaml` = `anomod_v1` + `endpoint_raw2` + `normal_v2`，而后两者已永久丢失：

| 数据集 | 磁盘 | DVC cache | remote | 可否重建 |
|---|---|---|---|---|
| `data/new_merge` | ✅ 本次恢复（124GB / 27 case） | ✗ | ✗ | ✅ |
| `data/anomod_v1` | ✅ 12 case / 41GB | — | — | ✅ |
| `AnoMod/2026_0729`（仓库外） | ✅ 28 case / 126GB | — | — | ep2_0729 的源 |
| `data/new_ep1` | ✗ | ✗ | ✗ | **永久丢失** |
| `data/normal_v2` | ✗ | ✗ | ✗ | **永久丢失** |
| `data/endpoint_raw2` | ✗（`326b885` 删除） | ✗ | ✗ | **永久丢失** |

即 `contract_v0` / `contract_v1` / `contract_v1_expanded`（RG）/ DWF 四条管线都无法从原始数据重建。entry 024 的 TODO 列表里"重建 + 重新训练评估"这一项对它们**不可执行**，那份列表是在不知道 `normal_v2` 也已丢失的前提下写的。这些 entry 的数字此后只能作为"当时如此"的历史记录，不能作为结论引用，也不能与本次数字并列比较。

需要特别标注的是 entry 018 对 DWF 的"不达标"判定（ABORT/REPLACE 宏平均 0.632/0.538，门槛 0.8）：该判定所依据的 eval 标签正是本次修掉的错误标签，且本次 DWF 在 `new_merge` 上 AUROC 0.887、AUPRC 0.731，与 L0 基本持平（0.902/0.732）。这**不构成对"DWF 不达标"的翻案**——两者数据集不同、判据不同（entry 018 看 ABORT/REPLACE 宏平均，本次是 overall），但意味着那条结论的证据基础已不可靠，若要重新判定 DWF 需在 `new_merge` 上按 entry 018 的宏平均口径重算，本次未做。

## 遗留 TODO

- **`ep2_0729` 链路未重建**：entry 024 的 §7 图（`08_endpoint_vs_case_level.png`）仍是旧口径。`configs/data/ep2_0729.yaml` 的 `roots` 指向已失效的 `/tmp/ep2_0729_nolog`，需先改指仓库外的 `AnoMod/2026_0729`（或先把数据搬进仓库）才能重跑。本次未做。
- **DWF 是否翻案未定**：需在 `new_merge` 上按 entry 018 的 ABORT/REPLACE 宏平均口径重算才能判定（见上）。
- **`Lv_E_HTTPPATCH_travel2` 的反信号（AUROC 0.2551）未查**：显著低于 0.5 说明分数方向系统性反了，与 entry 017 的"PATCH 隐形"（应表现为 ≈0.5）不是同一现象，值得单独定位。
- **`body_hash_mismatch_rate` 是 entry 017 "PATCH 类故障对现有 RED 特征全隐形"的一条线索**：备份里多出的 3 列之一直接刻画响应体变化，正是 PATCH 类故障的作用面。本次按用户指示不接入（用户提到可能已在另一设备验证过效果）。
- **DVC remote 为空，是备份卫生问题**：本次能恢复 `new_merge` 纯属另一台设备上有副本。`new_ep1`/`normal_v2` 就没这个运气。
- **CLAUDE.md 声称 `anomod_v1/Normal` 已归档至 `_archive/`，但该目录不存在**：文档与磁盘状态不符，本次未处理。
- **service 档 recover 行的吸收未做**（见上方关键决策），若未来要做需先验证 recover 阶段分布。
- **`01_data_quality.png` 的 `invert_yaxis()`**、**`artifacts/contract_ep2_0729*` 的 `.gitignore`**：entry 024 遗留，仍未处理。
