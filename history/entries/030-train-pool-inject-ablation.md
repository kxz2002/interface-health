# 030 · 训练池 inject 非目标行吸收对照（fraction 1.0→0.0）：5380 行无稳健净收益，L0/DWF 方向相反

- **日期**: 2026-09-19
- **PR**: #29（contract v2 计划 PR-3，栈在 #28 之上） · **Commit**: 见本 PR merge（产物入库 `ec45190`，对照脚本 `c52e310`/`19b1061`）
- **类型**: Experiment
- **影响域**: `configs/contract/v1_new_merge_inject0.yaml`, `scripts/compare_train_pool_ablation.py`, `artifacts/contract_new_merge_inject0/`, `artifacts/baseline_new_merge_inject0_*/`, `artifacts/train_pool_ablation/report.md`, 训练池吸收 fraction / 评估口径

## 做了什么

entry 027 P0 挂账项的实测裁决。现行 `v1_new_merge.yaml` 的 `fault_inject_nontarget_train_fraction=1.0`（entry 022 引入、entry 025 加第二道闸门）把故障 case inject 阶段的 **5380 行非目标行**吸进了 One-Class 训练池当正常数据。设计初衷是补训练池规模/类别平衡，但 entry 025/027 反复指出：故障传播期间"非目标 endpoint"的行并不真的正常（下游被拖累的 service 一样在产生异常 RED 特征）。"污染正常边界"与"平衡类别/扩展覆盖"哪个代价大，此前只有直觉，没有对照实验。

### 方法：单变量对照 + 共有 sample_id 子集口径

- 新增 `configs/contract/v1_new_merge_inject0.yaml`，与 `v1_new_merge.yaml` **仅** `fault_inject_nontarget_train_fraction` 1.0→0.0 一处之差（其余两个 fraction 不动：baseline 0.2、recover nontarget 1.0）。重建 contract：train **9513 → 4133** 行（差恰为 5380），eval_all **14186 → 19566** 行（同样差 5380，吸收摘除与留 eval 严格对偶），**两版正样本均为 916 行**——本实验动的只是负样本的归宿，正样本定义一行没碰。
- L0(concat) 与 DWF × seed{42,1,2,3} = **8 次新训练**（inject0 侧；expanded 侧复用 entry 023 既有 scores）。
- **关键口径：两版 AUROC 不能直接拿全量比**（eval 集构成不同，14186 vs 19566，CLAUDE.md 早就记录"expand_train_pool 开关会改变 eval_all 行数，不同取值不可直接横向比较"）。核验 expanded eval_all **⊆** inject0 eval_all（14186 行全部在 19566 中）后，把两版 scores 都限制到这 14186 行的**共有子集**上现算指标，训练效应与评估集构成效应就此隔离。共有行的 `is_endpoint_anomaly` 标签已逐行核验一致（正样本定义本就不随 fraction 变）。
- 对照脚本：`scripts/compare_train_pool_ablation.py`（对 scores.parquet 复用 `eval_baseline_v0.compute_stratified_metrics` 现算，指标实现无分叉）；产物报告 `artifacts/train_pool_ablation/report.md`，本 entry 全部数字以其为准。

### 主表：共有子集 per-case macro AUROC，4 seed

| fusion | expanded（吸收 5380 行） | inject0（不吸收） | mean Δ（inj−exp） | Δ std | Δ 符号（正/平/负） |
|---|---|---|---|---|---|
| **L0 concat** | **0.8964 ± 0.0110** | 0.8306 ± 0.0758 | **−0.0658** | 0.0808 | 1 / 1 / 2 |
| **DWF** | 0.8932 ± 0.0180 | **0.9330 ± 0.0188** | **+0.0398** | 0.0324 | 3 / 1 / 0 |

per-seed 明细（共有子集 macro / pooled）：

| fusion | seed | expanded macro | inject0 macro | Δ macro | expanded pooled | inject0 pooled |
|---|---|---|---|---|---|---|
| L0 | 42 | 0.9140 | 0.7840 | −0.1299 | 0.9095 | 0.7825 |
| L0 | 1 | 0.8835 | 0.8837 | +0.0001 | 0.8744 | 0.8822 |
| L0 | 2 | 0.8949 | 0.9221 | +0.0272 | 0.8915 | 0.9160 |
| L0 | 3 | 0.8931 | 0.7325 | −0.1606 | 0.8869 | 0.7262 |
| DWF | 42 | 0.9238 | 0.9207 | −0.0031 | 0.9200 | 0.9202 |
| DWF | 1 | 0.8826 | 0.9474 | +0.0649 | 0.8812 | 0.9452 |
| DWF | 2 | 0.8884 | 0.9089 | +0.0206 | 0.8822 | 0.8974 |
| DWF | 3 | 0.8780 | 0.9548 | +0.0767 | 0.8707 | 0.9563 |

（"平"= |Δ|<0.005。）

### 结论

1. **5380 行吸收没有稳健的净收益，证据整体倾向"无益/净损害"。**
2. **L0**：均值上 expanded 高 0.066，但 |mean Δ| < 1σ（0.066 vs 0.081），且 4 个 seed 方向反转（2 负 1 正 1 平）——均值"收益"不稳健，换一个 seed 结论就可能翻面。唯一稳健的效应是**吸收把 L0 的 seed std 从 0.076 压到 0.011**（约 7 倍方差压缩）：吸收行在均值意义上未必加分，但显著降低了训练结果对初始化的敏感度。
3. **DWF**：方向稳定地更好的是 inject0——3 正 1 平 0 负，|mean Δ| 略超 1σ（0.040 vs 0.032）。**吸收对零参数的 DWF 是净损害**。
4. 两种融合结论相反，说明这不是单边故事。**机制假设**（未做分解实验，保持假设级）：L0 的 Deep SVDD 直接吃拼接特征、训练池小（4133 行，其中纯 Normal `train_fit` 仅 478 行）时正常边界欠覆盖，5380 行故障期非目标行可能充当了隐式正则/覆盖扩展，代价与收益在不同 seed 上谁占上风不稳定；DWF 没有可训练参数，其 per-endpoint 偏离量参照被故障行直接带偏，没有任何训练机制能"消化"这些行，故损害单向呈现。entry 029 的 lean vs transductive 对照已从另一侧显示 train 池里的 inject 行会改变退化列/参照系行为，与该假设相容但不等同于证实。

### 对 contract v2（PR-4）的含义

contract v2 为保住 **v1↔v2 行等价回归断言**（同 config 产出的行集合必须逐行一致），三个 fraction 沿用 0.2 / 1.0 / 1.0 不动。本实验的证据意味着：**v2 主表上的 DWF 数字要带着"inject0 口径可能更优 +0.04"这个认知去读**，它未必是该融合机制在当前数据上的最好口径。是否把 inject fraction 正式退回 0.0 留作 v2 之后的独立决策——改 fraction 会改变行集合、直接破坏行等价基准，不能搭车进 v2。

## 关键决策（不在 commit 里）

- **主表只认共有子集，全量数字降级为附注**：两版 eval_all 构成不同，任何全量差距都可能只是"多出来的 5380 行本身更好/更差分"。实测这个坑的量级见下方——直接比全量会把 L0 seed42 的 inject0 读成 0.7521 的"惨相"，而同一模型在完全相同的 14186 行评估集上是 0.7840。
- **对照脚本对 scores.parquet 现算指标，而不是读两边 metrics.json**：expanded seed{1,2,3} 的 metrics.json 是 entry 023 时期的**旧 eval 格式**（无 `per_case_auroc_macro` 三键，entry 029 才新增；seed42 已在 entry 029 随 dvc stage 刷新为新格式）。若按 metrics.json 读，4 个 seed 里 3 个直接拿不到主指标。scores.parquet 是 scores_v0 契约钉住的接口墙，旧 scores 被新版 eval 无改造消费——接口墙的稳定性又一次被实际依赖验证。
- **只做 L0/DWF，不补 L1/L2/RG**：entry 023/027 已确定 L0/DWF 是当前最强且最稳的两种机制（RG 高方差、L1 不优于 L0），裁决"吸收是否值得保留"只需这两个代表；其余机制的同口径对照留待需要时按同脚本重跑（脚本本身支持任意已产出的 fusion 目录）。
- **不顺手改 `v1_new_merge.yaml` 默认值**：本 PR 只新增 inject0 config + 对照，现行 expanded 链路（dvc_new_merge 全部 stage、平凡基线 stage、entry 029 的常驻参照线）一行不动。改默认值是 v2 之后的独立决策（理由见上节）。

## 坑 / 已知问题

- **全量口径不可比的实际演示**（CLAUDE.md 既有 gotcha 的量化版）：expanded 的全量 = 共有子集（14186 即交集本身），inject0 全量是 19566。同一批 inject0 scores 上，全量 per-case macro 比共有子集**系统性偏低**：L0 四个 seed 分别低 0.032 / 0.023 / 0.014 / 0.012（seed42 0.7521 vs 0.7840），DWF 低 0.029 / 0.035 / 0.033 / 0.020。即多出来的 5380 行（正是被取消吸收的那些行）在 per-case macro 上整体更难，直接比全量会凭空给 inject0 叠加一个 −0.01~−0.04 的纯构成偏差。pooled 方向相同。
- **L0 inject0 的高方差（seed42 0.784 / seed3 0.733）不是训练发散**：8 次训练 loss 均正常收敛，scores.parquet 无 NaN（已逐文件核验，min/max 量级与 expanded 同域）。这是去掉 5380 行覆盖后**正常边界欠覆盖的方差放大**——训练池从 9513 缩到 4133、纯 Normal 仍只有 478 行，不同初始化圈出的边界在故障传播行上表现差异被放大。诊断上要与"优化崩坏"区分：看收敛曲线 + score NaN/越界，而不是看到低分 seed 就判训练失败。
- **子集前提自检必须在算指标之前失败**：共有子集口径隐含"expanded ⊆ inject0"前提，若两个 `--*-contract` 参数拿反（或误用别的数据集），交集会变成真子集，"expanded"列将静默失去与历史口径的可比性。脚本在读 scores 前先断言 `exp_ids <= inj_ids`，不满足直接 `SystemExit`（已做参数对调的负向验证：拿反时如期报错退出）；每个 run pair 另有"交集行数在两版 scores 中齐全"的断言，缺行即 `RuntimeError`。
- **n=4 的统计纪律**：|mean Δ| 超过 1 个 Δ std 只是启发式，不是显著性检验。DWF 的 +0.0398 vs std 0.0324 刚过 1σ，按 n=4 配对 Wilcoxon 最多只能到 p≈0.125（单侧，4/4 同向才到 0.0625，实际是 3 正 1 平），**不构成统计显著**。本 entry 结论措辞统一保持"方向 + 方差量级"，不写"显著"；论文若引用该对照，需配 bootstrap CI 或配对 Wilcoxon，并增大 seed 数。

## 遗留 TODO

- 论文引用前的统计加固：bootstrap CI / 配对 Wilcoxon + 增加 seed 数（reviewer 建议，n=4 对 Wilcoxon 的分辨力天花板太低）。
- 机制假设（L0 隐式正则 vs DWF 参照被带偏）未做分解实验；若 v2 之后决定动 inject fraction，先补 per-case 分解（哪些 anomaly_level 的行贡献了 Δ）与训练池规模/构成的析因对照，再改默认值。
- inject fraction 是否正式退回 0.0：v2（PR-4）行等价基准落地之后的独立决策；届时同步评估 entry 029 三条平凡基线在 inject0 口径下的变化（lean z-score 的 fit 源含这 5380 行，退回 0.0 会改变 l2/max 基线数字——entry 029 的 lean vs transductive 对照已预告该耦合）。
- L1/L2/RG 的同口径对照未跑，需要时用 `scripts/compare_train_pool_ablation.py` 对已补齐的 scores 目录直接重算，无需改脚本。
