# Contract v2 验收主表（PR-4 主验收，entry 031 素材）

- 数据集：`new_merge` 27 case，eval_all 14,186 行，23 个 case 同时含正负类（单类 case 不计入 per-case 宏平均）
- 主指标：**per-case macro AUROC**（正负样本同 case，消除 case 间基线差异；entry 027 P0）
- 种子：seed42 走 dvc，seed{1,2,3} 手动覆盖，共 4 seed；mean±std 为总体标准差（ddof=0）
- v1 参照：seed42 取入库 metrics.json，seed{1,2,3} 入库 metrics.json 为旧 eval 格式（无 per_case 键），
  表中 per-case macro 由 `scripts/eval_baseline_v0.py::compute_stratified_metrics` 对既有
  `scores.parquet` 现算得到（脚本：`outputs/v2_rerun_logs/collect_table.py`，未覆盖任何入库文件）
- 平凡基线各只有单次产物（无随机性），seed 列记 "—"
- rel_pos 基线不消费归一化特征，v1/v2 同口径同数字；zscore 基线直接吃 contract 归一化特征，v1/v2 口径不同

## 1. 主表：per-case macro AUROC（4-seed mean±std）

| 方法 | v1 macro AUROC | v2 macro AUROC | v2−v1 | v2 pooled AUROC（附注） |
|---|---|---|---|---|
| concat (L0) | 0.8964 ± 0.0110 | **0.9608 ± 0.0040** | +0.0644 | 0.9598 ± 0.0047 |
| independent_concat (L1) | 0.8566 ± 0.0198 | 0.9513 ± 0.0097 | +0.0947 | 0.9517 ± 0.0101 |
| gated (L2) | 0.8995 ± 0.0102 | 0.9350 ± 0.0061 | +0.0355 | 0.9343 ± 0.0060 |
| deviation_weighted_selfref | 0.8932 ± 0.0180 | 0.9495 ± 0.0033 | +0.0563 | 0.9514 ± 0.0038 |
| 平凡 rel_pos（排名特征） | 0.9656 | 0.9656 | 0 | 0.9303 |
| 平凡 zscore_l2 | 0.9539 | **0.9853** | +0.0314 | 0.9630 |
| 平凡 zscore_max | 0.9441 | 0.9810 | +0.0369 | 0.9548 |

**要点**：v2 四臂相对 v1 同臂全部改善（+0.036～+0.095），且 v2 方差全部收窄
（最大 std 0.0097 vs v1 最大 0.0198）。但四臂**没有一个打赢平凡基线**：最强臂
concat 0.9608 仍低于 v2 zscore_l2 0.9853（−0.0245）、v2 zscore_max 0.9810（−0.0202），
也略低于不依赖归一化的 rel_pos 0.9656（−0.0048）。Deep SVDD 表征学习在 per-case
归一化后的参照系下跑不赢直接聚合 z-score 的零参数基线。

## 2. per-case macro AUPRC（4-seed mean±std）

| 方法 | v1 macro AUPRC | v2 macro AUPRC | v1 pooled AUPRC | v2 pooled AUPRC |
|---|---|---|---|---|
| concat (L0) | 0.7194 ± 0.0259 | 0.8647 ± 0.0105 | 0.7201 ± 0.0292 | 0.8320 ± 0.0185 |
| independent_concat (L1) | 0.6542 ± 0.0232 | 0.8543 ± 0.0203 | 0.6350 ± 0.0326 | 0.8268 ± 0.0223 |
| gated (L2) | 0.7105 ± 0.0146 | 0.7846 ± 0.0266 | 0.7086 ± 0.0199 | 0.7576 ± 0.0351 |
| deviation_weighted_selfref | 0.7225 ± 0.0179 | 0.8413 ± 0.0126 | 0.7170 ± 0.0189 | 0.8229 ± 0.0094 |
| 平凡 rel_pos | 0.6008 | 0.6008 | 0.2990 | 0.2990 |
| 平凡 zscore_l2 | 0.7862 | 0.8223 | 0.7057 | 0.6891 |
| 平凡 zscore_max | 0.7921 | 0.8168 | 0.6875 | 0.6622 |

macro AUPRC 口径下 concat（0.8647）与 independent_concat（0.8543）、selfref（0.8413）
反超三条平凡基线（rel_pos 0.6008 / zscore_l2 0.8223 / zscore_max 0.8168），
gated（0.7846）仍略低于两条 zscore 基线。宏平均与 pooled 的排名差异源于平凡基线
在少数 case 上的极端输出。

## 3. 逐 seed 明细（per-case macro AUROC）

| 方法 | 口径 | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| concat | v2 | 0.9628 | 0.9580 | 0.9561 | 0.9663 |
| concat | v1 | 0.9140 | 0.8835 | 0.8949 | 0.8931 |
| independent_concat | v2 | 0.9546 | 0.9421 | 0.9657 | 0.9427 |
| independent_concat | v1 | 0.8333 | 0.8682 | 0.8826 | 0.8423 |
| gated | v2 | 0.9294 | 0.9299 | 0.9445 | 0.9362 |
| gated | v1 | 0.9131 | 0.8866 | 0.8936 | 0.9046 |
| deviation_weighted_selfref | v2 | 0.9507 | 0.9481 | 0.9541 | 0.9453 |
| deviation_weighted | v1 | 0.9238 | 0.8826 | 0.8884 | 0.8780 |

## 4. Canary：`Lv_P_CPU_preserve` 分层 AUROC（阈值 ≥ 0.9）

| 方法 | 口径 | seed42 | seed1 | seed2 | seed3 | mean±std | 判定 |
|---|---|---|---|---|---|---|---|
| concat | v2 | 0.9966 | 0.9973 | 0.9971 | 0.9966 | 0.9969 ± 0.0003 | PASS |
| independent_concat | v2 | 0.9922 | 0.9928 | 0.9977 | 0.9969 | 0.9949 ± 0.0024 | PASS |
| gated | v2 | 0.9303 | 0.9653 | **0.9026** | 0.9268 | 0.9313 ± 0.0223 | PASS（seed2 压线 0.9026） |
| deviation_weighted_selfref | v2 | 0.9972 | 0.9936 | 0.9969 | 0.9906 | 0.9946 ± 0.0027 | PASS |
| concat（v1 参照） | v1 | 0.9995 | 0.9994 | 0.9998 | 0.9999 | 0.9996 ± 0.0002 | — |
| 平凡 rel_pos | v1=v2 | 0.8895 | — | — | — | 0.8895 | 低于 0.9（非训练臂，仅记录） |
| 平凡 zscore_l2 | v1 / v2 | 0.9989 / 0.9977 | — | — | — | — | PASS |
| 平凡 zscore_max | v1 / v2 | 0.9982 / 0.9976 | — | — | — | — | PASS |

参照系未被污染：四臂 16 个 canary 全部 ≥ 0.9；v1 concat seed42 = 0.9995 与既有记录一致。
唯一压线项是 gated seed2 = 0.9026（边际通过），与 gated 臂整体最弱相符。
rel_pos 平凡基线在该 case 只有 0.8895（其排名特征对资源型 case 不敏感），不作为失败项。

## 5. 三项预期对照

1. **rel_pos 仍 ~0.9656**：v2 口径实测 0.9656（v1 同值，逐位一致），预期成立。
2. **SVDD vs zscore 基线的差距**：存在且方向为负——最强 SVDD 臂 concat（0.9608）
   落后 v2 zscore_l2（0.9853）0.0245、落后 zscore_max（0.9810）0.0202；四臂均值
   0.9492 落后两条 zscore 基线 0.03 以上。zscore 基线在 v2 per-case 归一化下还从
   v1 的 0.9539/0.9441 涨到 0.9853/0.9810：参照系修复对"直接消费归一化特征"的
   零参数基线增益最大。
3. **是否打赢基线**：AUROC 主指标下**没有打赢**（仅 concat 对 rel_pos 的 0.0048
   差距在一个 seed std 量级内）；macro AUPRC 下 concat/L1/selfref 三臂打赢全部
   平凡基线。总体结论：v2 修复参照系是正确且必要的（四臂一致改善、方差收窄、
   canary 全过），但 Deep SVDD + 现有融合相对平凡聚合机制没有体现出附加价值，
   这是 entry 031 需要正面记录的负面结果。

## 6. 训练健康度附注

- 12 次训练全部 50 epoch 正常收敛，scores.parquet 均 14,186 行、0 NaN/inf；
  loss 从 0.13～0.36 降到 7e-5（gated）/0.001～0.012（其余）量级。
- **gated gate 权重**（seed42 checkpoint 在 eval_all 上复算，16 维 sigmoid）：
  无 0/1 饱和（<0.05 与 >0.95 的占比各 ≤1.7%），但门值整体挤在 0.51 附近
  （p5=0.478, p95=0.573，16 维门均值跨度仅 0.098，行间均值 std=0.0096）——
  门控实际上近乎常数、调制作用很弱，与 gated 四臂垫底（0.9350）一致；
  不是 RG 式 softmax 塌缩（权重不归一、不互相竞争），但同样没有学出有效选择性。
- **selfref sigmoid**（固定 threshold=2、scale=1，不可学习）：无病理饱和
  （w<0.05 占 0%，w>0.95 仅 0.6%），权重有选择性——正样本行 w 均值 0.315、
  开启（>0.95）占 5.9%，负样本行 w 均值 0.163、开启占 0.2%；|x| p99=3.73，
  过渡带占 16.4%。
