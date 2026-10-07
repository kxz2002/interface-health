# Contract v2 验收主表（PR-4 主验收，entry 031 素材）

> ⚠ **数字更新（2026-10-07）**：本表 §1–§4 数字已按 C1+C2 重跑结果重写——
> C1：v1 四臂 seed{1,2,3} 以当前特征集重跑（修复 X1：旧产物为 #25 之前特征集，
> 见 entry 030"已知混淆"）；C2：30-30（v2 n=0 组 mean 回退修复，仅
> `endpoint_red__trace_5xx_rate` 一列变化：train 201 单元 / eval 48 单元，
> 其中 39 个正样本 z 1→0）后 v2 contract 重建、v2 四臂 × 4 seed 与三条 v2
> 平凡基线全部重跑。**§5/§6 的文字写于更新之前，与当前数字可能不一致，
> 待用户确认结论措辞后改写。**

- 数据集：`new_merge` 27 case，eval_all 14,186 行，23 个 case 同时含正负类（单类 case 不计入 per-case 宏平均）
- 主指标：**per-case macro AUROC**（正负样本同 case，消除 case 间基线差异；entry 027 P0）
- 种子：seed42 走 dvc，seed{1,2,3} 手动覆盖，共 4 seed；mean±std 为总体标准差（ddof=0）
- v2 contract 由 `build_contract.py --config configs/contract/v2_new_merge.yaml --dataset configs/data/new_merge.yaml --out-dir artifacts/contract_new_merge_v2 --seed 42` 手动构建（无 dvc 构建 stage，stage 待补，见 entry 031 遗留 TODO）；2026-10-07 起为 30-30 修复后版本
- v1 参照：seed{1,2,3} 已于 2026-10-07 以当前特征集重跑（C1）；全部 per-case macro
  由 `scripts/eval_baseline_v0.py::compute_stratified_metrics` 对 `scores.parquet`
  现算得到（脚本：`scripts/collect_v2_acceptance_table.py`，未覆盖任何入库文件）
- 平凡基线各只有单次产物（无随机性），seed 列记 "—"
- rel_pos 基线不消费归一化特征，v1/v2 同口径同数字；zscore 基线直接吃 contract 归一化特征，v1/v2 口径不同

## 1. 主表：per-case macro AUROC（4-seed mean±std）

| 方法 | v1 macro AUROC | v2 macro AUROC | v2−v1 | v2 pooled AUROC（附注） |
|---|---|---|---|---|
| concat (L0) | 0.8670 ± 0.0320 | **0.9454 ± 0.0083** | +0.0784 | 0.9468 ± 0.0089 |
| independent_concat (L1) | 0.8646 ± 0.0429 | 0.9465 ± 0.0055 | +0.0819 | 0.9500 ± 0.0072 |
| gated (L2) | 0.8581 ± 0.0815 | 0.9368 ± 0.0041 | +0.0787 | 0.9375 ± 0.0044 |
| deviation_weighted_selfref | 0.9507 ± 0.0169 | 0.9277 ± 0.0072 | −0.0230 | 0.9341 ± 0.0054 |
| 平凡 rel_pos（排名特征） | 0.9656 | 0.9656 | 0 | 0.9303 |
| 平凡 zscore_l2 | 0.9539 | **0.9853** | +0.0314 | 0.9630 |
| 平凡 zscore_max | 0.9441 | 0.9810 | +0.0369 | 0.9548 |

**要点**（2026-10-07 数字）：v2 三个 encoder 臂（concat/independent/gated）相对 v1
同臂全部改善（+0.079～+0.082）且方差全部收窄；但 **deviation_weighted_selfref 方向
反转**（v2 比 v1 低 0.0230，std 收窄约 2.3 倍）。四臂仍没有一个打赢平凡基线：
最强臂 independent_concat 0.9465 低于 v2 zscore_l2 0.9853（−0.0388）、
zscore_max 0.9810（−0.0345），也低于不依赖归一化的 rel_pos 0.9656。

## 2. per-case macro AUPRC（4-seed mean±std）

| 方法 | v1 macro AUPRC | v2 macro AUPRC | v1 pooled AUPRC | v2 pooled AUPRC |
|---|---|---|---|---|
| concat (L0) | 0.5662 ± 0.0835 | 0.8549 ± 0.0080 | 0.5550 ± 0.0801 | 0.8443 ± 0.0067 |
| independent_concat (L1) | 0.6346 ± 0.0823 | 0.8407 ± 0.0100 | 0.5974 ± 0.0661 | 0.8371 ± 0.0132 |
| gated (L2) | 0.6102 ± 0.1486 | 0.7880 ± 0.0247 | 0.5884 ± 0.1484 | 0.7671 ± 0.0333 |
| deviation_weighted_selfref | 0.8047 ± 0.0385 | 0.8204 ± 0.0074 | 0.7978 ± 0.0368 | 0.8179 ± 0.0156 |
| 平凡 rel_pos | 0.6008 | 0.6008 | 0.2990 | 0.2990 |
| 平凡 zscore_l2 | 0.7862 | 0.8223 | 0.7057 | 0.6891 |
| 平凡 zscore_max | 0.7921 | 0.8168 | 0.6875 | 0.6622 |

macro AUPRC 口径下 concat（0.8549）与 independent_concat（0.8407）反超三条平凡
基线（rel_pos 0.6008 / zscore_l2 0.8223 / zscore_max 0.8168），selfref（0.8204）
反超 rel_pos 与 zscore_max、但仍略低于 zscore_l2；gated（0.7880）仍低于两条
zscore 基线。宏平均与 pooled 的排名差异源于平凡基线在少数 case 上的极端输出。

## 3. 逐 seed 明细（per-case macro AUROC）

| 方法 | 口径 | seed42 | seed1 | seed2 | seed3 |
|---|---|---|---|---|---|
| concat | v2 | 0.9311 | 0.9495 | 0.9494 | 0.9515 |
| concat | v1 | 0.9140 | 0.8442 | 0.8781 | 0.8319 |
| independent_concat | v2 | 0.9421 | 0.9474 | 0.9553 | 0.9414 |
| independent_concat | v1 | 0.8333 | 0.8330 | 0.8549 | 0.9373 |
| gated | v2 | 0.9366 | 0.9316 | 0.9431 | 0.9360 |
| gated | v1 | 0.9131 | 0.8460 | 0.7306 | 0.9427 |
| deviation_weighted_selfref | v2 | 0.9187 | 0.9382 | 0.9295 | 0.9243 |
| deviation_weighted | v1 | 0.9238 | 0.9603 | 0.9686 | 0.9502 |

## 4. Canary：`Lv_P_CPU_preserve` 分层 AUROC（阈值 ≥ 0.9）

| 方法 | 口径 | seed42 | seed1 | seed2 | seed3 | mean±std | 判定 |
|---|---|---|---|---|---|---|---|
| concat | v2 | 0.9968 | 0.9954 | 0.9972 | 0.9931 | 0.9956 ± 0.0016 | PASS |
| independent_concat | v2 | 0.9952 | 0.9897 | 0.9938 | 0.9883 | 0.9918 ± 0.0028 | PASS |
| gated | v2 | 0.9453 | 0.9781 | 0.9592 | 0.9661 | 0.9622 ± 0.0121 | PASS |
| deviation_weighted_selfref | v2 | 0.9952 | 0.9956 | 0.9965 | 0.9957 | 0.9958 ± 0.0005 | PASS |
| concat（v1 参照） | v1 | 0.9995 | 0.9996 | 0.9995 | 1.0000 | 0.9997 ± 0.0002 | — |
| 平凡 rel_pos | v1=v2 | 0.8895 | — | — | — | 0.8895 | 低于 0.9（非训练臂，仅记录） |
| 平凡 zscore_l2 | v1 / v2 | 0.9989 / 0.9977 | — | — | — | — | PASS |
| 平凡 zscore_max | v1 / v2 | 0.9982 / 0.9976 | — | — | — | — | PASS |

参照系未被污染：四臂 16 个 canary 全部 ≥ 0.9（最低为 gated seed42 = 0.9453）。

## 5. 三项预期对照

> ⚠ 本节文字写于 2026-10-07 数字更新之前（当时 v2 concat 0.9608、selfref 0.9495、
> v1 列为 X1 混淆口径），引用的具体数字与 §1–§4 当前值不一致，待用户确认后改写。

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

> ⚠ 本节观察来自 2026-10-07 更新之前的训练（旧 v2 contract、X1 混淆口径 v1），
> gate 权重分布等细节需在新 checkpoint 上复算后再刷新，待用户确认。

- 16 次训练（seed{1,2,3} 手动 12 次 + seed42 走 dvc 4 次）全部 50 epoch 正常收敛，
  scores.parquet 均 14,186 行、0 NaN/inf（健康度检查覆盖手动 12 次的日志，
  见 `outputs/v2_rerun_logs/`）；
  loss 从 0.13～0.36 降到 7e-5（gated）/0.001～0.012（其余）量级。
- **gated gate 权重**（seed42 checkpoint 在 eval_all 上复算，16 维 sigmoid）：
  无 0/1 饱和（<0.05 与 >0.95 的占比各 ≤1.7%），但门值整体挤在 0.51 附近
  （p5=0.478, p95=0.573，16 维门均值跨度仅 0.098，行间均值 std=0.0096）——
  门控实际上近乎常数、调制作用很弱，与 gated 四臂垫底（0.9350）一致；
  不是 RG 式 softmax 塌缩（权重不归一、不互相竞争），但同样没有学出有效选择性。
- **selfref sigmoid**（固定 threshold=2、scale=1，不可学习）：无病理饱和
  （w<0.05 占 0%，w>0.95 仅 0.6%），权重有选择性——正样本行 w 均值 0.315、
  开启（>0.95）占 5.9%，负样本行 w 均值 0.163、开启占 0.2%；|x| p99=3.73
  （eval_all 口径；train 侧 p99=7.42），
  过渡带占 16.4%。
