# 训练池 inject 非目标行吸收对照（entry 027 P0 / Task 8）

- 生成时代码 commit：`19b10618e9b4c0ad1e35743e1e4447fc70786ba1`

重跑命令：

```bash
python scripts/compare_train_pool_ablation.py \
    --expanded-contract artifacts/contract_new_merge_expanded \
    --inject0-contract artifacts/contract_new_merge_inject0 \
    --scores-root artifacts \
    --out artifacts/train_pool_ablation/report.md
```

## 口径

- expanded = `fault_inject_nontarget_train_fraction=1.0`（现行 v1_new_merge，5380 行 inject 阶段非目标行被吸进 One-Class 训练池）
- inject0 = 同 fraction 退回 0.0（这些行整段留 eval_all）
- expanded eval_all 14186 行 / inject0 eval_all 19566 行 / 共有 sample_id 子集 **14186 行**
- 主指标：per-case macro AUROC（正负样本同 case，免疫 case 身份 shortcut，entry 027 起升为主指标），由 `scripts/eval_baseline_v0.py` 的 `compute_stratified_metrics` 计算，指标实现无分叉
- delta = inject0 − expanded；**delta<0 表示去掉吸收后变差，即吸收是净收益**；delta>0 表示吸收是净损害
- 结论按 4 seed mean±std 判断，不看单 seed（entry 025 教训：RG 单 seed 方向判反）

## 1. 共有子集口径（主结论依据，两版评估样本完全相同）

```
                  fusion  seed  expanded_macro  inject0_macro  delta_macro  expanded_pooled  inject0_pooled
             concat (L0)    42          0.9140         0.7840      -0.1299           0.9095          0.7825
             concat (L0)     1          0.8835         0.8837       0.0001           0.8744          0.8822
             concat (L0)     2          0.8949         0.9221       0.0272           0.8915          0.9160
             concat (L0)     3          0.8931         0.7325      -0.1606           0.8869          0.7262
deviation_weighted (DWF)    42          0.9238         0.9207      -0.0031           0.9200          0.9202
deviation_weighted (DWF)     1          0.8826         0.9474       0.0649           0.8812          0.9452
deviation_weighted (DWF)     2          0.8884         0.9089       0.0206           0.8822          0.8974
deviation_weighted (DWF)     3          0.8780         0.9548       0.0767           0.8707          0.9563
```

### 4 seed 汇总（共有子集）

```
                  fusion expanded macro (4 seed) inject0 macro (4 seed)  mean delta  delta std expanded pooled (4 seed) inject0 pooled (4 seed)
             concat (L0)         0.8964 ± 0.0110        0.8306 ± 0.0758     -0.0658     0.0808          0.8906 ± 0.0126         0.8268 ± 0.0760
deviation_weighted (DWF)         0.8932 ± 0.0180        0.9330 ± 0.0188      0.0398     0.0324          0.8885 ± 0.0187         0.9298 ± 0.0228
```

## 2. 全量口径（附注：两版 eval 集构成不同，14186 vs 19566 行，**不可直接比较，仅供参考**）

```
                  fusion  seed  exp_n  inj_n  expanded_macro  inject0_macro  expanded_pooled  inject0_pooled
             concat (L0)    42  14186  19566          0.9140         0.7521           0.9095          0.7454
             concat (L0)     1  14186  19566          0.8835         0.8603           0.8744          0.8502
             concat (L0)     2  14186  19566          0.8949         0.9079           0.8915          0.8921
             concat (L0)     3  14186  19566          0.8931         0.7206           0.8869          0.7127
deviation_weighted (DWF)    42  14186  19566          0.9238         0.8921           0.9200          0.8884
deviation_weighted (DWF)     1  14186  19566          0.8826         0.9125           0.8812          0.9086
deviation_weighted (DWF)     2  14186  19566          0.8884         0.8759           0.8822          0.8629
deviation_weighted (DWF)     3  14186  19566          0.8780         0.9346           0.8707          0.9246
```

## 3. 结论

### concat (L0)

- 共有子集 per-case macro AUROC：expanded 0.8964 ± 0.0110 → inject0 0.8306 ± 0.0758，mean delta = -0.0658（delta std 0.0808）
- 4 seed delta 符号：delta>0 1 个 / ≈0（|delta|<0.005） 1 个 / <0 2 个——方向跨 seed 反转
- 判定：inject 非目标行吸收在 4 seed 均值上是**净收益（去掉后更差）**；|mean delta| 未超过 1 个 delta std，种子间波动与均值同量级，结论需谨慎

### deviation_weighted (DWF)

- 共有子集 per-case macro AUROC：expanded 0.8932 ± 0.0180 → inject0 0.9330 ± 0.0188，mean delta = +0.0398（delta std 0.0324）
- 4 seed delta 符号：delta>0 3 个 / ≈0（|delta|<0.005） 1 个 / <0 0 个
- 判定：inject 非目标行吸收在 4 seed 均值上是**净损害（去掉后更好）**；|mean delta| 超过 1 个 delta std，方向较稳

---

判读规则：主结论只采用第 1 节（同一样本子集）；第 2 节全量数字因评估集不同，
任何表面差距都可能只是构成差异，不作为依据。
