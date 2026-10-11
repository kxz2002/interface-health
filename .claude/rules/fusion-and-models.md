---
paths:
  - "src/fusion/**"
  - "src/models/**"
  - "configs/fusion/**"
  - "configs/model/**"
  - "scripts/train_baseline_v0.py"
  - "src/data/contract_dataloader.py"
  - "scripts/analyze_gate_weights.py"
  - "tests/test_*fusion*.py"
  - "tests/test_deep_svdd.py"
  - "tests/test_train_baseline_v0.py"
  - "tests/test_contract_dataloader.py"
  - "tests/test_e2e_*.py"
---

# 融合与模型

从根 CLAUDE.md 的 Known Gotchas 迁出（entry 033），原文保留。"高方差机制不能只看单 seed"的方法学结论留在根文件。

- **组合模型 optimizer 必须覆盖全部子模块**：`instantiate(optimizer_cfg, params=svdd.parameters())` 只传 SVDD 参数时，fusion encoder 层永远停在随机初始化——L1/L2 在 entry 012 实际命中此 bug（消融结果与 L0 无差异直到修复）。任何新增 fusion+model 组合上线前须验证 optimizer 参数集覆盖所有 `nn.Module`
- **eval NaN 填补用 train 均值**：`ContractDataset(fit_on_parquet=train.parquet)` 用训练集统计量填 eval 的 NaN，严禁用 eval 自身均值，否则 eval 统计泄漏到特征
- **RG softmax gate collapse**（**RG 已于 entry 031 从 pipeline 退役**：两个 stage 从 `dvc_new_merge/dvc.yaml` 移除，类代码/配置/历史产物保留在树中备查）：`ReliabilityGatedFusion` 默认 `gate_normalization=softmax` 在 `merged_v2`/`endpoint_raw2` 上于所有 27 种故障类型下均收敛到 `w_svc≈1.0`（完全信任 service 分支，endpoint 分支权重归零）。根因：两分支初始偏差尺度 ~1.2× 不对称 + softmax 竞争归一化形成正反馈放大。`independent_sigmoid` 消融（`configs/fusion/reliability_gate_ablation_indep_sigmoid.yaml`）可规避此问题。**具体数值不在此固化**：entry 014 的 0.6598 vs 0.6024 依赖已永久丢失的 `endpoint_raw2`/`normal_v2` 且建立在 entry 025 修掉的错误标签之上，已作废；entry 023 在 `new_merge` 上曾测得两者几乎持平，但那也是修复前口径。entry 025 修复后（`new_merge`，四 seed 均值）softmax 0.6575→0.8038、indep_sigmoid 0.6570→0.7459，**两者均值皆改善、无一退化**。**RG 的真实问题是方差极大**（softmax std 0.0352→0.1352，约 3.8 倍；indep_sigmoid 0.0554→0.0977，约 1.8 倍），远超稳定机制（L0/L2/DWF std ≤0.012）——entry 025 起草时只有 seed42、据此判定"RG 唯一退化"（softmax 单点 0.6015），补齐 seed{1,2,3} 后发现 seed42 恰是塌陷点、其余三个 seed 都在 0.858~0.878。教训是**单 seed 报告高方差机制连"方向"都会错**。以最新 history entry 的表格为准
- **RG coupling 已收束（见 history/entries/015）**：① fusion 构造改为 `FusionModule.from_contract` 钩子，RG 覆写自行加载 `EndpointBaselineStats`+派生 `id_to_endpoint_key`，`train_baseline_v0.py` 不再按 `_target_` 字符串分支；② `EndpointBaselineStats` fit/save 与 `endpoint_id` 列派生改由 `ContractConfig.fit_endpoint_baseline_stats` 开关门控（v1.yaml=false / v1_expanded_pool.yaml=true），不再无条件 fit；③ 3 个 RG 专属 stage 隔离到 `dvc_reliability_gate/dvc.yaml`，根目录无 target 的 `dvc repro` 不触发。`scripts/analyze_gate_weights.py`（一次性分析脚本）维持直接构造 RG 的用法，不在收束范围内
- **DeviationWeightedFusion（DWF）零参数逐特征加权，判定不达标不做后续深化（见 history/entries/018）**（v2 另有自参照新类 `SelfReferentialDeviationFusion`：公式不变、直读特征当偏离量、无 sidecar，是 v2 四臂之一，见 history/entries/031）：ABORT/REPLACE 宏平均 AUROC 0.632/0.538，与 RG 基本无差异（REPLACE 更差）。构造时用 `EndpointBaselineStats.red_cols`/`svc_cols` 属性按值校验传入的 `red_cols`/`svc_cols` 顺序与内部 fit 顺序一致（不只是长度），因为 `schema.json` 的列顺序与 `EndpointBaselineStats` 的 fit 顺序是独立派生的两条链路，理论上可能同长度但顺序不同，此时 `branch_stats()` 的 mean/std 会被错位应用到错误的特征位置，计算不报错但结果是错的
- **POT/GPD 动态阈值**只需调 q 一个参数，但极度依赖异常分数分布假设（仓库内尚无实现）
