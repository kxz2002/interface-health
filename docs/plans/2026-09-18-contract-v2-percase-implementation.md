# Contract v2 per-case 归一化 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把主 pipeline 的归一化参照系从"单一 Normal case 全局 min-max"改为"per-case z-score + 层级 shrinkage"，让评估摆脱采集漂移污染，并把平凡基线固化为常驻参照线。

**Architecture:** 新增 contract v2（不改 v1，两者并列做消融）。`Normalizer` 扩两个 per-case scope + `z_score` method + 三级 shrinkage 回退；`build_contract.py` 把故障 case 的时序切分提到归一化之前，使 fit 集合严格 ⊆ 训练池行；DWF 以新类实现自参照版本（旧类保留以维持 v1 回归门）；RG 退役；平凡基线以独立 scores 生产者进入现有接口墙。

**Tech Stack:** Python 3 / pandas / PyTorch / Hydra / DVC / pytest。conda 环境名 `interface`。

**设计依据:** `docs/plans/2026-09-18-contract-v2-percase-design.md`（决策 D1–D6 与验收标准均以该文档为准）

---

## 环境与前置约定（每个任务都适用）

```bash
# 所有 python / pytest / dvc 命令必须走 interface 环境
# base 环境没有 dvc、也没有 torch（CLAUDE.md Known Gotchas 已记录）
conda run -n interface pytest tests/ -q
conda run -n interface dvc repro dvc_new_merge/dvc.yaml
```

- 当前分支 `exp/new-merge-rerun-pr24-pr25`，起点 HEAD = `e8deb74`
- 提交格式必须是 `[Type]: 描述`（见 CLAUDE.md），type 取 `[Feature]`/`[Bugfix]`/`[Experiment]`/`[Docs]`/`[Test]`/`[Refactor]`
- pre-commit 会自动跑 isort + black；若提交被格式化改写，重新 `git add` 后再提交
- **每个 PR 合并前必须写 history entry**（项目硬规则），entry 编号接续现有最大号（当前最大 027）

---

## 本计划相对设计文档的一处偏离，及理由

设计文档 D4 写的是"DWF **原地**重写为无状态纯函数"。本计划改为**新建类** `SelfReferentialDeviationFusion`，旧 `DeviationWeightedFusion` 一行不改。

理由：设计文档 §6 的回归门要求"**v1 口径 repro 数字逐位不变**"。v1 的 DWF arm 依赖 sidecar 与旧公式，原地重写会让该 arm 数字变化，回归门失去意义，v1↔v2 也不再是"只有归一化不同"的干净对比。新建类使 v1 arm 完全冻结、对比变量唯一。旧类在 RG 退役后仍是 v1 arm 的活跃依赖，不删。

---

# PR-1：落地 exp 分支（零代码改动）

## Task 1: 补写 entry 028（PR #26 欠的 history entry）

**Files:**
- Create: `history/entries/028-downstream-benchmark.md`
- Modify: `history/index.md`

**Step 1: 读取事实来源**

```bash
git show 25cd20b --stat | head -30
sed -n '1,120p' artifacts/downstream_benchmark/summary.md
conda run -n interface python -c "
import pandas as pd
df = pd.read_csv('artifacts/downstream_benchmark/_summary.tsv', sep='\t')
g = df[df.pool_mode=='expanded'].groupby(['feature_set','model'])['per_case_auroc_macro'].agg(['mean','std'])
print(g.round(4))"
```

**Step 2: 按 `history/entries/_template.md` 的结构写 entry**

必须覆盖以下内容（这些事实在 onboarding 时已核实，散落在 summary.md / _summary.tsv / commit message 三处）：

- 做了什么：26 runs × {F-inherited 18 维, F-full 21 维} × {DeepSVDD, OCSVM, IF} × 4 seed，回应 Reviewer 2 "只报字段可分性、没跑真实检测器"
- 核心结论（expanded 池，per-case macro）：DeepSVDD 0.866→0.957（+0.091）、OCSVM 0.836→0.950、IF 0.733→0.814，三检测器方向一致
- 缺失值混淆消融：`missing_indicator_only` 仅 0.667；`full_with_missing_indicator` 相对 F-full 只 +0.002 ⇒ 增益来自字段真实取值而非 NaN 模式
- **坑 1**：commit message 与 PR 描述里的 0.851→0.951 是 review 修复前的旧数字，已过时；引用以 `summary.md` / `_summary.tsv` 为准（0.866→0.957）
- **坑 2**：per-case macro 实际只覆盖 16 个 `Lv_E_HTTP*` endpoint 级 case（其余 10 个正样本为 0，不进 macro，见每份 metrics.json 的 `n_cases_with_both_classes_for_auroc: 16`）。引用"26 runs"时必须同时声明此范围
- **坑 3**：PR 描述承诺的 `_summary_agg.tsv` 实际不存在
- **坑 4**：本 benchmark 是与 contract 并行的**第二套**特征/切分实现（直读 `tt_fused_15s.csv`，自带 21 维清单与 0.8 吸收比例，不挂 DVC），未来有口径漂移风险
- 平凡基线（与 entry 027 的关键连接）：expanded 池 `_internal_rel_pos_per_case_auroc` = 0.5786、`_internal_zscore_per_case_auroc` = 0.7657，**远低于 headline 0.957 ⇒ 论文返修数字安全**；但 normal_only 池 rel_pos 0.800 高于该池所有学习模型（DeepSVDD 0.636），按当时决定仅存 `_internal_only_` 键、不入返修表

**Step 3: 更新 index**

在 `history/index.md` 的条目列表加一行，并把影响域倒排索引里 `scripts/benchmark_downstream.py`、`artifacts/downstream_benchmark/` 指向 028。

**Step 4: 提交**

```bash
git add history/entries/028-downstream-benchmark.md history/index.md
git commit -m "[Docs]: 补写 entry 028（PR #26 下游 benchmark 欠账）

PR #26 合并时未按项目规则写 history entry，本次补齐。记录 26 runs ×
F-inherited/F-full × 三检测器的结论（DeepSVDD 0.866→0.957），以及四个坑：
commit message 旧数字、per-case macro 实际只覆盖 16 个 endpoint 级 case、
承诺的 _summary_agg.tsv 不存在、benchmark 是与 contract 并行的第二套口径。

同时记录平凡基线在该 benchmark 上的数值（expanded 池 rel_pos 0.579 /
zscore 0.766，远低于 headline 0.957），确认论文返修数字不受 entry 027
发现的时间混淆问题影响。"
```

## Task 2: entry 027 定稿

**Files:**
- Modify: `history/entries/027-temporal-confounding-trivial-baseline.md`

**Step 1: 补齐元信息与交叉引用**

- 头部 `**PR**: 待定 · **Commit**: 待定（分支 ...）` → 填入真实 PR 号与 commit `56c4561`
- "遗留 TODO" 的 P0 段落加一行，指向 `docs/plans/2026-09-18-contract-v2-percase-design.md` 与本实施计划
- P0 清单里"归一化改 per-case 自适应"一项补注：**已定案用 z-score 而非 min-max**，理由见设计文档 D1（fit 子集 median 16 行、26/208 组 <5 行的实测）
- "坑 / 已知问题"里 RG indep_sigmoid 塌陷那条补注：**已定案 RG 整体退役，不修**（设计文档 D5）

**Step 2: 提交**

```bash
git add history/entries/027-temporal-confounding-trivial-baseline.md
git commit -m "[Docs]: entry 027 定稿（补 PR/commit 元信息与设计文档交叉引用）"
```

## Task 3: rebase 到 master 并推送

**Step 1: 确认零冲突（onboarding 已核实，此处复验）**

```bash
git fetch origin
comm -12 <(git diff --name-only $(git merge-base HEAD origin/master)..HEAD | sort) \
         <(git diff --name-only $(git merge-base HEAD origin/master)..origin/master | sort)
```
Expected: 空输出（分支与 #26 零文件交集）

**Step 2: rebase**

```bash
git rebase origin/master
```
Expected: 无冲突完成。若出现冲突，**停下来报告**，不要自行 `--skip` 或 `--force`。

**Step 3: 全量测试**

```bash
conda run -n interface pytest tests/ -q
```
Expected: 全绿

**Step 4: 推送并建 PR**

```bash
git push -u origin exp/new-merge-rerun-pr24-pr25
gh pr create --title "[Experiment]: 025+026 组合重跑 + 时间混淆诊断（平凡基线打败全部融合机制）" --body "<见下>"
```

PR 描述必须包含：两部分工作（组合重跑解除 entry 026 数字失效声明 / 时间混淆诊断）、平凡基线三个数字、RG indep_sigmoid 塌陷根因、entry 027+028、设计文档与实施计划链接、**明确声明本 PR 零代码改动**（只有实验产物与文档）。

---

# PR-2：eval 参照线（v1 口径立即生效）

## Task 4: eval 脚本加 per-case 宏平均指标

**Files:**
- Modify: `scripts/eval_baseline_v0.py`
- Test: `tests/test_eval_per_case_macro.py`（新建）

**Step 1: 写失败测试**

```python
"""per-case 宏平均指标：只对同时含正负类的 case 取均值。"""
from __future__ import annotations

import pandas as pd

from scripts.eval_baseline_v0 import compute_stratified_metrics


def _row(case, sample, score, pos):
    return {
        "sample_id": sample,
        "score": score,
        "y_true": int(pos),
        "is_endpoint_anomaly": bool(pos),
        "case_id": case,
        "endpoint_key": "ep1",
        "phase": "inject" if pos else "baseline",
        "anomaly_type": case,
        "anomaly_level": "Lv_E",
        "label_granularity": "endpoint",
    }


def test_per_case_macro_skips_single_class_cases():
    # caseA 完美可分（AUROC 1.0），caseB 全负（单类，必须被跳过）
    rows = [
        _row("caseA", "a1", 0.1, False),
        _row("caseA", "a2", 0.9, True),
        _row("caseB", "b1", 0.5, False),
        _row("caseB", "b2", 0.6, False),
    ]
    m = compute_stratified_metrics(pd.DataFrame(rows))
    assert m["per_case_auroc_macro"] == 1.0
    assert m["n_cases_with_both_classes"] == 1


def test_per_case_macro_averages_across_cases():
    # caseA AUROC 1.0，caseC AUROC 0.0 → 宏平均 0.5
    rows = [
        _row("caseA", "a1", 0.1, False),
        _row("caseA", "a2", 0.9, True),
        _row("caseC", "c1", 0.9, False),
        _row("caseC", "c2", 0.1, True),
    ]
    m = compute_stratified_metrics(pd.DataFrame(rows))
    assert m["per_case_auroc_macro"] == 0.5
    assert m["n_cases_with_both_classes"] == 2
    assert "per_case_auprc_macro" in m
```

**Step 2: 运行确认失败**

```bash
conda run -n interface pytest tests/test_eval_per_case_macro.py -v
```
Expected: FAIL — `KeyError: 'per_case_auroc_macro'`

**Step 3: 实现**

在 `scripts/eval_baseline_v0.py` 的 `compute_stratified_metrics` 里、现有 `by_endpoint` 分层之后加入：

```python
    # per-case 宏平均升为主指标（entry 027 P0）：pooled AUROC 会被 case 间基线
    # 差异影响，per-case 口径的正负样本都来自同一 case，是更严格的度量。只统计
    # 同时含正负类的 case——单类 case 的 AUROC 无定义，计入会静默拉偏均值
    # （PR #26 的 benchmark 已按同一规则处理，此处保持一致以便两套口径可比）。
    case_aurocs: list[float] = []
    case_auprcs: list[float] = []
    for _case_id, sub in df.groupby("case_id"):
        y = sub[label_col]
        if y.nunique() < 2:
            continue
        case_aurocs.append(roc_auc_score(y, sub["score"]))
        case_auprcs.append(average_precision_score(y, sub["score"]))

    result["per_case_auroc_macro"] = (
        float(sum(case_aurocs) / len(case_aurocs)) if case_aurocs else None
    )
    result["per_case_auprc_macro"] = (
        float(sum(case_auprcs) / len(case_auprcs)) if case_auprcs else None
    )
    result["n_cases_with_both_classes"] = len(case_aurocs)
```

这三个键放在**顶层** metrics dict（不是 `stratified` 子字典），因为它们是主指标。`metrics_v0` 契约显式允许任意额外字段（`src/contracts/metrics_v0.py:19-22`），**无需改契约**。

**Step 4: 运行确认通过**

```bash
conda run -n interface pytest tests/test_eval_per_case_macro.py -v
conda run -n interface pytest tests/ -q
```
Expected: 全绿

**Step 5: 提交**

```bash
git add scripts/eval_baseline_v0.py tests/test_eval_per_case_macro.py
git commit -m "[Feature]: eval 增加 per-case 宏平均 AUROC/AUPRC 主指标

entry 027 P0：pooled AUROC 受 case 间基线差异影响，per-case 口径正负样本
同源，是更严格的度量。只统计同时含正负类的 case（单类 AUROC 无定义，计入
会静默拉偏），与 PR #26 benchmark 的同名口径保持一致以便两套结果可比。
metrics_v0 契约允许额外字段，无需升版。"
```

## Task 5: 平凡基线 scores 生产者

**Files:**
- Create: `scripts/score_trivial_baselines.py`
- Test: `tests/test_score_trivial_baselines.py`

**Step 1: 写失败测试**

```python
"""平凡基线：rel_pos / zscore_l2 / zscore_max，且统计量只用训练池行。"""
from __future__ import annotations

import pandas as pd

from scripts.score_trivial_baselines import compute_rel_pos_scores, compute_zscore_scores


def _contract_frame():
    """2 个 case × 1 endpoint × 4 窗；每个 case 后两窗为 inject。"""
    rows = []
    for case, base in (("c1", 0.0), ("c2", 10.0)):
        for i, ts in enumerate([1000, 2000, 3000, 4000]):
            inject = i >= 2
            rows.append(
                {
                    "sample_id": f"{case}_{ts}",
                    "case_id": case,
                    "endpoint_key": "ep1",
                    "timestamp_window_ms": ts,
                    "phase": "inject" if inject else "baseline",
                    "is_endpoint_anomaly": inject,
                    "f__a": base + (100.0 if inject else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_rel_pos_is_monotone_within_case_and_case_independent():
    out = compute_rel_pos_scores(_contract_frame())
    assert list(out.columns) == ["sample_id", "score", "y_true"]
    s = out.set_index("sample_id")["score"]
    assert s["c1_1000"] < s["c1_2000"] < s["c1_3000"] < s["c1_4000"]
    # 两个 case 的同序位窗口得分相同（rel_pos 不含 case 身份信息）
    assert s["c1_1000"] == s["c2_1000"]


def test_zscore_uses_only_allowed_rows_no_leakage():
    """统计量只能来自 fit_df。把 fit_df 限成前两窗，eval 侧 inject 巨值不得
    参与 mean/std——否则 z 会被自身拉平、分数塌陷。"""
    df = _contract_frame()
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    s = out.set_index("sample_id")["score"]
    assert s["c1_3000"] > 10 * max(s["c1_1000"], s["c1_2000"])


def test_zscore_degenerate_std_does_not_explode():
    """fit 段零方差列不得用 1e-9 兜底（会放大 1e9 倍——history 013 的爆值与
    entry 027 的 RG sigmoid 饱和同源），须用 1.0 哨兵。"""
    df = _contract_frame()
    df["f__const"] = 5.0
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__const"], agg="max")
    assert out["score"].max() < 1e3


def test_scores_satisfy_contract():
    from src.contracts.scores_v0 import validate_scores_df

    validate_scores_df(compute_rel_pos_scores(_contract_frame()))
```

**Step 2: 运行确认失败**

```bash
conda run -n interface pytest tests/test_score_trivial_baselines.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.score_trivial_baselines'`

**Step 3: 实现**

`scripts/score_trivial_baselines.py`，docstring 与关键约束：

```python
"""平凡基线打分器：产出契约合规的 scores.parquet，走现有 train↔eval 接口墙。

entry 027 实测这两个不含任何学习过程的基线打败了全部六种融合机制
（rel_pos per-case macro 0.9656 / zscore_l2 0.9415 vs 最好的 DWF 0.9238）。
把它们固化成常驻 dvc stage，使以后每个实验的 metrics.json 旁边都有参照线
——否则"有没有真的超过平凡规则"这个判断会被遗忘（P0 首要动机）。

**lean 口径**：z-score 的 mean/std 只来自训练池行（train.parquet），不用
eval 侧留作负样本的 baseline 行。entry 027 诊断脚本里的 transductive 版
（用全 baseline 段）留在那个一次性脚本里作附注，不进本 stage——拿一个偷看过
eval 的参照来审判模型，赢输都说不清。
"""
```

实现约束（逐条落实，每条都对应一个已踩过的坑）：

1. `rel_pos` = 窗口在 case 内的序号 ÷ (窗口总数 − 1)，按 `case_id` 分组、`timestamp_window_ms` 排序。case 只有 1 窗时置 `0.0`（避免除零）。
2. z-score：`fit_df` 按 `(case_id, endpoint_key)` 分组算 mean/std；某组在 `fit_df` 里不存在或零方差时，std 用 **1.0 哨兵**（不是 `1e-9`——那是 entry 013 的 1e9 爆值与 entry 027 的 RG sigmoid 饱和的共同根因）。
3. `agg="l2"` → `sqrt(sum(z**2))`；`agg="max"` → `max(|z|)`。NaN 特征按 0 偏离处理（缺测不等于异常，与 `contract_dataloader` 的均值填补精神一致）。
4. 输出列严格 `["sample_id", "score", "y_true"]`，`y_true` 取 `is_endpoint_anomaly`，过 `validate_scores_df` 后落盘。
5. 特征列从 `<contract_dir>/schema.json` 的 `feature_groups` 派生，**不按列名前缀扫描**（`build_contract.py` 曾因前缀扫描产生 schema 维度错位，见 PR #25）。

CLI：`--contract-dir`、`--baseline {rel_pos,zscore_l2,zscore_max}`、`--out`。z-score 的 fit 源固定读 `<contract_dir>/train.parquet`（该文件即训练池，天然满足 D2 不变量）。

**Step 4: 运行确认通过**

```bash
conda run -n interface pytest tests/test_score_trivial_baselines.py -v
```
Expected: 4 passed

**Step 5: 在真实 contract 上冒烟并核对 entry 027 数字**

```bash
conda run -n interface python scripts/score_trivial_baselines.py \
  --contract-dir artifacts/contract_new_merge_expanded \
  --baseline rel_pos --out /tmp/tb_rel_pos.parquet
conda run -n interface python scripts/eval_baseline_v0.py \
  --scores /tmp/tb_rel_pos.parquet --out /tmp/tb_rel_pos_metrics.json
conda run -n interface python -c "
import json; m=json.load(open('/tmp/tb_rel_pos_metrics.json'))
print('per_case_macro', m['per_case_auroc_macro'], '| pooled', m['auroc'])"
```
Expected: per-case macro ≈ **0.9656**（entry 027 记录值）。**若偏差 >0.01，停下来排查**——要么实现有别，要么 entry 027 的数字口径需要重新核对；不要带着不一致往下做。

注：`zscore_l2` 的 lean 版数字**预期低于** entry 027 的 0.9415（那是 transductive 版，用了全 baseline 段）。这个差值本身值得记进 entry——它量化了"参照系是否偷看 eval"的代价。

**Step 6: 提交**

```bash
git add scripts/score_trivial_baselines.py tests/test_score_trivial_baselines.py
git commit -m "[Feature]: 平凡基线 scores 生产者（rel_pos / zscore_l2 / zscore_max）

entry 027 P0：把打败全部融合机制的两个零参数基线固化为契约合规的 scores
生产者，走现有接口墙由 eval_baseline_v0.py 原样消费，eval 脚本无需读
contract（不打洞）。

z-score 用 lean 口径——统计量只取 train.parquet（训练池）行，不用 eval 侧
baseline 负样本，守住'fit 集合 ⊆ 训练池'不变量。退化列 std 用 1.0 哨兵而非
1e-9：后者是 history 013 的 1e9 爆值与 entry 027 RG sigmoid 饱和的共同根因。

rel_pos 在 contract_new_merge_expanded 上复现 entry 027 的 0.9656。"
```

## Task 6: 注册 dvc stage

**Files:**
- Modify: `dvc_new_merge/dvc.yaml`

**Step 1: 加 6 个 stage**

3 个基线 × (score + eval)，全部消费 `artifacts/contract_new_merge_expanded`，产物落 `artifacts/trivial_baseline_<name>/`。照现有 stage 的写法（`cmd` / `deps` / `outs` / `metrics` 带 `cache: false`）。

**Step 2: 跑通**

```bash
conda run -n interface dvc repro dvc_new_merge/dvc.yaml \
  -s score_trivial_rel_pos eval_trivial_rel_pos
```
Expected: 两个 stage 成功，`artifacts/trivial_baseline_rel_pos/metrics.json` 的 `per_case_auroc_macro` ≈ 0.9656

**Step 3: 提交 + 写 entry 029 + 提 PR**

entry 029 记录：per-case macro 升主指标的理由、三条基线的 lean 数字、lean vs transductive 的差值、以及"以后每个实验必须并列这三个数字"的约定。

---

# PR-3：fraction 0.0 对照实验

## Task 7: 建对照 config 并重建 contract

**Files:**
- Create: `configs/contract/v1_new_merge_inject0.yaml`

复制 `v1_new_merge.yaml`，仅把 `fault_inject_nontarget_train_fraction` 改成 `0.0`，头部注释写明用途与对照方法。

```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v1_new_merge_inject0.yaml \
  --dataset configs/data/new_merge.yaml \
  --out-dir artifacts/contract_new_merge_inject0 --seed 42
```

Expected（**逐项核对，不符说明 config 没生效**）：`train.parquet` 的 `source_phase` 计数里 `fault_inject_nontarget` = **0**（v1 为 5380），train 总行数从 9513 降到约 4133，eval_all 从 14186 升到约 19566。

## Task 8: 跑 L0/DWF × 4 seed 并做共有子集对照

**关键：不能直接比两个口径的 AUROC。** fraction 1.0→0.0 让 5380 行从训练池回到 eval_all，两版 eval 集不同（CLAUDE.md 已记录的坑）。

**Files:**
- Create: `scripts/compare_train_pool_ablation.py`（一次性分析脚本，不进 dvc，沿用 `analyze_temporal_confounding.py` 惯例）

脚本必须：
1. 求两版 `eval_all.parquet` 的 `sample_id` 交集
2. 两版 scores 都**限制到该交集**后再算 per-case macro AUROC
3. 同时报告全量数字并显式标注"eval 集不同、仅供参考"

跑法：L0 与 DWF 各 4 seed，两个口径共 16 次训练+评估，产物落 `artifacts/baseline_new_merge_inject0_<fusion>_seed<s>/`。

**验收**：报告里必须同时出现"共有子集口径"与"全量口径"两组数字，并给出结论——inject 非目标行吸收究竟是净收益还是净损害。

## Task 9: 写 entry 030 + 提 PR

---

# PR-4：contract v2 主线

## Task 10: ContractConfig 支持 v2 归一化枚举

**Files:**
- Modify: `src/contracts/contract_config.py:18-20`（`_VALID_NORMALIZATIONS` 硬编码 3 元 frozenset）
- Test: `tests/test_contract_config.py`

**Step 1: 写失败测试**

```python
def test_per_case_zscore_normalization_accepted():
    spec = ModalitySpec(
        preprocessor="X", preprocessor_version="1",
        features=("a",), normalization="per_case_endpoint_z_score",
    )
    assert spec.normalization == "per_case_endpoint_z_score"


def test_unknown_normalization_still_rejected():
    with pytest.raises(ValueError, match="未知 normalization"):
        ModalitySpec(
            preprocessor="X", preprocessor_version="1",
            features=("a",), normalization="per_case_endpoint_minmax_typo",
        )
```

**Step 2: 运行确认失败** → **Step 3:** 把 `_VALID_NORMALIZATIONS` 扩为 5 元，加 `per_case_endpoint_z_score`、`per_case_service_z_score`，注释说明这两个只被 v2 消费 → **Step 4:** 测试通过 → **Step 5:** 提交。

## Task 11: Normalizer per-case z-score + 三级 shrinkage

**这是本 PR 的核心任务。**

**Files:**
- Modify: `src/data/normalization.py`
- Test: `tests/test_normalization.py`

**Step 1: 写失败测试（6 个）**

```python
# 1. per-case 分组正确性：同 endpoint 不同 case 的统计量互不影响
# 2. z-score 数学正确性：手算 (x - mean) / std 逐值比对
# 3. 三级 shrinkage：n < k 时向"该 endpoint 跨 case 汇总"收缩，
#    w = n/(n+k)，k=10，手算比对
# 4. 零方差回退链：组内 std≈0 → endpoint 级 std → 全局 std → **1.0 哨兵**。
#    断言输出 |z| < 1e3（不得出现 1e9）
# 5. 未知 group（fit 时未见过的 case）→ 回退到 endpoint 级统计量，不抛异常、
#    不留原始量纲（per-case 下原始值跨 case 不可比，透传是错的）
# 6. save/load roundtrip：含三级统计量的 JSON 往返后 transform 结果逐值相同
```

**Step 2: 运行确认失败**

**Step 3: 实现要点**

- `Scope` Literal 加 `"per_case_endpoint"` / `"per_case_service"`；`Method` Literal 加 `"z_score"`
- `_GROUP_COL`（`:13-17`）当前是 `scope → 单列名`，per-case 需要**两列组合键**：改为 `scope → tuple[str, ...]`，JSON 序列化时把组合键 join 成 `"case::endpoint"` 字符串（保持 save/load 格式可读）
- `_Stats` 增加两个 fallback 层（`by_endpoint` 跨 case 汇总、`global_stat`）与每组样本数 `n`（shrinkage 需要）
- shrinkage 公式（写进代码注释说明选择理由）：
  ```
  w = n / (n + k),  k = 10
  mean_eff = w * mean_group + (1 - w) * mean_fallback
  std_eff  = w * std_group  + (1 - w) * std_fallback
  ```
- **哨兵是 1.0 而非 1e-9**，注释写明这是 entry 013 的 1e9 爆值与 entry 027 的 RG sigmoid 饱和（z 爆到 1e9 → logits 1e8 → 梯度消失）的共同根因。这是本次实现相对旧代码的实质改进。
- 既有三个 scope（`global`/`per_endpoint`/`per_service`）与 `min_max` 行为**逐行不变**——现有 `test_normalization.py` 全部原样通过是硬门槛

**Step 4: 运行确认通过**

```bash
conda run -n interface pytest tests/test_normalization.py -v
```
Expected: 新增 6 项 + 既有全部通过

**Step 5: 提交**

## Task 12: build_contract 重排（split 提到归一化之前）

**Files:**
- Modify: `scripts/build_contract.py`（`_SCOPE_MAP`、`main()` 的 465–556 段）
- Test: `tests/test_build_contract_v2_fit_scope.py`（新建）

**Step 1: 写失败测试 —— 泄漏断言（本 PR 最重要的测试）**

用 monkeypatch 截获 `Normalizer.fit` 的实参，断言其行集恰好等于「Normal 的 `train_fit` ∪ 各故障 case 的 baseline 前 20% 窗口」，且与最终 `train.parquet` 满足 ⊆ 关系。

```python
def test_v2_normalizer_fit_rows_subset_of_train_pool(tmp_path, monkeypatch):
    captured = {}
    orig_fit = Normalizer.fit

    def spy(self, df):
        captured["sample_ids"] = set(df["sample_id"])
        return orig_fit(self, df)

    monkeypatch.setattr(Normalizer, "fit", spy)
    # ... 用 tests 里既有的合成 case fixture 跑 build_contract 的 v2 路径
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert captured["sample_ids"] <= set(train["sample_id"])          # D2 不变量
    assert not (captured["sample_ids"] & set(eval_all["sample_id"]))  # 与 eval 无交集
```

**Step 2: 运行确认失败** → **Step 3: 实现**

- `_SCOPE_MAP` 加两个新 normalization 字符串 → `(Scope, Method)` 的映射
- `main()` 里 `contract_version == "v2"` 分支：先调 `split_fault_phase_temporal` 得出各 case 的 baseline-train 行 mask，与 Normal 的 `train_fit` 合并成 `fit_df`，再 `normalizer.fit(fit_df)`
- **切分结果必须复用**：把这次切分的产物传给 `_write_v1`，不要让它再切一次（两处独立切分 = 参数漂移风险；当前实现虽确定性，但依赖"两次调用参数恰好一致"是脆弱的）
- v2 分支跳过 rate `clip(0,1)`（`:529`）与 content_length ±5 clip（`:544-551`）
- v2 分支不 fit、不落盘 `endpoint_baseline_stats.json`
- **v1 代码路径一行不改**

**Step 4: 测试通过 + 全量回归**

```bash
conda run -n interface pytest tests/ -q
```

## Task 13: contract 校验放宽（仅 v2 分支）

**Files:**
- Modify: `src/contracts/contract_v0.py`
- Test: `tests/test_contract_v0_schema.py`

v2 下 rate 列的 `[0,1]` 校验替换为量级 sanity（`|x| ≤ 1e3`）。**保留不变**：sample_id 唯一、`phase ≡ is_anomaly`、`label_granularity` 枚举、`is_endpoint_anomaly ⇒ is_anomaly`、行守恒。

测试要同时断言"v2 允许 z-score 的负值与超 1 值"和"v2 仍然拒绝 1e9 量级"。

## Task 14: v2 配置 + 构建 + 行等价断言

**Files:**
- Create: `configs/contract/v2_new_merge.yaml`
- Test: `tests/test_contract_v1_v2_row_equivalence.py`（新建）

**Step 1: 写 config**

基于 `v1_new_merge.yaml`：`contract_version: "v2"`、`fit_endpoint_baseline_stats: false`、三个 fraction 沿用 `0.2 / 1.0 / 1.0`（保证行等价），各模态 `normalization` 改成 per-case z-score 版本。

**Step 2: 构建**

```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v2_new_merge.yaml \
  --dataset configs/data/new_merge.yaml \
  --out-dir artifacts/contract_new_merge_v2 --seed 42
```

**Step 3: 行等价断言（最强回归测试）**

```python
def test_v1_v2_eval_all_rows_and_labels_identical():
    """v2 只改归一化，split/标签/吸收闸门逻辑与 v1 逐行一致，因此 eval_all 的
    行集与全部标签列必须逐行相等。这条断言把'归一化改动'与'划分改动'彻底
    隔离——v1↔v2 的 AUROC 差异只可能来自特征值本身。"""
    v1 = pd.read_parquet("artifacts/contract_new_merge_expanded/eval_all.parquet")
    v2 = pd.read_parquet("artifacts/contract_new_merge_v2/eval_all.parquet")
    label_cols = ["sample_id", "case_id", "endpoint_key", "phase", "is_anomaly",
                  "is_endpoint_anomaly", "label_granularity", "label_target_observable"]
    pd.testing.assert_frame_equal(
        v1[label_cols].sort_values("sample_id").reset_index(drop=True),
        v2[label_cols].sort_values("sample_id").reset_index(drop=True),
    )
```

Expected: PASS。**失败即停**——说明 v2 意外改动了划分或标签逻辑，这比任何 AUROC 数字都重要。

**Step 4: 退化审计**

构建日志统计每列的 shrinkage 回退占比（group 数预期从 ~10 涨到 ~416）。回退率 >50% 的列记进 entry 并解释。

## Task 15: 自参照 DWF 新类

**Files:**
- Create: `src/fusion/deviation_weighted_selfref.py`
- Create: `configs/fusion/deviation_weighted_selfref.yaml`
- Test: `tests/test_fusion_deviation_weighted_selfref.py`

**Step 1: 写失败测试**

```python
# 1. 权重公式：手算 sigmoid((|x| - 2.0) / 1.0) * x 逐值比对
# 2. 无 sidecar 构造：只给 modality_dims 就能实例化（不需要 contract_dir、
#    不需要 endpoint_baseline_stats、不需要 red_cols/svc_cols）
# 3. endpoint_id 传 None 与传张量结果相同（本类不使用 endpoint 信息）
# 4. output_dim == ep_dim + svc_dim
# 5. 梯度通畅：本类零参数，但下游 SVDD 的梯度必须能穿过它回传
#    （entry 012 的 optimizer 漏接 fusion 参数是同源教训）
# 6. 批量向量化：大 batch 下无 per-sample Python 循环（shape 断言 + 计时兜底）
```

**Step 2: 运行确认失败** → **Step 3: 实现**

```python
class SelfReferentialDeviationFusion(FusionModule):
    """逐特征偏离量加权，零参数、零状态。

    与 DeviationWeightedFusion（entry 018）的关系：**公式完全相同**，区别只在
    偏离量从哪来。旧类查 EndpointBaselineStats sidecar 拿 per-endpoint 的
    normal-only mean/std 算 z；本类直接把输入特征当偏离量——因为 contract v2
    的 per-case z-score 归一化已使特征值本身就是「相对自身 baseline 的偏离」。

    结论因此可以写成：DWF 的公式一开始就是对的，错的是参照系（entry 027）。

    本类同时是 max|z| 平凡基线（entry 027 实测 per-case macro 0.9403）的可微
    版本——SVDD 学表征、本类做软 top-k 偏离聚合。
    """

    def __init__(self, modality_dims: dict[str, int],
                 threshold: float = 2.0, scale: float = 1.0):
        ...

    def forward(self, modality_dict, endpoint_id=None):
        x = torch.cat([modality_dict[m] for m in MODALITY_ORDER], dim=-1)
        w = torch.sigmoid((x.abs() - self._threshold) / self._scale)
        return x * w
```

不需要 `from_contract` 覆写（基类默认实现即可）。旧 `DeviationWeightedFusion` 及其 config、测试**一行不改**。

**Step 4/5: 测试通过 → 提交**

## Task 16: RG 退役 + v2 四臂 stage

**Files:**
- Modify: `dvc_new_merge/dvc.yaml`

- 移除 2 个 RG train/eval stage 对（类代码保留，git 历史即档案）
- 新增 v2 四臂 × (train + eval)：`concat` / `independent_concat` / `gated` / `deviation_weighted_selfref`，`contract_dir=artifacts/contract_new_merge_v2`
- 给 v2 口径也注册 3 条平凡基线 stage

```bash
conda run -n interface dvc repro dvc_new_merge/dvc.yaml
```

## Task 17: 四臂 × 4 seed 重跑 + 验收

seed 42 走 dvc，seed{1,2,3} 手动覆盖（沿用项目惯例，`out=artifacts/baseline_new_merge_v2_<fusion>_seed<s>/scores.parquet`）。

**验收检查清单（逐项核对，任一失败即停下报告）：**

1. **Canary**：`Lv_P_CPU_preserve` 分层 AUROC ≥ 0.9（v1 组合口径实测 **0.9995**，见 `artifacts/baseline_new_merge_concat/metrics.json`）
2. **行等价**：Task 14 的断言仍然通过
3. **泄漏**：Task 12 的 fit ⊆ train 断言通过
4. **v1 回归**：v1 口径 6 个 metrics.json 逐位不变（`git diff` 应为空）
5. **参照线齐全**：主表含 v2 四臂 × 4 seed + v1 同臂 + 3 条平凡基线，per-case macro 为主指标

**三条预期（写进 entry，避免数字被误读）：**

1. `rel_pos` 基线仍会是 ~0.9656 —— per-case 归一化治漂移与参照系，对"inject 起点固定在 rel_pos 0.586±0.015"零作用，那只能靠重采（P1）
2. 最可能的结果是 **SVDD ≈ z-score 基线** —— 同参照系下输入信息量基本一致，这是 entry 027 P2 已预演的合法结论
3. 本轮成功定义是"拿到参照线齐全、泄漏可证的可信排名"，**不是"打赢基线"**。若仍输给 lean z-score，按 P2 预案诚实转向"benchmark 缺陷 + 规范协议"的论文形态

## Task 18: 写 entry 031 + 提 PR

entry 必须记录：D1–D6 六项决策与依据、v1↔v2 主表对比、退化审计结果、canary 结果、RG 两次塌陷根因（作为负面结果归档）、`per_case_endpoint` 与 `per_case_service` 在本数据集上恰好同为 208 组（inner join 把每 service 塌成 1 endpoint，entry 017）因此**不要围绕该区分设计消融**、以及上面三条预期的实际落地情况。

---

## 非目标（本轮不做）

- 重采数据集（entry 027 P1）—— AnoMod 正在筹划投稿
- 新增任何融合机制变体（entry 027 "不要做"清单）
- DANN / gradient reversal 消除 case 身份 —— entry 027 已证伪
- 修 RG independent_sigmoid 塌陷 —— 直接退役
- 通用极端值裁剪框架（entry 026 已定案不做）
- 统一 `benchmark_downstream.py` 与 contract 两套口径（entry 028 记录风险，不在本轮解决）
