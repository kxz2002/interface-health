"""时间混淆与朴素基线诊断（entry 027）。

回答四个问题，全部只读已有 contract parquet + scores.parquet，不训练、不碰生产代码：

1. **case 身份 shortcut 有没有？** pooled AUROC vs per-case 宏平均 AUROC。
   per-case 口径的正负样本都取自同一 case，case 身份在该口径下不可利用；
   两者若接近，说明判别力不来自跨 case 基线差异（本项目实测结论：接近，gap ≈ 0.004）。

2. **时间位置 shortcut 有没有？** 用"窗口在 case 内的相对时间位置"(rel_pos) 单独当异常
   分数。采集脚本用固定时序模板（inject 恒起于 rel_pos≈0.58，std 0.015），因此这个
   不含任何观测数据的标量就能拿到高 AUROC —— 即 Wu & Keogh 的 run-to-failure bias
   在 case 内的版本。注意模型输入的 21 个特征里没有时间戳字段，所以模型在结构上
   无法利用 rel_pos；这个基线衡量的是**数据集**的缺陷，不是模型在作弊。

3. **特征是否间接携带时间信息？** GroupKFold(by case) 回归 rel_pos 的 R²/Spearman。
   若很低，说明模型确实是从特征学的（问题 2 才能被解释为纯数据集缺陷）。

4. **模型有没有挣到它的复杂度？** 两组无学习基线：
   (a) 21 个单特征各自当分数（取符号最优）；
   (b) 每个 (case, endpoint) 用**自己 baseline 阶段**的 mean/std 算 z-score，
       再取 max|z| / L2 范数 / mean|z| 聚合。
   (b) 对采集漂移天然免疫（参照系是 case 自身而非全局 Normal），是 AIOps 的标准做法
   （StepWise、DCASE per-section 同思路），也正是 Quo Vadis (ICML'24) 的 L2-norm 基线。

与 analyze_gate_weights.py / analyze_modality_observability.py 同类：一次性诊断脚本，
不进 dvc pipeline（只读、不产出被下游 stage 消费的产物），调用方式记在 CLAUDE.md。

用法：
    python scripts/analyze_temporal_confounding.py \
        --contract-dir artifacts/contract_new_merge_expanded \
        --scores-dir artifacts \
        --out artifacts/temporal_confounding/report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

# 与 dvc_new_merge/dvc.yaml 里六个 train stage 的输出目录对应
SCORE_RUNS = {
    "concat (L0)": "baseline_new_merge_concat",
    "independent_concat (L1)": "baseline_new_merge_independent_concat",
    "gated (L2)": "baseline_new_merge_gated",
    "reliability_gate softmax (RG)": "baseline_new_merge_reliability_gate",
    "reliability_gate indep_sigmoid": "baseline_new_merge_reliability_gate_indep_sigmoid",
    "deviation_weighted (DWF)": "baseline_new_merge_deviation_weighted",
}
LABEL_COL = "is_endpoint_anomaly"
# baseline 阶段至少要有这么多窗口才够估 mean/std；低于此则该 (case,endpoint) 不参与 z-score
MIN_BASELINE_WINDOWS = 5
PROBE_MAX_ITER = 150
PROBE_SEED = 0


def _add_rel_pos(df: pd.DataFrame) -> pd.DataFrame:
    """每行在其所属 case 的采集窗口内的相对时间位置，归一到 [0, 1]。"""
    g = df.groupby("case_id")["timestamp_window_ms"]
    lo = g.transform("min")
    span = g.transform("max") - lo
    # 单窗口 case 的 span 为 0，置 0.0 而非 NaN，避免污染下游 AUROC
    df["rel_pos"] = np.where(span > 0, (df["timestamp_window_ms"] - lo) / span, 0.0)
    return df


def _macro_auroc(scores: pd.Series, labels: pd.Series, groups: pd.Series) -> tuple[float, int]:
    """per-case 宏平均 AUROC：每个 case 内单独算再取均值。

    单一类别的 case（如 Normal，或 label_target_observable=False 导致 n_pos=0 的 case）
    AUROC 无定义，直接跳过 —— 与 eval_baseline_v0.py 对分层 AUROC 产出 null 的惯例一致。
    """
    per = []
    for idx in groups.groupby(groups).groups.values():
        y = labels.loc[idx]
        if y.nunique() < 2:
            continue
        s = scores.loc[idx]
        if s.isna().all():
            continue
        per.append(roc_auc_score(y, s.fillna(0.0)))
    return (float(np.mean(per)), len(per)) if per else (float("nan"), 0)


def _collection_order(case_ids: pd.Series) -> pd.DataFrame:
    """从 case_id 里的 ISO 时间戳还原串行采集顺序。"""
    rows = []
    for cid in case_ids.drop_duplicates():
        m = re.search(r"(\d{8}T\d{6})Z", cid)
        rows.append({"case_id": cid, "collected_at": m.group(1) if m else None})
    out = pd.DataFrame(rows).sort_values("collected_at").reset_index(drop=True)
    out["order"] = range(len(out))
    return out


def _zscore_vs_own_baseline(ev: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """每个 (case, endpoint) 用自己 baseline/normal 阶段的 mean/std 做 z-score。

    这是无学习基线的核心：参照系是 case 自身而非全局 Normal case，因此采集漂移
    在分子里自动抵消。零方差列 std 置 NaN（而非兜底 epsilon），让该列的 z-score 变
    NaN 后在聚合时被 skipna 忽略 —— 与 Normalizer 跳过退化 group 的既有决策同向
    （entry 007/013），避免把噪声放大成 1e9 量级的假信号。
    """
    z = pd.DataFrame(index=ev.index, columns=feats, dtype=float)
    for _, g in ev.groupby(["case_id", "endpoint_key"], sort=False):
        base = g[g["phase"].isin(["baseline", "normal"])]
        if len(base) < MIN_BASELINE_WINDOWS:
            continue
        mu = base[feats].mean()
        sd = base[feats].std().replace(0.0, np.nan)
        z.loc[g.index, :] = ((g[feats] - mu) / sd).values
    return z


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-dir", type=Path, required=True)
    ap.add_argument("--scores-dir", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    schema = json.loads((args.contract_dir / "schema.json").read_text())
    feats = [c for grp in schema["feature_groups"].values() for c in grp["columns"]]

    ev = _add_rel_pos(pd.read_parquet(args.contract_dir / "eval_all.parquet"))
    y = ev[LABEL_COL].astype(int)
    lines: list[str] = ["# 时间混淆与朴素基线诊断（entry 027）", ""]

    order = _collection_order(ev["case_id"])
    is_normal = order["case_id"].str.startswith("Normal")
    lines += [
        "## 1. 串行采集布局",
        "",
        f"- case 数：{len(order)}，其中 Normal case {int(is_normal.sum())} 个",
        f"- 采集跨度：{order['collected_at'].iloc[0]} → {order['collected_at'].iloc[-1]}",
        f"- Normal case 采集序号：{order.loc[is_normal, 'order'].tolist()}",
        "",
        "```",
        order.to_string(index=False),
        "```",
        "",
    ]

    inj = ev[ev["phase"] == "inject"].groupby("case_id")["rel_pos"].agg(["min", "max", "count"])
    lines += [
        "## 2. inject 阶段的时间位置一致性",
        "",
        "跨 case 的 inject 起止 rel_pos 分布（std 越小 → 固定时序模板越强 → shortcut 越强）：",
        "",
        "```",
        inj.describe().round(4).to_string(),
        "```",
        "",
    ]

    lines += ["## 3. pooled vs per-case 宏平均（case 身份 shortcut 检验）", ""]
    rows = []
    for name, sub in SCORE_RUNS.items():
        p = args.scores_dir / sub / "scores.parquet"
        if not p.exists():
            logger.warning("缺 %s，跳过", p)
            continue
        s = pd.read_parquet(p)
        sy = s[LABEL_COL].astype(int)
        mac, n = _macro_auroc(s["score"], sy, s["case_id"])
        pooled = roc_auc_score(sy, s["score"])
        rows.append(
            {
                "model": name,
                "pooled_auroc": pooled,
                "macro_case_auroc": mac,
                "gap": pooled - mac,
                "pooled_auprc": average_precision_score(sy, s["score"]),
                "n_valid_case": n,
            }
        )
    model_tbl = pd.DataFrame(rows)
    lines += [
        "```",
        model_tbl.round(4).to_string(index=False),
        "```",
        "",
        "gap ≈ 0 说明判别力不来自跨 case 基线差异（per-case 口径下 case 身份不可利用）。",
        "",
    ]

    lines += ["## 4. rel_pos 平凡基线（时间位置 shortcut）", ""]
    rp_macro, rp_n = _macro_auroc(ev["rel_pos"], y, ev["case_id"])
    lines += [
        f"- 只用 rel_pos 当异常分数：pooled AUROC **{roc_auc_score(y, ev['rel_pos']):.4f}** / "
        f"per-case 宏平均 **{rp_macro:.4f}** (n={rp_n})",
        f"- pooled AUPRC {average_precision_score(y, ev['rel_pos']):.4f}",
        "",
    ]

    x = ev[feats].fillna(0.0).values
    pred = np.zeros(len(ev))
    for tr, te in GroupKFold(n_splits=5).split(x, ev["rel_pos"], groups=ev["case_id"]):
        m = HistGradientBoostingRegressor(max_iter=PROBE_MAX_ITER, random_state=PROBE_SEED)
        m.fit(x[tr], ev["rel_pos"].values[tr])
        pred[te] = m.predict(x[te])
    lines += [
        "特征是否间接携带时间位置（GroupKFold by case）：",
        "",
        f"- R² = **{r2_score(ev['rel_pos'], pred):.4f}**, "
        f"Spearman ρ = **{spearmanr(ev['rel_pos'], pred).statistic:.4f}**",
        "- 低 R² ⇒ 模型不是靠特征闻到时间，rel_pos 高分是数据集缺陷而非模型作弊",
        "",
    ]

    lines += ["## 5. 无学习基线（模型有没有挣到复杂度）", "", "### 5.1 单特征", ""]
    sf = []
    for f in feats:
        a, _ = _macro_auroc(ev[f], y, ev["case_id"])
        sf.append({"feature": f, "macro_auroc": a, "signed": max(a, 1 - a)})
    sf_tbl = pd.DataFrame(sf).sort_values("signed", ascending=False)
    lines += ["```", sf_tbl.round(4).to_string(index=False), "```", ""]

    z = _zscore_vs_own_baseline(ev, feats)
    zrows = []
    for nm, v in {
        "max|z|": z.abs().max(axis=1),
        "L2 ||z||": np.sqrt((z**2).sum(axis=1)),
        "mean|z|": z.abs().mean(axis=1),
    }.items():
        a, n = _macro_auroc(v, y, ev["case_id"])
        zrows.append({"baseline": nm, "macro_case_auroc": a, "n_valid_case": n})
    z_tbl = pd.DataFrame(zrows)
    lines += [
        "### 5.2 per-case baseline 自适应 z-score（零参数、零训练）",
        "",
        "```",
        z_tbl.round(4).to_string(index=False),
        "```",
        "",
    ]

    best_model = model_tbl["macro_case_auroc"].max() if not model_tbl.empty else float("nan")
    best_z = z_tbl["macro_case_auroc"].max()
    lines += [
        "## 结论",
        "",
        f"- 最好的学习模型 per-case 宏平均 AUROC：**{best_model:.4f}**",
        f"- 最好的零参数 z-score 基线：**{best_z:.4f}**",
        f"- rel_pos 平凡基线：**{rp_macro:.4f}**",
        "",
        f"→ 零参数基线{'超过' if best_z > best_model else '未超过'}学习模型"
        f"（差 {best_z - best_model:+.4f}）。",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入 %s", args.out)


if __name__ == "__main__":
    main()
