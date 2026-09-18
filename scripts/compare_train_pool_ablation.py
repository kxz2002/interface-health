#!/usr/bin/env python3
"""训练池 inject 非目标行吸收对照（entry 027 P0 / Task 8）。

回答一个问题：``fault_inject_nontarget_train_fraction=1.0``（expanded 口径，
5380 行 inject 阶段非目标行进 One-Class 训练池）相对 0.0（inject0 口径，
这些行整段留 eval_all）究竟是净收益还是净损害。

**为什么不能直接比两版 metrics.json**：fraction 取值不同会改变 eval_all 的
构成——expanded 版 14186 行，inject0 版 19566 行，两份 scores 的评估样本集
不同，overall/per-case AUROC 不可横向比较（CLAUDE.md Known Gotchas 已记录
此坑）。本脚本取两版 eval_all 的 ``sample_id`` 交集（Task 7 已核验 expanded
版全部 14186 行是 inject0 版的子集），把两版 scores 都限制到该交集后，复用
``scripts/eval_baseline_v0.py`` 的 ``compute_stratified_metrics`` 重算 per-case
macro AUROC，再做 4 seed 对照。

全量口径的数字也一并报告，但只作附注——两版 eval 集不同，不能据此下结论。

一次性分析脚本，不进 dvc pipeline（与 analyze_temporal_confounding.py 同类，
只读 contract parquet + scores.parquet，不训练、不碰生产代码）。

用法（确切重跑命令同时写入报告头部）：

    python scripts/compare_train_pool_ablation.py \\
        --expanded-contract artifacts/contract_new_merge_expanded \\
        --inject0-contract artifacts/contract_new_merge_inject0 \\
        --scores-root artifacts \\
        --out artifacts/train_pool_ablation/report.md
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# 以 `python scripts/xxx.py` 直接执行时 sys.path[0] 是 scripts/ 目录本身，
# 补 repo root 后 `scripts.` 命名空间包才可导入（与 tests/ 里的既有导入同源）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_baseline_v0 import compute_stratified_metrics  # noqa: E402

logger = logging.getLogger(__name__)

SEEDS = [42, 1, 2, 3]
# |delta macro AUROC| 小于此值视为两口径持平，用于结论节的符号计数
DELTA_TIE_EPS = 0.005
FUSIONS = {
    "concat (L0)": "concat",
    "deviation_weighted (DWF)": "deviation_weighted",
}


def _seed_suffix(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


@dataclass(frozen=True)
class RunPair:
    fusion_label: str
    seed: int
    expanded_path: Path
    inject0_path: Path


def _build_run_pairs(scores_root: Path) -> list[RunPair]:
    pairs: list[RunPair] = []
    for label, fusion in FUSIONS.items():
        for seed in SEEDS:
            suffix = _seed_suffix(seed)
            pairs.append(
                RunPair(
                    fusion_label=label,
                    seed=seed,
                    expanded_path=scores_root
                    / f"baseline_new_merge_{fusion}{suffix}"
                    / "scores.parquet",
                    inject0_path=scores_root
                    / f"baseline_new_merge_inject0_{fusion}{suffix}"
                    / "scores.parquet",
                )
            )
    return pairs


def _row_for_pair(pair: RunPair, common_ids: set[str]) -> dict:
    """对一个 (fusion, seed) 计算共有子集 + 全量两套指标。"""
    exp = pd.read_parquet(pair.expanded_path)
    inj = pd.read_parquet(pair.inject0_path)

    exp_common = exp[exp["sample_id"].isin(common_ids)]
    inj_common = inj[inj["sample_id"].isin(common_ids)]

    # 交集应在两版 scores 里同样齐全（Task 7 已核验 expanded eval_all ⊆ inject0）
    if len(exp_common) != len(common_ids):
        raise RuntimeError(
            f"{pair.expanded_path} 在共有子集上只有 {len(exp_common)}/{len(common_ids)} 行"
        )
    if len(inj_common) != len(common_ids):
        raise RuntimeError(
            f"{pair.inject0_path} 在共有子集上只有 {len(inj_common)}/{len(common_ids)} 行"
        )

    m_exp = compute_stratified_metrics(exp_common)
    m_inj = compute_stratified_metrics(inj_common)
    m_exp_full = compute_stratified_metrics(exp)
    m_inj_full = compute_stratified_metrics(inj)

    return {
        "fusion": pair.fusion_label,
        "seed": pair.seed,
        "exp_macro": m_exp["per_case_auroc_macro"],
        "inj_macro": m_inj["per_case_auroc_macro"],
        "delta_macro": m_inj["per_case_auroc_macro"] - m_exp["per_case_auroc_macro"],
        "exp_pooled": m_exp["auroc"],
        "inj_pooled": m_inj["auroc"],
        "exp_n_full": m_exp_full["n_samples"],
        "inj_n_full": m_inj_full["n_samples"],
        "exp_macro_full": m_exp_full["per_case_auroc_macro"],
        "inj_macro_full": m_inj_full["per_case_auroc_macro"],
        "exp_pooled_full": m_exp_full["auroc"],
        "inj_pooled_full": m_inj_full["auroc"],
    }


def _fmt_mean_std(vals: list[float]) -> str:
    return f"{np.mean(vals):.4f} ± {np.std(vals, ddof=0):.4f}"


def _git_head() -> str:
    """报告生成时的代码 commit（项目 Reproducibility 规则：实验记录须含 git commit hash）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("取 git HEAD 失败：%s", exc)
        return "unknown"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded-contract", type=Path, required=True)
    ap.add_argument("--inject0-contract", type=Path, required=True)
    ap.add_argument("--scores-root", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    exp_eval = pd.read_parquet(args.expanded_contract / "eval_all.parquet", columns=["sample_id"])
    inj_eval = pd.read_parquet(args.inject0_contract / "eval_all.parquet", columns=["sample_id"])
    exp_ids = set(exp_eval["sample_id"])
    inj_ids = set(inj_eval["sample_id"])
    # 共有子集口径的隐含前提：交集必须等于整个 expanded eval_all（Task 7 已核验
    # 14186 ⊆ 19566）。若两个 contract 版本拿反（或误用了别的数据集），交集会
    # 变成真子集，限制后的"expanded"分数将与历史口径不可比——直接失败比静默
    # 产出一张误导性对照表安全。
    if not exp_ids <= inj_ids:
        raise SystemExit(
            "共有子集口径要求 expanded eval_all ⊆ inject0 eval_all，"
            f"但 expanded 有 {len(exp_ids - inj_ids)} 行不在 inject0 eval_all 中，"
            "请确认 --expanded-contract / --inject0-contract 两个版本拿对"
        )
    common_ids = exp_ids & inj_ids
    logger.info(
        "expanded eval_all=%d, inject0 eval_all=%d, 交集=%d",
        len(exp_eval),
        len(inj_eval),
        len(common_ids),
    )

    pairs = _build_run_pairs(args.scores_root)
    missing = [p for p in pairs if not (p.expanded_path.exists() and p.inject0_path.exists())]
    if missing:
        for p in missing:
            logger.error("缺 scores：%s 或 %s", p.expanded_path, p.inject0_path)
        raise SystemExit("有 scores.parquet 缺失，先完成 8 次训练+评估")

    rows = [_row_for_pair(p, common_ids) for p in pairs]
    df = pd.DataFrame(rows)

    head_sha = _git_head()
    lines: list[str] = [
        "# 训练池 inject 非目标行吸收对照（entry 027 P0 / Task 8）",
        "",
        f"- 生成时代码 commit：`{head_sha}`",
        "",
        "重跑命令：",
        "",
        "```bash",
        f"python scripts/compare_train_pool_ablation.py \\",
        f"    --expanded-contract {args.expanded_contract} \\",
        f"    --inject0-contract {args.inject0_contract} \\",
        f"    --scores-root {args.scores_root} \\",
        f"    --out {args.out}",
        "```",
        "",
        "## 口径",
        "",
        "- expanded = `fault_inject_nontarget_train_fraction=1.0`（现行 v1_new_merge，"
        "5380 行 inject 阶段非目标行被吸进 One-Class 训练池）",
        "- inject0 = 同 fraction 退回 0.0（这些行整段留 eval_all）",
        f"- expanded eval_all {len(exp_eval)} 行 / inject0 eval_all {len(inj_eval)} 行 / "
        f"共有 sample_id 子集 **{len(common_ids)} 行**",
        "- 主指标：per-case macro AUROC（正负样本同 case，免疫 case 身份 shortcut，"
        "entry 027 起升为主指标），由 `scripts/eval_baseline_v0.py` 的 "
        "`compute_stratified_metrics` 计算，指标实现无分叉",
        "- delta = inject0 − expanded；**delta<0 表示去掉吸收后变差，即吸收是净收益**；"
        "delta>0 表示吸收是净损害",
        "- 结论按 4 seed mean±std 判断，不看单 seed（entry 025 教训：RG 单 seed 方向判反）",
        "",
        "## 1. 共有子集口径（主结论依据，两版评估样本完全相同）",
        "",
        "```",
        df[["fusion", "seed", "exp_macro", "inj_macro", "delta_macro", "exp_pooled", "inj_pooled"]]
        .rename(
            columns={
                "exp_macro": "expanded_macro",
                "inj_macro": "inject0_macro",
                "delta_macro": "delta_macro",
                "exp_pooled": "expanded_pooled",
                "inj_pooled": "inject0_pooled",
            }
        )
        .round(4)
        .to_string(index=False),
        "```",
        "",
    ]

    summary_rows = []
    for label in FUSIONS:
        sub = df[df["fusion"] == label]
        summary_rows.append(
            {
                "fusion": label,
                "expanded macro (4 seed)": _fmt_mean_std(sub["exp_macro"].tolist()),
                "inject0 macro (4 seed)": _fmt_mean_std(sub["inj_macro"].tolist()),
                "mean delta": sub["delta_macro"].mean(),
                "delta std": sub["delta_macro"].std(ddof=0),
                "expanded pooled (4 seed)": _fmt_mean_std(sub["exp_pooled"].tolist()),
                "inject0 pooled (4 seed)": _fmt_mean_std(sub["inj_pooled"].tolist()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    lines += [
        "### 4 seed 汇总（共有子集）",
        "",
        "```",
        summary.round(4).to_string(index=False),
        "```",
        "",
    ]

    full = df[
        [
            "fusion",
            "seed",
            "exp_n_full",
            "inj_n_full",
            "exp_macro_full",
            "inj_macro_full",
            "exp_pooled_full",
            "inj_pooled_full",
        ]
    ]
    full = full.rename(
        columns={
            "exp_n_full": "exp_n",
            "inj_n_full": "inj_n",
            "exp_macro_full": "expanded_macro",
            "inj_macro_full": "inject0_macro",
            "exp_pooled_full": "expanded_pooled",
            "inj_pooled_full": "inject0_pooled",
        }
    )
    lines += [
        f"## 2. 全量口径（附注：两版 eval 集构成不同，{len(exp_eval)} vs {len(inj_eval)} 行，"
        "**不可直接比较，仅供参考**）",
        "",
        "```",
        full.round(4).to_string(index=False),
        "```",
        "",
    ]

    # 结论按 mean delta 与 delta std 的相对大小自动生成，避免手抄数字
    concl_lines = ["## 3. 结论", ""]
    for label in FUSIONS:
        sub = df[df["fusion"] == label]
        md = sub["delta_macro"].mean()
        ds = sub["delta_macro"].std(ddof=0)
        # 符号计数：|delta|<DELTA_TIE_EPS 视为持平；让"方向跨 seed 反转"直接可读，
        # 避免只凭均值下结论（entry 025：RG 单 seed 方向判反）。
        n_pos = int((sub["delta_macro"] > DELTA_TIE_EPS).sum())
        n_neg = int((sub["delta_macro"] < -DELTA_TIE_EPS).sum())
        n_tie = int(len(sub) - n_pos - n_neg)
        direction = "净损害（去掉后更好）" if md > 0 else "净收益（去掉后更差）"
        strength = "超过" if abs(md) > ds else "未超过"
        concl_lines += [
            f"### {label}",
            "",
            f"- 共有子集 per-case macro AUROC：expanded "
            f"{_fmt_mean_std(sub['exp_macro'].tolist())} → inject0 "
            f"{_fmt_mean_std(sub['inj_macro'].tolist())}，"
            f"mean delta = {md:+.4f}（delta std {ds:.4f}）",
            f"- 4 seed delta 符号：delta>0 {n_pos} 个 / ≈0（|delta|<{DELTA_TIE_EPS}）"
            f" {n_tie} 个 / <0 {n_neg} 个"
            f"{'——方向跨 seed 反转' if n_pos > 0 and n_neg > 0 else ''}",
            f"- 判定：inject 非目标行吸收在 4 seed 均值上是**{direction}**；"
            f"|mean delta| {strength} 1 个 delta std，"
            f"{'方向较稳' if abs(md) > ds else '种子间波动与均值同量级，结论需谨慎'}",
            "",
        ]

    lines += concl_lines
    lines += [
        "---",
        "",
        "判读规则：主结论只采用第 1 节（同一样本子集）；第 2 节全量数字因评估集不同，",
        "任何表面差距都可能只是构成差异，不作为依据。",
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已写入 %s", args.out)


if __name__ == "__main__":
    main()
