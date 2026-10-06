#!/usr/bin/env python3
"""汇总 v2 四臂×4seed / v1 同臂×4seed / 平凡基线的 per-case macro 与 pooled 指标。

v1 seed{1,2,3} 的入库 metrics.json 是旧 eval 格式（无 per_case 键），
这里直接从 scores.parquet 用 scripts/eval_baseline_v0.py 的
compute_stratified_metrics 现算，不覆盖任何已入库文件（验收 4 要求 v1 diff 为空）。

一次性验收汇总脚本，不进 dvc pipeline（与 compare_train_pool_ablation.py 同类，
只读 scores.parquet）。输出为 JSON，打印到 **stdout**（不落盘、不打 logging）：
结果是给 main_table.md 誊数字用的结构化数据，重定向 `> file.json` 即可保存；
`artifacts/contract_v2_acceptance/main_table.md` 的数字以此脚本输出为准。
（本脚本自 outputs/v2_rerun_logs/collect_table.py 迁入——outputs/ 被 gitignore，
脚本本体随 main_table.md 引用需要入 git。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# 以 `python scripts/xxx.py` 直接执行时 sys.path[0] 是 scripts/ 目录本身，
# 补 repo root 后 `scripts.` 命名空间包才可导入（与 compare_train_pool_ablation.py 同源）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_baseline_v0 import compute_stratified_metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

V2_ARMS = ["concat", "independent_concat", "gated", "deviation_weighted_selfref"]
V1_ARMS = {
    "concat": "concat",
    "independent_concat": "independent_concat",
    "gated": "gated",
    "deviation_weighted_selfref": "deviation_weighted",
}
SEEDS = [42, 1, 2, 3]


def from_scores(path: Path) -> dict:
    df = pd.read_parquet(path)
    return compute_stratified_metrics(df)


def collect_arm(prefix: str, name: str, seeds: list[int]) -> dict:
    rows = []
    for s in seeds:
        d = f"{ART}/{prefix}{name}" + (f"_seed{s}" if s != 42 else "")
        m = from_scores(Path(d) / "scores.parquet")
        rows.append(
            {
                "seed": s,
                "per_case_auroc": m["per_case_auroc_macro"],
                "per_case_auprc": m["per_case_auprc_macro"],
                "pooled_auroc": m["auroc"],
                "pooled_auprc": m["auprc"],
                "canary_cpu_preserve": m["stratified"]["by_anomaly_type"]["Lv_P_CPU_preserve"][
                    "auroc"
                ],
            }
        )
    return rows


def stats(vals: list[float]) -> tuple[float, float]:
    import numpy as np

    return float(np.mean(vals)), float(np.std(vals, ddof=0))


def main() -> None:
    out: dict = {"v2": {}, "v1": {}, "trivial": {}}

    for arm in V2_ARMS:
        out["v2"][arm] = collect_arm("baseline_new_merge_v2_", arm, SEEDS)

    for arm2, dirname in V1_ARMS.items():
        out["v1"][arm2] = collect_arm("baseline_new_merge_", dirname, SEEDS)

    for name in [
        "trivial_baseline_rel_pos",
        "trivial_baseline_zscore_l2",
        "trivial_baseline_zscore_max",
        "trivial_baseline_v2_rel_pos",
        "trivial_baseline_v2_zscore_l2",
        "trivial_baseline_v2_zscore_max",
    ]:
        m = from_scores(ART / name / "scores.parquet")
        out["trivial"][name] = {
            "per_case_auroc": m["per_case_auroc_macro"],
            "per_case_auprc": m["per_case_auprc_macro"],
            "pooled_auroc": m["auroc"],
            "pooled_auprc": m["auprc"],
            "canary_cpu_preserve": m["stratified"]["by_anomaly_type"]["Lv_P_CPU_preserve"]["auroc"],
        }

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
