#!/usr/bin/env python3
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

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts.scores_v0 import validate_scores_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# eval_baseline_v0.py 的分层指标硬依赖这些诊断列（is_endpoint_anomaly 缺失会
# 显式报错；case_id 支撑 per-case macro；anomaly_type 无守卫直接 groupby），
# 与 train_baseline_v0.py 的落盘列保持同款 10 列。
DIAGNOSTIC_COLUMNS = [
    "case_id",
    "endpoint_key",
    "phase",
    "anomaly_type",
    "anomaly_level",
    "label_granularity",
    "is_endpoint_anomaly",
]

_BASELINE_CHOICES = ("rel_pos", "zscore_l2", "zscore_max")


def compute_rel_pos_scores(df: pd.DataFrame) -> pd.DataFrame:
    """窗口在 case 内的相对时序位置：唯一窗口的 dense 序号 ÷ (窗口数 − 1)。

    eval_all 是 endpoint×window 粒度，同一时间窗有多行（每 endpoint 一行），
    故按 timestamp 的 dense 排名取序号，让同窗各行同分。等距 15s 窗口下与
    entry 027 诊断脚本的 min-max 归一恒等。单窗 case 置 0.0，避免除零。
    """
    pos = df.groupby("case_id")["timestamp_window_ms"].rank(method="dense").astype(int) - 1
    n_windows_minus_one = pos.groupby(df["case_id"]).transform("max")
    score = np.where(n_windows_minus_one > 0, pos / n_windows_minus_one, 0.0)
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "score": score.astype(float),
            "y_true": df["is_endpoint_anomaly"].astype(int),
        }
    )


def compute_zscore_scores(
    df: pd.DataFrame, fit_df: pd.DataFrame, feature_cols: list[str], agg: str
) -> pd.DataFrame:
    """逐特征 z-score 后聚合：agg="l2" → sqrt(Σz²)；agg="max" → max|z|。

    统计量只来自 fit_df（调用方固定传 train.parquet），按 (case_id, endpoint_key)
    分组——per-case 自适应参照系，采集漂移在分子里抵消。退化规则：

    - 组存在但零方差：std 置 **1.0 哨兵**（保留原始量纲）。绝不能用 1e-9 兜底：
      eval 侧任何非零差值会被放大 1e9 倍（history 013 的爆值与 entry 027 的 RG
      sigmoid 饱和同根因）。
    - 组在 fit_df 里不存在（如故障 case 的 baseline 行整窗未被训练池吸收）：
      没有参照就不判异常，该组 z 为 NaN，最终按 0 偏离处理。
    - 特征本身 NaN（缺测）同样按 0 偏离——缺测不等于异常，与 contract_dataloader
      的均值填补精神一致。
    """
    if agg not in ("l2", "max"):
        raise ValueError(f"agg must be 'l2' or 'max', got {agg!r}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from scored frame: {missing}")
    missing_fit = [c for c in feature_cols if c not in fit_df.columns]
    if missing_fit:
        raise ValueError(f"feature columns missing from fit frame: {missing_fit}")

    keys = ["case_id", "endpoint_key"]
    means = fit_df.groupby(keys, sort=False)[feature_cols].mean().reset_index()
    stds = fit_df.groupby(keys, sort=False)[feature_cols].std().reset_index()

    mu = df[keys].merge(means, on=keys, how="left", validate="many_to_one")[feature_cols]
    sd = df[keys].merge(stds, on=keys, how="left", validate="many_to_one")[feature_cols]
    mu_arr = mu.to_numpy(dtype=float)
    sd_arr = sd.to_numpy(dtype=float)
    # NaN（组不存在）也落到 1.0；此时 mu 同为 NaN，z 仍是 NaN，下游统一归零。
    sd_arr = np.where((~np.isfinite(sd_arr)) | (sd_arr == 0.0), 1.0, sd_arr)

    x = df[feature_cols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        z = (x - mu_arr) / sd_arr
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    if agg == "l2":
        score = np.sqrt(np.sum(z**2, axis=1))
    else:
        score = np.max(np.abs(z), axis=1)

    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "score": score.astype(float),
            "y_true": df["is_endpoint_anomaly"].astype(int),
        }
    )


def _feature_cols_from_schema(schema: dict) -> list[str]:
    """从 schema.json 的 feature_groups 保序派生特征列。

    不按列名前缀扫描——build_contract.py 曾因前缀扫描产生 schema 维度错位
    （PR #25），以契约声明为唯一准绳。
    """
    cols: list[str] = []
    for group in schema["feature_groups"].values():
        cols.extend(group["columns"])
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, choices=_BASELINE_CHOICES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    contract_dir: Path = args.contract_dir
    eval_df = pd.read_parquet(contract_dir / "eval_all.parquet")
    logger.info("Loaded %d eval rows from %s", len(eval_df), contract_dir / "eval_all.parquet")

    if args.baseline == "rel_pos":
        scores = compute_rel_pos_scores(eval_df)
    else:
        train_df = pd.read_parquet(contract_dir / "train.parquet")
        schema = json.loads((contract_dir / "schema.json").read_text())
        feature_cols = _feature_cols_from_schema(schema)
        logger.info(
            "z-score lean 口径：%d 个特征，统计量仅取 train.parquet %d 行",
            len(feature_cols),
            len(train_df),
        )
        agg = "l2" if args.baseline == "zscore_l2" else "max"
        scores = compute_zscore_scores(eval_df, train_df, feature_cols, agg)

    # 3 列函数输出 + 7 个诊断列在落盘层装配成 10 列，供 eval 原样消费
    # （接口墙不打洞：eval 仍只读 scores.parquet）。
    diag = eval_df[["sample_id", *DIAGNOSTIC_COLUMNS]].copy()
    out_df = diag.merge(scores[["sample_id", "score"]], on="sample_id", validate="one_to_one")
    out_df["y_true"] = out_df["is_endpoint_anomaly"].astype(int)
    out_df = out_df[
        [
            "sample_id",
            "score",
            "y_true",
            "case_id",
            "endpoint_key",
            "phase",
            "anomaly_type",
            "anomaly_level",
            "is_endpoint_anomaly",
            "label_granularity",
        ]
    ]

    validate_scores_df(out_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    logger.info("写出 %s scores：%d 行 → %s", args.baseline, len(out_df), args.out)


if __name__ == "__main__":
    main()
