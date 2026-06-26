#!/usr/bin/env python3
"""eval_baseline_v0: 从 scores.parquet 计算分层 AUROC，输出 metrics.json。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_auroc(y_true, scores) -> float | None:
    """单类时返回 None 而非抛出异常。"""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def compute_stratified_metrics(df: pd.DataFrame) -> dict:
    result: dict = {}

    result["overall"] = {"auroc": _safe_auroc(df["y_true"].values, df["score"].values)}

    result["by_anomaly_type"] = {}
    for atype, grp in df.groupby("anomaly_type"):
        result["by_anomaly_type"][atype] = {
            "auroc": _safe_auroc(grp["y_true"].values, grp["score"].values),
            "n_samples": len(grp),
        }

    result["by_anomaly_level"] = {}
    if "anomaly_level" in df.columns:
        for level, grp in df.groupby("anomaly_level"):
            result["by_anomaly_level"][level] = {
                "auroc": _safe_auroc(grp["y_true"].values, grp["score"].values),
                "n_samples": len(grp),
            }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.scores)
    logger.info("Loaded %d samples from %s", len(df), args.scores)

    metrics = compute_stratified_metrics(df)
    logger.info("Overall AUROC: %s", metrics["overall"]["auroc"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", args.out)


if __name__ == "__main__":
    main()
