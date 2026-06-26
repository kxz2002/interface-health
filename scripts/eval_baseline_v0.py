#!/usr/bin/env python3
"""eval_baseline_v0: 从 scores.parquet 计算分层 AUROC，输出 metrics.json。"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.contracts.metrics_v0 import validate_metrics_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_auroc(y_true, scores) -> float | None:
    """单类时返回 None 而非抛出异常。"""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _safe_auprc(y_true, scores) -> float | None:
    """单类时返回 None 而非抛出异常。"""
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def compute_stratified_metrics(df: pd.DataFrame) -> dict:
    stratified: dict = {}

    stratified["overall"] = {"auroc": _safe_auroc(df["y_true"].values, df["score"].values)}

    stratified["by_anomaly_type"] = {}
    for atype, grp in df.groupby("anomaly_type"):
        stratified["by_anomaly_type"][atype] = {
            "auroc": _safe_auroc(grp["y_true"].values, grp["score"].values),
            "n_samples": len(grp),
        }

    stratified["by_anomaly_level"] = {}
    if "anomaly_level" in df.columns:
        for level, grp in df.groupby("anomaly_level"):
            stratified["by_anomaly_level"][level] = {
                "auroc": _safe_auroc(grp["y_true"].values, grp["score"].values),
                "n_samples": len(grp),
            }

    auroc = _safe_auroc(df["y_true"].values, df["score"].values)
    auprc = _safe_auprc(df["y_true"].values, df["score"].values)
    # metrics_v0 契约允许 has_labels=False：单类数据无法计算 AUROC/AUPRC，
    # 此时 auroc/auprc 均为 None，契约要求两者也为 None，逻辑自洽。
    has_labels = auroc is not None

    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": len(df),
        "has_labels": has_labels,
        "auroc": auroc,
        "auprc": auprc,
        "stratified": stratified,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    df = pd.read_parquet(args.scores)
    logger.info("Loaded %d samples from %s", len(df), args.scores)

    metrics = compute_stratified_metrics(df)
    logger.info("Overall AUROC: %s", metrics["stratified"]["overall"]["auroc"])

    validate_metrics_dict(metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics written to %s", args.out)


if __name__ == "__main__":
    main()
