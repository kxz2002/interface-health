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


# scores_v0 通用契约（src/contracts/scores_v0.py）只强制 sample_id/score/y_true，
# 因为它同时被 toy pipeline（train_baseline.py/eval.py）共用，那条链路永远不会有
# is_endpoint_anomaly 列。这里单独校验本脚本实际会读的 per-endpoint 标签列。
def compute_stratified_metrics(df: pd.DataFrame) -> dict:
    if "is_endpoint_anomaly" not in df.columns:
        raise KeyError(
            "scores.parquet 缺少 is_endpoint_anomaly 列。"
            "这通常说明 artifacts/baseline_v0/scores.parquet 是 per-endpoint-label-eval "
            "接入之前生成的旧产物——请先重跑 `dvc repro train_v0` 重新生成 scores.parquet，"
            "再运行本脚本。"
        )

    stratified: dict = {}
    label_col = df["is_endpoint_anomaly"].astype(int)

    stratified["overall"] = {"auroc": _safe_auroc(label_col.values, df["score"].values)}

    stratified["by_anomaly_type"] = {}
    for atype, grp in df.groupby("anomaly_type"):
        grp_label = grp["is_endpoint_anomaly"].astype(int).values
        stratified["by_anomaly_type"][atype] = {
            "auroc": _safe_auroc(grp_label, grp["score"].values),
            "n_samples": len(grp),
        }

    stratified["by_anomaly_level"] = {}
    if "anomaly_level" in df.columns:
        for level, grp in df.groupby("anomaly_level"):
            grp_label = grp["is_endpoint_anomaly"].astype(int).values
            stratified["by_anomaly_level"][level] = {
                "auroc": _safe_auroc(grp_label, grp["score"].values),
                "n_samples": len(grp),
            }

    # by_endpoint: 按 endpoint_key 分组，对全部数据一视同仁——不按 label_granularity
    # 过滤或拆分。precise（endpoint 级）与 fallback（case 级）标签来源的行混在同一个
    # endpoint_key 分组里统一计算，与 by_anomaly_type/by_anomaly_level 完全对称。
    stratified["by_endpoint"] = {}
    if "endpoint_key" in df.columns:
        for ep, grp in df.groupby("endpoint_key"):
            grp_label = grp["is_endpoint_anomaly"].astype(int).values
            stratified["by_endpoint"][ep] = {
                "auroc": _safe_auroc(grp_label, grp["score"].values),
                "n_samples": len(grp),
            }

    auroc = _safe_auroc(label_col.values, df["score"].values)
    auprc = _safe_auprc(label_col.values, df["score"].values)
    # metrics_v0 契约允许 has_labels=False：单类数据无法计算 AUROC/AUPRC，
    # 此时 auroc/auprc 均为 None，契约要求两者也为 None，逻辑自洽。
    has_labels = auroc is not None

    # per-case 宏平均升为主指标（entry 027 P0）：pooled AUROC 会被 case 间基线
    # 差异影响，per-case 口径的正负样本都来自同一 case，是更严格的度量。只统计
    # 同时含正负类的 case——单类 case 的 AUROC 无定义，计入会静默拉偏均值
    # （PR #26 的 benchmark 已按同一规则处理，此处保持一致以便两套口径可比）。
    case_aurocs: list[float] = []
    case_auprcs: list[float] = []
    for _case_id, sub in df.groupby("case_id"):
        y = sub["is_endpoint_anomaly"].astype(int)
        if y.nunique() < 2:
            continue
        case_aurocs.append(roc_auc_score(y, sub["score"]))
        case_auprcs.append(average_precision_score(y, sub["score"]))

    per_case_auroc_macro = float(sum(case_aurocs) / len(case_aurocs)) if case_aurocs else None
    per_case_auprc_macro = float(sum(case_auprcs) / len(case_auprcs)) if case_auprcs else None

    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": len(df),
        "has_labels": has_labels,
        "auroc": auroc,
        "auprc": auprc,
        "per_case_auroc_macro": per_case_auroc_macro,
        "per_case_auprc_macro": per_case_auprc_macro,
        "n_cases_with_both_classes": len(case_aurocs),
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
    # 末尾换行必须由生产者自带：否则 pre-commit 的 end-of-file-fixer 改写后，
    # git 里的 metrics.json 与 DVC 原生产出不同字节，dvc.lock 条目会沦为
    # md5=有换行/size=无换行的混合体（entry 029）。
    args.out.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("Metrics written to %s", args.out)


if __name__ == "__main__":
    main()
