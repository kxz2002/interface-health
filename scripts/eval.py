"""评测脚本：读取 scores.parquet → 计算 AUROC/AUPRC → 写 metrics.json。

本脚本是 train 与下游对比之间的"接口层"，不感知具体模型，
只对 scores contract v0 的产物负责。任何模型只要产出符合 contract 的
scores.parquet，都可以直接用本脚本评测。

CLI:
    python scripts/eval.py [--scores PATH] [--out PATH]

退出码：
    0  评测成功，metrics.json 已写出
    1  输入文件缺失 / contract 校验失败 / 标签退化为单类（AUROC 未定义）
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.contracts import validate_metrics_dict, validate_scores_df
from src.contracts.metrics_v0 import MetricsContractError
from src.contracts.scores_v0 import ScoresContractError

logger = logging.getLogger(__name__)

DEFAULT_SCORES = Path("artifacts/scores.parquet")
DEFAULT_OUT = Path("artifacts/metrics.json")


def _read_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.error("scores file not found: %s", path)
        sys.exit(1)
    return pd.read_parquet(path)


def _score_stats(score: pd.Series) -> dict:
    arr = score.astype(float).to_numpy()
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _label_dist(y: pd.Series) -> dict:
    y_int = y.astype(int)
    n_pos = int((y_int == 1).sum())
    n_neg = int((y_int == 0).sum())
    n = n_pos + n_neg
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": (n_pos / n) if n > 0 else 0.0,
    }


def _compute_metrics(df: pd.DataFrame) -> dict:
    y = df["y_true"].astype(int).to_numpy()
    s = df["score"].astype(float).to_numpy()

    # AUROC/AUPRC 在只有单类标签时数学上未定义，sklearn 会抛 ValueError；
    # 这里提前判断并以可读错误退出，避免栈追踪混淆调用方。
    if len(np.unique(y)) < 2:
        logger.error(
            "AUROC/AUPRC undefined: y_true has only one class. pos=%d neg=%d",
            int((y == 1).sum()),
            int((y == 0).sum()),
        )
        sys.exit(1)

    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": int(len(df)),
        "has_labels": True,
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
        "score_stats": _score_stats(df["score"]),
        "label_distribution": _label_dist(df["y_true"]),
    }


def _git_commit() -> str | None:
    """尝试拿当前 commit hash；CI 浅克隆或非 git 环境下返回 None。"""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_metrics(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 末尾换行符遵守 POSIX 文本文件惯例，避免 pre-commit end-of-file-fixer 反复修改
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Evaluate scores.parquet -> metrics.json (contract v0)"
    )
    parser.add_argument(
        "--scores", type=Path, default=DEFAULT_SCORES, help="path to scores.parquet"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="path to write metrics.json")
    args = parser.parse_args()

    df = _read_scores(args.scores)

    try:
        validate_scores_df(df)
    except ScoresContractError as exc:
        logger.error("scores contract validation failed: %s", exc)
        return 1

    metrics = _compute_metrics(df)
    metrics["meta"] = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_path": str(args.scores),
    }

    # eval 自己产物再走一遍 metrics contract，
    # 防止某天改 _compute_metrics 时悄悄破坏下游接口。
    try:
        validate_metrics_dict(metrics)
    except MetricsContractError as exc:
        logger.error("metrics contract validation failed (bug in eval.py): %s", exc)
        return 1

    _write_metrics(metrics, args.out)
    logger.info(
        "wrote %s | n=%d auroc=%.4f auprc=%.4f",
        args.out,
        metrics["n_samples"],
        metrics["auroc"],
        metrics["auprc"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
