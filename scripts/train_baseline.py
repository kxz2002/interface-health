"""Toy baseline 训练脚本：仅 --toy 模式，用于打通 train → eval pipeline。

这不是一个真模型。它的存在是为了在没有任何模型代码、没有真实数据的前提下，
让 DVC pipeline 和 CI 端到端跑通：
    train_baseline.py --toy -> artifacts/scores.parquet -> eval.py -> metrics.json

未来真模型替换它时，只需替换 dvc.yaml 中 train stage 的命令；eval、tests、
contract 一行不用改——这就是 contract 的价值。

模型逻辑（玩具版本）：
- 训练集只看 normal 数据，模型 = X_train.mean(axis=0)（高斯分布的中心估计）
- 推理 score = L2(X_test - train_mu)（越远越异常 -> higher_is_more_anomalous=True）
- normal 测试样本 ~ N(0, I)、anomaly 样本 ~ N(mu_shift, I)，期望 AUROC > 0.9

CLI:
    python scripts/train_baseline.py --toy [--out PATH] [--seed INT]
                                          [--n-normal INT] [--n-anomaly INT]
                                          [--dim INT] [--anomaly-shift FLOAT]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("artifacts/scores.parquet")


def _generate_toy(
    rng: np.random.Generator,
    n_normal: int,
    n_anomaly: int,
    dim: int,
    anomaly_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成玩具数据。

    返回:
        X_train  (n_normal // 2, dim)   仅 normal，给"模型"算均值用
        X_test   (n_normal - n_normal//2 + n_anomaly, dim)  评测集
        y_test   同长度的 0/1 标签

    把 normal 切一半给训练、一半给测试，是为了避免训练样本和评测样本重叠
    导致 score=0 的退化情形。
    """
    n_train = n_normal // 2
    n_test_normal = n_normal - n_train

    X_train = rng.standard_normal(size=(n_train, dim))
    X_test_normal = rng.standard_normal(size=(n_test_normal, dim))
    X_test_anomaly = rng.standard_normal(size=(n_anomaly, dim)) + anomaly_shift

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([np.zeros(n_test_normal, dtype=int), np.ones(n_anomaly, dtype=int)])
    return X_train, X_test, y_test


def _score_l2(X: np.ndarray, train_mu: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - train_mu, axis=1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Toy baseline (placeholder) for DVC pipeline smoke."
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="must be set; v0 only supports --toy mode",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-normal", type=int, default=1000, dest="n_normal")
    parser.add_argument("--n-anomaly", type=int, default=200, dest="n_anomaly")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument(
        "--anomaly-shift",
        type=float,
        default=2.5,
        dest="anomaly_shift",
        help="mean shift of anomaly distribution; larger => easier",
    )
    args = parser.parse_args()

    if not args.toy:
        logger.error("v0 only supports --toy mode (real training not implemented)")
        return 1

    rng = np.random.default_rng(args.seed)
    X_train, X_test, y_test = _generate_toy(
        rng=rng,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        dim=args.dim,
        anomaly_shift=args.anomaly_shift,
    )

    train_mu = X_train.mean(axis=0)
    scores = _score_l2(X_test, train_mu)

    df = pd.DataFrame(
        {
            "sample_id": [f"toy__{i}" for i in range(len(scores))],
            "score": scores.astype(float),
            "y_true": y_test.astype(int),
        }
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    logger.info(
        "wrote %s | n=%d (pos=%d, neg=%d) seed=%d shift=%.2f",
        args.out,
        len(df),
        int((df.y_true == 1).sum()),
        int((df.y_true == 0).sum()),
        args.seed,
        args.anomaly_shift,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
