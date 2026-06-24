"""tests/test_e2e_toy_pipeline.py — train → eval 端到端冒烟测试。

通过 subprocess 调 CLI（而非 import 函数），是为了真实验证 dvc repro 会跑的
那条命令链，包括参数解析、退出码、文件落盘。任何一环坏掉这里都会红。

只走 toy 模式，不碰真实数据集，保证 CI 在没有 data/ 的环境下也能跑。
产物全部写入 pytest tmp_path，不污染仓库 artifacts/。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_metrics_dict, validate_scores_df

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(args: list[str]) -> subprocess.CompletedProcess:
    """在项目根目录跑脚本，捕获输出便于失败时诊断。"""
    return subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_e2e_toy_train_then_eval(tmp_path):
    scores_path = tmp_path / "scores.parquet"
    metrics_path = tmp_path / "metrics.json"

    # ---- train ----
    train = _run(
        [
            "scripts/train_baseline.py",
            "--toy",
            "--out",
            str(scores_path),
            "--seed",
            "42",
        ]
    )
    assert train.returncode == 0, f"train failed:\n{train.stderr}"
    assert scores_path.exists(), "train did not write scores.parquet"

    # 产物自身要满足 scores contract（防止 train 写出非法表却被 eval 漏过）
    scores_df = pd.read_parquet(scores_path)
    validate_scores_df(scores_df)

    # ---- eval ----
    ev = _run(
        [
            "scripts/eval.py",
            "--scores",
            str(scores_path),
            "--out",
            str(metrics_path),
        ]
    )
    assert ev.returncode == 0, f"eval failed:\n{ev.stderr}"
    assert metrics_path.exists(), "eval did not write metrics.json"

    # ---- 校验 metrics.json ----
    metrics = json.loads(metrics_path.read_text())
    validate_metrics_dict(metrics)

    assert metrics["protocol_version"] == "v0"
    assert metrics["has_labels"] is True
    assert metrics["n_samples"] > 0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0


def test_e2e_toy_is_reproducible(tmp_path):
    """同一 seed 跑两次，scores 应逐值相同——pipeline 可复现性的最小保证。"""
    out_a = tmp_path / "a.parquet"
    out_b = tmp_path / "b.parquet"

    for out in (out_a, out_b):
        proc = _run(["scripts/train_baseline.py", "--toy", "--out", str(out), "--seed", "7"])
        assert proc.returncode == 0, f"train failed:\n{proc.stderr}"

    df_a = pd.read_parquet(out_a)
    df_b = pd.read_parquet(out_b)
    pd.testing.assert_frame_equal(df_a, df_b)
