import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts.metrics_v0 import validate_metrics_dict

REPO_ROOT = Path(__file__).parents[1]


def test_metrics_has_stratified_keys(tmp_path):
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "score": [0.1] * 10 + [0.9] * 10,
            "y_true": [0] * 10 + [1] * 10,
            "case_id": ["Normal"] * 10 + ["Lv_P_cpu"] * 10,
            "anomaly_type": ["Normal"] * 10 + ["Lv_P_cpu"] * 10,
            "anomaly_level": ["none"] * 10 + ["performance"] * 10,
        }
    )
    scores = tmp_path / "scores.parquet"
    df.to_parquet(scores)

    out = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, "scripts/eval_baseline_v0.py", "--scores", str(scores), "--out", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    metrics = json.loads(out.read_text())
    assert "protocol_version" in metrics
    assert "auroc" in metrics
    assert "auprc" in metrics
    assert "stratified" in metrics
    assert "overall" in metrics["stratified"]
    assert "by_anomaly_type" in metrics["stratified"]
    assert "by_anomaly_level" in metrics["stratified"]
    assert "Lv_P_cpu" in metrics["stratified"]["by_anomaly_type"]
    assert "performance" in metrics["stratified"]["by_anomaly_level"]
    assert 0 <= metrics["stratified"]["overall"]["auroc"] <= 1
    validate_metrics_dict(metrics)


def test_metrics_skips_single_class_group(tmp_path):
    """某个 group 只有一类标签时跳过而不崩溃。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(10)],
            "score": [0.1] * 10,
            "y_true": [0] * 10,  # 只有正常
            "case_id": ["Normal"] * 10,
            "anomaly_type": ["Normal"] * 10,
            "anomaly_level": ["none"] * 10,
        }
    )
    scores = tmp_path / "scores.parquet"
    df.to_parquet(scores)

    out = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, "scripts/eval_baseline_v0.py", "--scores", str(scores), "--out", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    metrics = json.loads(out.read_text())
    # 单类 group 可以是 null 或 skipped，不能 KeyError / crash
    assert metrics["stratified"]["overall"]["auroc"] is None
    assert metrics["auroc"] is None
    validate_metrics_dict(metrics)
