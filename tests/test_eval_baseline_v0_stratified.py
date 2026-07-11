import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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
            "is_endpoint_anomaly": [0] * 10 + [1] * 10,
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
            "is_endpoint_anomaly": [0] * 10,
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


def test_stratified_layers_use_is_endpoint_anomaly_not_y_true(tmp_path):
    """构造 y_true 与 is_endpoint_anomaly 取值相反的数据，断言四层输出的数值
    对应 is_endpoint_anomaly，不是 y_true——验证标签口径切换正确。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "score": [0.1] * 10 + [0.9] * 10,
            # y_true 与 is_endpoint_anomaly 故意完全相反
            "y_true": [1] * 10 + [0] * 10,
            "is_endpoint_anomaly": [0] * 10 + [1] * 10,
            "case_id": ["Normal"] * 10 + ["Lv_E_HTTPABORT_assurance_mini"] * 10,
            "anomaly_type": ["Normal"] * 10 + ["Lv_E_HTTPABORT_assurance"] * 10,
            "anomaly_level": ["none"] * 10 + ["endpoint"] * 10,
            "endpoint_key": ["ep_a"] * 10 + ["ep_a"] * 10,
            "label_granularity": ["case"] * 10 + ["endpoint"] * 10,
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
    # score 越高越异常；用 is_endpoint_anomaly 时高分组(score=0.9)对应 is_endpoint_anomaly=1，
    # 完全可分，AUROC 应为 1.0。若代码仍误用 y_true，会因高分组 y_true=0 而得到 AUROC=0.0。
    assert metrics["stratified"]["overall"]["auroc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)


def test_by_endpoint_stratum_groups_by_endpoint_key(tmp_path):
    """by_endpoint 新增分层：按 endpoint_key 分组算 AUROC，不区分 label_granularity。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "score": [0.1, 0.9] * 10,
            "y_true": [0, 1] * 10,
            "is_endpoint_anomaly": [0, 1] * 10,
            "case_id": ["c1"] * 10 + ["c2"] * 10,
            "anomaly_type": ["Lv_X"] * 20,
            "anomaly_level": ["endpoint"] * 20,
            # 前 10 行是 endpoint A（label_granularity=endpoint），
            # 后 10 行是 endpoint B（label_granularity=case，即 fallback 数据源）
            "endpoint_key": ["ep_A"] * 10 + ["ep_B"] * 10,
            "label_granularity": ["endpoint"] * 10 + ["case"] * 10,
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
    assert "by_endpoint" in metrics["stratified"]
    by_ep = metrics["stratified"]["by_endpoint"]
    # 两个 endpoint 都要出现，不因 label_granularity 不同被排除或拆分成两组
    assert set(by_ep.keys()) == {"ep_A", "ep_B"}
    assert by_ep["ep_A"]["n_samples"] == 10
    assert by_ep["ep_B"]["n_samples"] == 10
    assert by_ep["ep_A"]["auroc"] == pytest.approx(1.0)
    assert by_ep["ep_B"]["auroc"] == pytest.approx(1.0)
