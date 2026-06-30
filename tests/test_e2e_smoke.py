"""E2E smoke test：验证 raw → contract → train → eval 完整 pipeline。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    contract_dir = out / "contract_v0"
    baseline_dir = out / "baseline_v0"

    # Stage 1: build_contract
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--data-root",
            str(MINI_DATA_ROOT),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    # Stage 2: train_baseline_v0
    scores_path = baseline_dir / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            "--contract-dir",
            str(contract_dir),
            "--out",
            str(scores_path),
            "--seed",
            "42",
            "--epochs",
            "2",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    # Stage 3: eval_baseline_v0
    metrics_path = baseline_dir / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_baseline_v0.py",
            "--scores",
            str(scores_path),
            "--out",
            str(metrics_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    return {
        "contract_dir": contract_dir,
        "scores": scores_path,
        "metrics": metrics_path,
    }


def test_contract_parquet_schema_stable(pipeline_out):
    """V1: contract parquet 有固定 18-dim 特征列。"""
    schema = json.loads((pipeline_out["contract_dir"] / "schema.json").read_text())
    assert schema["feature_dim"] == 18


def test_scores_parquet_passes_contract(pipeline_out):
    """V2: scores.parquet 通过 validate_scores_df 契约校验。"""
    from src.contracts import validate_scores_df

    df = pd.read_parquet(pipeline_out["scores"])
    validate_scores_df(df)


def test_scores_has_diagnostic_columns(pipeline_out):
    """V3: scores.parquet 含诊断列供分层评估使用。"""
    df = pd.read_parquet(pipeline_out["scores"])
    assert "case_id" in df.columns
    assert "anomaly_type" in df.columns


def test_metrics_json_overall_auroc(pipeline_out):
    """V4: metrics.json 顶层有 auroc 且在 [0,1]。"""
    metrics = json.loads(pipeline_out["metrics"].read_text())
    auroc = metrics["auroc"]
    # mini data 可能全为一类，auroc 可以是 None
    if auroc is not None:
        assert 0.0 <= auroc <= 1.0


def test_metrics_json_has_stratified_keys(pipeline_out):
    """V5: metrics.json 含 stratified.by_anomaly_type / by_anomaly_level。"""
    metrics = json.loads(pipeline_out["metrics"].read_text())
    assert "stratified" in metrics
    assert "by_anomaly_type" in metrics["stratified"]
    assert "by_anomaly_level" in metrics["stratified"]


def test_metrics_json_passes_contract_validation(pipeline_out):
    """V7: metrics.json 通过 metrics contract v0 校验。"""
    from src.contracts.metrics_v0 import validate_metrics_dict

    metrics = json.loads(pipeline_out["metrics"].read_text())
    validate_metrics_dict(metrics)


def test_pipeline_reproducible(tmp_path):
    """V6: 同 seed 两次运行 scores 完全一致。"""

    def run_pipeline(out_dir):
        contract_dir = out_dir / "contract_v0"
        scores_path = out_dir / "scores.parquet"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_contract.py",
                "--config",
                str(REPO_ROOT / "configs/contract/v0.yaml"),
                "--data-root",
                str(MINI_DATA_ROOT),
                "--out-dir",
                str(contract_dir),
                "--seed",
                "42",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/train_baseline_v0.py",
                "--contract-dir",
                str(contract_dir),
                "--out",
                str(scores_path),
                "--seed",
                "42",
                "--epochs",
                "2",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )
        return pd.read_parquet(scores_path)["score"].values

    scores1 = run_pipeline(tmp_path / "run1")
    scores2 = run_pipeline(tmp_path / "run2")
    np.testing.assert_array_equal(scores1, scores2)
