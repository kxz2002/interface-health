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
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
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


@pytest.fixture(scope="module")
def merged_pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e_merged")
    contract_dir = out / "contract_v0"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/merged_mini.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    return {"contract_dir": contract_dir}


@pytest.fixture(scope="module")
def merged_v2_pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e_merged_v2")
    contract_dir = out / "contract_v0"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/merged_v2_mini.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    return {"contract_dir": contract_dir}


def test_merged_v2_normal_case_count(merged_v2_pipeline_out):
    """V9: merged_v2 场景下 Normal case 数为 2，均来自 normal_v2_root。"""
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    assert (train_df["anomaly_type"] == "Normal").all()
    normal_case_ids = set(train_df["case_id"])
    assert normal_case_ids == {"normal_0711_30_mini", "normal_0711_60_mini"}


def test_merged_v2_total_case_count(merged_v2_pipeline_out):
    """V10: merged_v2 场景下全量 case 数为 4（1 case 级 + 1 endpoint 级 + 2 Normal）。"""
    eval_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    all_case_ids = set(eval_df["case_id"]) | set(train_df["case_id"])
    assert all_case_ids == {
        "Lv_P_DISKIO_preserve",
        "Lv_E_HTTPABORT_assurance_mini",
        "normal_0711_30_mini",
        "normal_0711_60_mini",
    }


def test_merged_v2_archived_case_excluded(merged_v2_pipeline_out):
    """V11: 归档的 Normal_old 不出现在任何输出 case_id 中。"""
    eval_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    all_case_ids = set(eval_df["case_id"]) | set(train_df["case_id"])
    assert "Normal_old" not in all_case_ids


def test_endpoint_level_case_in_eval_and_normal_source(merged_pipeline_out):
    """V8: 合并后 eval_all 含 endpoint 级 mini case；train(Normal) 不含它。"""
    eval_df = pd.read_parquet(merged_pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(merged_pipeline_out["contract_dir"] / "train.parquet")
    assert "Lv_E_HTTPABORT_travel_mini" in set(eval_df["case_id"])
    assert (train_df["anomaly_type"] == "Normal").all()
    assert "Lv_E_HTTPABORT_travel_mini" not in set(train_df["case_id"])


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
                "--dataset",
                str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
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
