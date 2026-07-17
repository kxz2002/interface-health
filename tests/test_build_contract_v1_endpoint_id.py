import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.data.endpoint_baseline_stats import EndpointBaselineStats

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_endpoint_id_column_present_and_matches_sorted_mapping(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    expected_mapping = {ep: i for i, ep in enumerate(sorted(ep_to_svc.keys()))}

    for name in ("train_fit", "eval_all"):
        df = pd.read_parquet(out_dir / f"{name}.parquet")
        assert "endpoint_id" in df.columns
        assert df["endpoint_id"].equals(df["endpoint_key"].map(expected_mapping))


def test_endpoint_baseline_stats_sidecar_written_and_loadable(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    sidecar = out_dir / "endpoint_baseline_stats.json"
    assert sidecar.exists()
    stats = EndpointBaselineStats.load(sidecar)

    # sidecar 覆盖的 endpoint 必须是 train_fit 里实际出现过的 endpoint_key 子集
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    fit_endpoints = set(train_fit["endpoint_key"].unique())
    assert stats.fitted_endpoints() == fit_endpoints


def test_endpoint_id_not_added_in_v0(tmp_path):
    """v0 路径不受影响——endpoint_id 是 v1 新增列，不应该悄悄出现在 v0 产物里
    （否则 v0 的既有契约测试的列集合断言会被意外改变行为）。"""
    out_dir = tmp_path / "contract_v0"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "endpoint_id" not in train.columns
    assert not (out_dir / "endpoint_baseline_stats.json").exists()
