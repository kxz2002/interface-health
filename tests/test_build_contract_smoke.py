import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts.contract_v0 import validate_contract_df

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"
CONFIG = REPO_ROOT / "configs/contract/v0.yaml"


def _run_build(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(CONFIG),
            "--data-root",
            str(MINI_DATA_ROOT),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_build_contract_produces_valid_parquet(tmp_path):
    out_dir = tmp_path / "contract_v0"
    _run_build(out_dir)

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval_all.parquet").exists()
    assert (out_dir / "normalization_stats.json").exists()
    assert (out_dir / "schema.json").exists()

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    validate_contract_df(train, str(CONFIG))
    validate_contract_df(eval_all, str(CONFIG))

    assert (train["case_id"].str.startswith("Normal")).all()
    assert eval_all["case_id"].nunique() >= 2
    # inject 窗口必须被标为异常（评估正样本存在）
    assert eval_all["is_anomaly"].any()
    # train 集只含正常样本
    assert not train["is_anomaly"].any()


def test_build_contract_is_reproducible(tmp_path):
    hashes = []
    for run in range(2):
        out_dir = tmp_path / f"run_{run}"
        _run_build(out_dir)
        hashes.append(hashlib.sha256((out_dir / "train.parquet").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]
