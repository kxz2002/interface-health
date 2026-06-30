import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


def _build_contract(contract_dir: Path) -> None:
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
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_train_baseline_v0_writes_scores_contract(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            "--contract-dir",
            str(contract_dir),
            "--out",
            str(out),
            "--seed",
            "42",
            "--epochs",
            "2",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(out)
    validate_scores_df(df)
    assert "case_id" in df.columns
    assert "anomaly_type" in df.columns
    # 行数应与 eval_all 一致（每个评估样本输出一个 score）
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)


def test_train_baseline_v0_is_reproducible(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out1 = tmp_path / "scores1.parquet"
    out2 = tmp_path / "scores2.parquet"
    for out in (out1, out2):
        subprocess.run(
            [
                sys.executable,
                "scripts/train_baseline_v0.py",
                "--contract-dir",
                str(contract_dir),
                "--out",
                str(out),
                "--seed",
                "42",
                "--epochs",
                "2",
            ],
            check=True,
            cwd=str(REPO_ROOT),
        )

    df1 = pd.read_parquet(out1).sort_values("sample_id").reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_series_equal(df1["score"], df2["score"])
