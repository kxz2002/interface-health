import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


def _build_contract(contract_dir: Path, config: str = "configs/contract/v0.yaml") -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / config),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _train(contract_dir: Path, out: Path, seed: int = 42, epochs: int = 2) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={out}",
            f"seed={seed}",
            f"training.epochs={epochs}",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_train_baseline_v0_writes_scores_contract(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

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
        _train(contract_dir, out)

    df1 = pd.read_parquet(out1).sort_values("sample_id").reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_series_equal(df1["score"], df2["score"])


def test_train_baseline_v0_scores_carry_endpoint_label_columns(tmp_path):
    """scores.parquet 必须显式带上 is_endpoint_anomaly / label_granularity，
    否则四层 eval 分层无法计算——这是历史上 is_target_endpoint 被静默丢弃的同一种坑。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert "is_endpoint_anomaly" in df.columns
    assert "label_granularity" in df.columns

    merged = df.merge(
        eval_all[["sample_id", "is_endpoint_anomaly", "label_granularity"]],
        on="sample_id",
        suffixes=("_out", "_src"),
    )
    assert (
        merged["is_endpoint_anomaly_out"].astype(bool)
        == merged["is_endpoint_anomaly_src"].astype(bool)
    ).all()
    assert (merged["label_granularity_out"] == merged["label_granularity_src"]).all()


def test_train_baseline_v0_works_end_to_end_on_v1_contract(tmp_path):
    """v1 训练路径此前无任何端到端覆盖——v1 专属问题（如 train_fit 太小、
    eval_all schema 漂移）此前会绿着上线。复用同一份训练脚本对 v1 contract 跑一遍。"""
    contract_dir = tmp_path / "contract_v1"
    _build_contract(contract_dir, config="configs/contract/v1.yaml")

    out = tmp_path / "scores_v1.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
