import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]


def test_deviation_weighted_selfref_e2e_on_v2_fixture(tmp_path):
    """v2 + 自参照 DWF 的端到端冒烟：v2 contract（per-case z-score、无 EBS
    sidecar）→ SelfReferentialDeviationFusion 训练 2 epoch → scores 过契约校验。

    selfref 不读 endpoint_baseline_stats / endpoint_id（v2 均不产），本冒烟锁死
    "v2 产物能被 v2 新类无改造消费"这一最低集成保证（旧 DWF 需要 sidecar，
    在 v2 contract 上构造即失败，两个融合类的 sidecar 依赖差异不可互换）。
    """
    contract_dir = tmp_path / "contract_v2"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v2_new_merge.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/nontarget_split_mini.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    scores_path = tmp_path / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={scores_path}",
            "seed=1",
            "training.epochs=2",
            "fusion=deviation_weighted_selfref",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(scores_path)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
