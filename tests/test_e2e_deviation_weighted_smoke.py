import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]


def test_deviation_weighted_e2e_on_mini_fixture(tmp_path):
    # 复用 RG e2e 冒烟测试同款 fixture 选择理由（见 test_e2e_reliability_gate_smoke.py
    # 的详细注释）：DeviationWeightedFusion 同样需要 EndpointBaselineStats 覆盖
    # eval_all 里出现的全部 endpoint，nan_propagation_mini.yaml 的 Normal/故障 case
    # 共用同一 endpoint，不会触发稀疏 endpoint 覆盖不全的 KeyError 边界情形。
    contract_dir = tmp_path / "contract_v1"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1_expanded_pool.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/nan_propagation_mini.yaml"),
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
            "fusion=deviation_weighted",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(scores_path)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
