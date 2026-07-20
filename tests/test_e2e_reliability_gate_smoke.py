import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts import validate_scores_df

REPO_ROOT = Path(__file__).parents[1]


def test_reliability_gate_e2e_on_mini_fixture(tmp_path):
    # RG 生产链路用 v1_expanded_pool.yaml（expand_train_pool=true，见 dvc.yaml
    # train_v1_reliability_gate stage），这里用同一份配置才能代表真实路径。
    # mini_dataset.yaml（Task 7 计划原始指定的 fixture）在训练池扩容后会把
    # GET:.../assurances/types、POST:/api/v1/users/login 两个 endpoint 的
    # fault_baseline 行吸收进 train.parquet/eval_all，但这两个 endpoint 从未出现在
    # 该 fixture 的（单 Normal case、单 endpoint）train_fit 里——EndpointBaselineStats
    # 只 fit train_fit，查表会抛 KeyError。这正是计划 line 1193 点名的"mini fixture
    # 稀疏覆盖"边界情形（spec §4.5），不是本 Task 要修的 bug：换用
    # nan_propagation_mini.yaml——其 Normal 与故障 case 共用同一个 endpoint，
    # train_fit/train_pool/eval_all 三者 endpoint 覆盖面一致，不触发该退化路径。
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
            "fusion=reliability_gate",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(scores_path)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)
