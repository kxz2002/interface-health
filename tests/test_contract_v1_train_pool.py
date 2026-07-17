import subprocess
import sys
from pathlib import Path

import pandas as pd

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


def test_train_pool_includes_fault_baseline_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    train = pd.read_parquet(out_dir / "train.parquet")
    # mini fixture 含故障 case（如 Lv_P_DISKIO_preserve），其 baseline 阶段行
    # 现在应出现在 train.parquet 里（扩容前 train 只含 Normal case）。
    assert (
        train["anomaly_type"] != "Normal"
    ).any(), "train.parquet 未吸收任何故障 case 的 baseline 行，训练池扩容未生效"
    # 吸收的行必须确实是 baseline 阶段，不能混入 inject/recover
    non_normal = train[train["anomaly_type"] != "Normal"]
    assert (non_normal["phase"] == "baseline").all()


def test_train_pool_excludes_inject_and_recover_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert not (train["phase"] == "recover").any()
    assert not (train["phase"] == "inject").any()


def test_source_phase_column_present_and_correct(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "source_phase" in train.columns
    normal_rows = train[train["anomaly_type"] == "Normal"]
    fault_rows = train[train["anomaly_type"] != "Normal"]
    assert (normal_rows["source_phase"] == "normal_case").all()
    # mini fixture 含故障 case，fault_rows 不应为空——若为空说明训练池扩容未生效，
    # 这里必须硬断言，不能因为 len(fault_rows)==0 而静默跳过下面的校验
    assert len(fault_rows) > 0
    assert (fault_rows["source_phase"] == "fault_baseline").all()


def test_eval_all_excludes_train_pool_rows_no_leakage(tmp_path):
    """核心防泄漏不变量：train_pool 里的行（含新吸收的故障 baseline 行）
    必须同步从 eval_all 摘除，否则复现 v0 的 train⊆eval 泄漏 bug——这正是
    spec §3.4 点名要求的强制项，不是可选加固。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()

    # eval_all 不应再含任何故障 case 的 baseline 行（全部被吸收进训练池并摘除）
    fault_baseline_in_eval = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    assert len(fault_baseline_in_eval) == 0
