import pandas as pd

from src.contracts.split_v1 import split_normal_rows_temporal

# 合成两个 Normal case，模拟真实结构：每个 case 多个时间窗，每窗含全部 8 endpoint。
# 真实数据两 case 各 108/156 窗（已实测），这里各用 100 窗，够切且比例干净。
_ENDPOINTS = [f"ep{i}" for i in range(8)]
_CASES = {"NormalA": 100, "NormalB": 100}


def _synth_normal_df() -> pd.DataFrame:
    rows = []
    for case_id, n_win in _CASES.items():
        for w in range(n_win):
            ts = 1_000 + w * 15_000  # 15s 一窗，单调递增
            for ep in _ENDPOINTS:
                rows.append(
                    {
                        "sample_id": f"{case_id}__{ep}__{ts}",
                        "case_id": case_id,
                        "endpoint_key": ep,
                        "timestamp_window_ms": ts,
                        "feat": float(w),
                    }
                )
    return pd.DataFrame(rows)


def test_split_three_way_sample_id_disjoint():
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    ids = {k: set(v["sample_id"]) for k, v in parts.items()}
    assert ids["train_fit"] & ids["train_val"] == set()
    assert ids["train_fit"] & ids["eval_normal_holdout"] == set()
    assert ids["train_val"] & ids["eval_normal_holdout"] == set()
    total = sum(len(v) for v in parts.values())
    assert total == len(_synth_normal_df())  # 无行丢失


def test_all_endpoints_present_in_every_split():
    """时序切分的核心目的：每个 endpoint 在三份里都出现，不因切分整体消失
    （这正是相对 service 切分的改进——避免 eval 出现训练时没见过的 endpoint）。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    for name, df in parts.items():
        assert set(df["endpoint_key"].unique()) == set(_ENDPOINTS), f"{name} 缺 endpoint"


def test_temporal_no_overlap_per_case():
    """每个 case 内：train_fit 的窗全部早于 train_val，train_val 全部早于 holdout。
    时序分离是"评估样本训练时未见过"的保证来源。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    for case_id in _CASES:
        fit_ts = parts["train_fit"].query("case_id == @case_id")["timestamp_window_ms"]
        val_ts = parts["train_val"].query("case_id == @case_id")["timestamp_window_ms"]
        hold_ts = parts["eval_normal_holdout"].query("case_id == @case_id")["timestamp_window_ms"]
        assert fit_ts.max() < val_ts.min()
        assert val_ts.max() < hold_ts.min()


def test_split_ratio_within_tolerance():
    """比例目标 60/20/20。时序切分按窗数分配，比例比 service 切分更均匀，容差 ±5 个百分点。"""
    parts = split_normal_rows_temporal(_synth_normal_df(), seed=42)
    total = sum(len(v) for v in parts.values())
    frac = {k: len(v) / total for k, v in parts.items()}
    assert abs(frac["train_fit"] - 0.60) <= 0.05
    assert abs(frac["train_val"] - 0.20) <= 0.05
    assert abs(frac["eval_normal_holdout"] - 0.20) <= 0.05


def test_too_few_windows_degrades_gracefully():
    """case 只有 1 个时间窗：切不出三份，全归 train_fit，另两个空，不抛异常。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"X__{ep}__1000" for ep in _ENDPOINTS],
            "case_id": "X",
            "endpoint_key": _ENDPOINTS,
            "timestamp_window_ms": 1000,
            "feat": 1.0,
        }
    )
    parts = split_normal_rows_temporal(df, seed=42)
    assert len(parts["train_fit"]) == len(_ENDPOINTS)
    assert len(parts["train_val"]) == 0
    assert len(parts["eval_normal_holdout"]) == 0
    assert list(parts["train_val"].columns) == list(df.columns)


import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_build_contract_v1_smoke(tmp_path):
    out_dir = tmp_path / "contract_v1"
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
    for name in ("train_fit", "train_val", "eval_normal_holdout", "train", "eval_all"):
        assert (out_dir / f"{name}.parquet").exists(), f"{name}.parquet 缺失"

    import json

    schema = json.loads((out_dir / "schema.json").read_text())
    assert schema["contract_version"] == "v1"

    fit = pd.read_parquet(out_dir / "train_fit.parquet")
    val = pd.read_parquet(out_dir / "train_val.parquet")
    holdout = pd.read_parquet(out_dir / "eval_normal_holdout.parquet")
    ids = [set(d["sample_id"]) for d in (fit, val, holdout)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()
