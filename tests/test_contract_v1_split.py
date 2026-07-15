import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.contracts.split_v1 import split_normal_rows_temporal

REPO_ROOT = Path(__file__).parents[1]

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

    schema = json.loads((out_dir / "schema.json").read_text())
    assert schema["contract_version"] == "v1"

    fit = pd.read_parquet(out_dir / "train_fit.parquet")
    val = pd.read_parquet(out_dir / "train_val.parquet")
    holdout = pd.read_parquet(out_dir / "eval_normal_holdout.parquet")
    ids = [set(d["sample_id"]) for d in (fit, val, holdout)]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()

    # 核心不变量：train.parquet 与 eval_all.parquet 的 sample_id 必须互斥。
    # 这正是 Contract v1 存在的理由（修复 v0 的 train ⊆ eval_all 泄漏）；若
    # _write_v1 回归成 eval_all = anomaly_df + 全部 normal_df，其他测试均不会
    # 发现，只有这条断言会失守。
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set(), (
        "train.parquet 与 eval_all.parquet 的 sample_id 有重叠——"
        "这正是 Contract v1 存在的理由（修复 v0 的 train ⊆ eval_all），"
        "此断言失守说明泄漏修复被破坏"
    )


def test_build_contract_v1_normalizer_excludes_holdout(tmp_path):
    """回归测试：Normalizer 必须只在 train_fit 上 fit，不能看到 eval_normal_holdout。

    直接断言 normalization_stats.json 里的 min/max 与 train_fit-only 重新算出来的
    min/max 一致；如果未来有人把 fit 范围改回全部 Normal 行（fit+holdout），
    min/max 会被 holdout 的极值拉宽，与这里的期望值不再相等，从而被本测试捕获。
    """
    out_dir = tmp_path / "contract_v1_norm"
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

    holdout = pd.read_parquet(out_dir / "eval_normal_holdout.parquet")
    assert len(holdout) > 0, "mini fixture 必须产出非空 holdout，否则本测试无法证明隔离"

    stats = json.loads((out_dir / "normalization_stats.json").read_text())
    col = "endpoint_red__trace_request_count"
    group = "POST:/api/v1/travelservice/trips/left"
    persisted_lo, persisted_hi = stats[col]["by_group"][group]

    # mini fixture 的原始 CSV 里该 endpoint 三个时间窗的 trace_request_count 依次为
    # 8/10/6（见 tests/fixtures/mini_data_root/Normal/_pipeline_out/tt_traces_red_15s.csv）。
    # fit_frac=0.6 对 3 个窗取 int(3*0.6)=1 个窗给 train_fit，即只有第一个窗（值=8）。
    # 若 fit 范围正确收窄到 train_fit，min/max 应为 [8.0, 8.0]；若退化回旧 bug
    # （在全部 Normal 行上 fit，即 8/10/6 都参与），min/max 会变成 [6.0, 10.0]——
    # 与这里的期望值不同，从而让本测试在 bug 重新引入时失败。
    assert (persisted_lo, persisted_hi) == (8.0, 8.0), (
        f"{col}/{group} 的 fit 统计量应只由 train_fit 那 1 行（值=8.0）决定，"
        f"实际 persisted=[{persisted_lo}, {persisted_hi}]，说明 fit 范围泄漏了"
        "train_fit 之外的 Normal 行（可能是 train_val 或 eval_normal_holdout）"
    )
