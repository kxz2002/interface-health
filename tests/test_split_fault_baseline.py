"""split_fault_baseline_temporal 的单元测试：两路时序切分故障 baseline 行。"""

from __future__ import annotations

import pandas as pd

from src.contracts.split_fault_baseline import split_fault_baseline_temporal

_ENDPOINTS = [f"ep{i}" for i in range(4)]


def _synth_baseline_df(windows_per_case: dict[str, int]) -> pd.DataFrame:
    """合成"故障 case 的 baseline 阶段行"：每个 case 若干时间窗，每窗含全部 endpoint。"""
    rows = []
    for case_id, n_win in windows_per_case.items():
        for w in range(n_win):
            ts = 1_000 + w * 15_000  # 15s 一窗，单调递增
            for ep in _ENDPOINTS:
                rows.append(
                    {
                        "sample_id": f"{case_id}__{ep}__{ts}",
                        "case_id": case_id,
                        "endpoint_key": ep,
                        "timestamp_window_ms": ts,
                        "phase": "baseline",
                        "feat": float(w),
                    }
                )
    return pd.DataFrame(rows)


def test_split_disjoint_and_lossless():
    df = _synth_baseline_df({"F1": 5, "F2": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert set(train["sample_id"]) & set(eval_["sample_id"]) == set()
    assert len(train) + len(eval_) == len(df)


def test_fraction_0_2_takes_earliest_one_of_five_windows():
    """5 窗 × fraction=0.2 → int(5*0.2)=1 窗进 train，其余 4 窗进 eval。
    且进 train 的必须是最早那一窗（时序切分）。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    train_windows = sorted(train["timestamp_window_ms"].unique())
    eval_windows = sorted(eval_["timestamp_window_ms"].unique())
    assert len(train_windows) == 1
    assert len(eval_windows) == 4
    assert max(train_windows) < min(eval_windows)  # train 全部早于 eval


def test_window_not_split_across_train_and_eval():
    """同一时间窗的所有 endpoint 行必须整体归属同一侧，不能拆散。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    train_windows = set(train["timestamp_window_ms"])
    eval_windows = set(eval_["timestamp_window_ms"])
    assert train_windows & eval_windows == set()


def test_single_window_natural_truncation_no_fallback():
    """1 窗 × fraction=0.2 → int(1*0.2)=0 窗进 train，该 case 全部 baseline 行进 eval。
    刻意不做"至少 1 窗给 train"的兜底（不同于 split_normal_rows_temporal）——
    训练池已有纯 Normal 打底，某 case 贡献 0 行不影响训练可行性；强行留 1 窗反而
    会让该 case 在 eval 里的 baseline 覆盖率归零。"""
    df = _synth_baseline_df({"F1": 1})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == len(_ENDPOINTS)


def test_fraction_1_0_all_to_train():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=1.0)
    assert len(train) == len(df)
    assert len(eval_) == 0


def test_fraction_0_0_all_to_eval():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.0)
    assert len(train) == 0
    assert len(eval_) == len(df)


def test_per_case_independent_split():
    """每个 case 独立按自身窗数切分——不同 case 窗数不同时各切各的。"""
    df = _synth_baseline_df({"F1": 5, "F2": 10})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    f1_train = train[train["case_id"] == "F1"]["timestamp_window_ms"].nunique()
    f2_train = train[train["case_id"] == "F2"]["timestamp_window_ms"].nunique()
    assert f1_train == 1  # int(5*0.2)
    assert f2_train == 2  # int(10*0.2)


def test_empty_input_returns_two_empty_frames():
    """无故障 baseline 行（如数据集里没有故障 case）时返回两个空帧，不抛异常。"""
    df = pd.DataFrame(
        columns=["sample_id", "case_id", "endpoint_key", "timestamp_window_ms", "phase", "feat"]
    )
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == 0
    assert list(train.columns) == list(df.columns)
