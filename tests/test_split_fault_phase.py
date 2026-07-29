"""split_fault_phase_temporal 的单元测试：两路时序切分故障 baseline 行。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.contracts.split_fault_phase import split_fault_phase_temporal

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
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    assert set(train["sample_id"]) & set(eval_["sample_id"]) == set()
    assert len(train) + len(eval_) == len(df)


def test_fraction_0_2_takes_earliest_one_of_five_windows():
    """5 窗 × fraction=0.2 → int(5*0.2)=1 窗进 train，其余 4 窗进 eval。
    且进 train 的必须是最早那一窗（时序切分）。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    train_windows = sorted(train["timestamp_window_ms"].unique())
    eval_windows = sorted(eval_["timestamp_window_ms"].unique())
    assert len(train_windows) == 1
    assert len(eval_windows) == 4
    assert max(train_windows) < min(eval_windows)  # train 全部早于 eval


def test_window_not_split_across_train_and_eval():
    """同一时间窗的所有 endpoint 行必须整体归属同一侧，不能拆散。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    train_windows = set(train["timestamp_window_ms"])
    eval_windows = set(eval_["timestamp_window_ms"])
    assert train_windows & eval_windows == set()


def test_single_window_natural_truncation_no_fallback():
    """1 窗 × fraction=0.2 → int(1*0.2)=0 窗进 train，该 case 全部 baseline 行进 eval。
    刻意不做"至少 1 窗给 train"的兜底（不同于 split_normal_rows_temporal）——
    训练池已有纯 Normal 打底，某 case 贡献 0 行不影响训练可行性；强行留 1 窗反而
    会让该 case 在 eval 里的 baseline 覆盖率归零。"""
    df = _synth_baseline_df({"F1": 1})
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == len(_ENDPOINTS)


def test_fraction_1_0_all_to_train():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_phase_temporal(df, fraction=1.0)
    assert len(train) == len(df)
    assert len(eval_) == 0


def test_fraction_0_0_all_to_eval():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_phase_temporal(df, fraction=0.0)
    assert len(train) == 0
    assert len(eval_) == len(df)


def test_per_case_independent_split():
    """每个 case 独立按自身窗数切分——不同 case 窗数不同时各切各的，且互相不泄漏
    （F1 的行不会混进 F2 的 train/eval，反之亦然）。"""
    df = _synth_baseline_df({"F1": 5, "F2": 10})
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    f1_train = train[train["case_id"] == "F1"]["timestamp_window_ms"].nunique()
    f2_train = train[train["case_id"] == "F2"]["timestamp_window_ms"].nunique()
    assert f1_train == 1  # int(5*0.2)
    assert f2_train == 2  # int(10*0.2)

    f1_expected_sample_ids = set(df[df["case_id"] == "F1"]["sample_id"])
    f2_expected_sample_ids = set(df[df["case_id"] == "F2"]["sample_id"])
    f1_actual = set(train[train["case_id"] == "F1"]["sample_id"]) | set(
        eval_[eval_["case_id"] == "F1"]["sample_id"]
    )
    f2_actual = set(train[train["case_id"] == "F2"]["sample_id"]) | set(
        eval_[eval_["case_id"] == "F2"]["sample_id"]
    )
    assert f1_actual == f1_expected_sample_ids
    assert f2_actual == f2_expected_sample_ids
    # 交叉验证没有串号：F1 的 sample_id 不会出现在 F2 的输出里，反之亦然
    assert f1_actual.isdisjoint(f2_expected_sample_ids - f1_expected_sample_ids)


def test_empty_input_returns_two_empty_frames():
    """无故障 baseline 行（如数据集里没有故障 case）时返回两个空帧，不抛异常。"""
    df = pd.DataFrame(
        columns=["sample_id", "case_id", "endpoint_key", "timestamp_window_ms", "phase", "feat"]
    )
    train, eval_ = split_fault_phase_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == 0
    assert list(train.columns) == list(df.columns)


def test_degenerate_side_preserves_columns():
    """fraction=1.0/0.0 时一侧为空，但列结构必须与输入一致（不同于"整个输入为空"
    的 test_empty_input_returns_two_empty_frames，这里输入非空，只是切分后某一侧
    恰好为空）。如果 _concat 对"部分行但拼接后为空列表"的情形返回了列不一致的帧，
    下游 pd.concat 会静默产出列错位的结果。"""
    df = _synth_baseline_df({"F1": 5})
    train_all, eval_none = split_fault_phase_temporal(df, fraction=1.0)
    assert list(eval_none.columns) == list(df.columns)
    train_none, eval_all = split_fault_phase_temporal(df, fraction=0.0)
    assert list(train_none.columns) == list(df.columns)


def test_negative_fraction_raises():
    """负数 fraction 若不校验，会触发 Python 负索引切片导致方向反转的错误切分
    （而非空切分）——必须在入口拒绝，而不是产出看似合理但错误的数据。"""
    df = _synth_baseline_df({"F1": 5})
    with pytest.raises(ValueError, match="fraction"):
        split_fault_phase_temporal(df, fraction=-0.5)


def test_fraction_greater_than_one_raises():
    df = _synth_baseline_df({"F1": 5})
    with pytest.raises(ValueError, match="fraction"):
        split_fault_phase_temporal(df, fraction=1.5)


def test_nan_fraction_raises_regardless_of_df_emptiness():
    """NaN fraction 此前的行为不一致：df 非空时深埋在 int(n*fraction) 里抛出无上下文
    的 ValueError，df 为空时因早退检查在校验之前而静默成功。现在校验必须先于
    empty 检查执行，两种情况都应一致地在入口报错。"""
    non_empty = _synth_baseline_df({"F1": 5})
    with pytest.raises(ValueError, match="fraction"):
        split_fault_phase_temporal(non_empty, fraction=float("nan"))

    empty = pd.DataFrame(
        columns=["sample_id", "case_id", "endpoint_key", "timestamp_window_ms", "phase", "feat"]
    )
    with pytest.raises(ValueError, match="fraction"):
        split_fault_phase_temporal(empty, fraction=float("nan"))


def test_missing_case_col_raises_keyerror_with_context():
    df = _synth_baseline_df({"F1": 5}).drop(columns=["case_id"])
    with pytest.raises(KeyError, match="case_id"):
        split_fault_phase_temporal(df, fraction=0.2)


def test_missing_time_col_raises_keyerror_with_context():
    df = _synth_baseline_df({"F1": 5}).drop(columns=["timestamp_window_ms"])
    with pytest.raises(KeyError, match="timestamp_window_ms"):
        split_fault_phase_temporal(df, fraction=0.2)


def test_nan_timestamp_raises_instead_of_silently_misordering():
    """time_col 含 NaN 时，sorted() 不会报错但产出的顺序不是真正的时间顺序
    （NaN 比较恒为 False），会静默破坏"train 窗全部早于 eval 窗"这个不变量。
    必须显式拒绝，而不是让下游拿到一个违反文档承诺的切分结果。"""
    df = _synth_baseline_df({"F1": 5})
    df.loc[df.index[0], "timestamp_window_ms"] = float("nan")
    with pytest.raises(ValueError, match="timestamp_window_ms"):
        split_fault_phase_temporal(df, fraction=0.2)
