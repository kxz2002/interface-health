"""平凡基线：rel_pos / zscore_l2 / zscore_max，且统计量只用训练池行。"""

from __future__ import annotations

import pandas as pd

from scripts.score_trivial_baselines import compute_rel_pos_scores, compute_zscore_scores


def _contract_frame():
    """2 个 case × 1 endpoint × 4 窗；每个 case 后两窗为 inject。"""
    rows = []
    for case, base in (("c1", 0.0), ("c2", 10.0)):
        for i, ts in enumerate([1000, 2000, 3000, 4000]):
            inject = i >= 2
            rows.append(
                {
                    "sample_id": f"{case}_{ts}",
                    "case_id": case,
                    "endpoint_key": "ep1",
                    "timestamp_window_ms": ts,
                    "phase": "inject" if inject else "baseline",
                    "is_endpoint_anomaly": inject,
                    "f__a": base + (100.0 if inject else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_rel_pos_is_monotone_within_case_and_case_independent():
    out = compute_rel_pos_scores(_contract_frame())
    assert list(out.columns) == ["sample_id", "score", "y_true"]
    s = out.set_index("sample_id")["score"]
    assert s["c1_1000"] < s["c1_2000"] < s["c1_3000"] < s["c1_4000"]
    # 两个 case 的同序位窗口得分相同（rel_pos 不含 case 身份信息）
    assert s["c1_1000"] == s["c2_1000"]


def test_zscore_uses_only_allowed_rows_no_leakage():
    """统计量只能来自 fit_df。把 fit_df 限成前两窗，eval 侧 inject 巨值不得
    参与 mean/std——否则 z 会被自身拉平、分数塌陷。"""
    df = _contract_frame()
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    s = out.set_index("sample_id")["score"]
    assert s["c1_3000"] > 10 * max(s["c1_1000"], s["c1_2000"])


def test_zscore_degenerate_std_does_not_explode():
    """fit 段零方差列不得用 1e-9 兜底（会放大 1e9 倍——history 013 的爆值与
    entry 027 的 RG sigmoid 饱和同源），须用 1.0 哨兵。"""
    df = _contract_frame()
    df["f__const"] = 5.0
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__const"], agg="max")
    assert out["score"].max() < 1e3


def test_scores_satisfy_contract():
    from src.contracts.scores_v0 import validate_scores_df

    validate_scores_df(compute_rel_pos_scores(_contract_frame()))
