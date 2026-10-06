"""平凡基线：rel_pos / zscore_l2 / zscore_max，且统计量只用训练池行。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.score_trivial_baselines import compute_rel_pos_scores, compute_zscore_scores

REPO_ROOT = Path(__file__).parents[1]


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
    """fit 段零方差列须用 1.0 哨兵，不得用 1e-9 兜底（会放大 1e9 倍——
    history 013 的爆值与 entry 027 的 RG sigmoid 饱和同源）。

    有意偏离计划模板：原模板 fit/eval 两侧 f__const 恒为 5.0，分子永远是 0，
    哨兵取 1.0 还是 1e-9 得分都为 0——即使实现退回 1e-9 该测试也通过，没有
    判别力（reviewer 实测）。eval 侧 inject 行改为 6.0 制造非零分子后，
    (6-5)/1.0 恰为 1.0：哨兵若是 1e-9 得分会是 1e9（>1e3 断言兜住），哨兵若
    取其他 >1 的值也不会恰好等于 1.0，两个方向同时锁死。
    """
    df = _contract_frame()
    df["f__const"] = 5.0
    df.loc[df["timestamp_window_ms"] >= 3000, "f__const"] = 6.0
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__const"], agg="max")
    assert out["score"].max() == 1.0
    assert out["score"].max() < 1e3


def test_scores_satisfy_contract():
    from src.contracts.scores_v0 import validate_scores_df

    validate_scores_df(compute_rel_pos_scores(_contract_frame()))


def test_zscore_group_key_includes_case_id():
    """两个 case 基线电平不同（0 vs 100），inject 增量相对各自基线都很小。
    分组键若丢掉 case_id 退成只按 endpoint 分组，c2 的 baseline 行会被拉到
    跨 case 汇总参照（mean≈50）上而得分远非 0。"""
    rows = []
    for case, base in (("c1", 0.0), ("c2", 100.0)):
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
                    "f__a": base + (1.0 if inject else 0.0),
                }
            )
    df = pd.DataFrame(rows)
    fit_df = df[~df["is_endpoint_anomaly"]]
    out = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    s = out.set_index("sample_id")["score"]
    # c2 baseline 行相对自身 (case, endpoint) 参照（mean=100，零方差→1.0 哨兵）偏离为 0
    assert s["c2_1000"] == 0.0
    assert s["c2_2000"] == 0.0


def test_zscore_nan_feature_counts_as_zero_deviation():
    """eval 侧某特征为 NaN：按 0 偏离处理——分数有限，且等于只用其余特征算出的分数。"""
    df = _contract_frame()
    df["f__b"] = 7.0
    df.loc[df["timestamp_window_ms"] >= 3000, "f__b"] = float("nan")
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    full = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a", "f__b"], agg="l2")
    partial = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    assert np.isfinite(full["score"].to_numpy()).all()
    pd.testing.assert_series_equal(full["score"], partial["score"], check_names=False)


def test_zscore_l2_and_max_are_distinguishable():
    """构造 z=(3,4)：l2 应得 5、max 应得 4。fit 取 {-1,0,1} 使 mean=0、std=1（ddof=1）。"""
    fit_df = pd.DataFrame(
        {
            "sample_id": ["f1", "f2", "f3"],
            "case_id": ["c1"] * 3,
            "endpoint_key": ["ep1"] * 3,
            "timestamp_window_ms": [1, 2, 3],
            "phase": ["baseline"] * 3,
            "is_endpoint_anomaly": [False] * 3,
            "f__a": [-1.0, 0.0, 1.0],
            "f__b": [-1.0, 0.0, 1.0],
        }
    )
    df = pd.DataFrame(
        {
            "sample_id": ["e1"],
            "case_id": ["c1"],
            "endpoint_key": ["ep1"],
            "timestamp_window_ms": [4],
            "phase": ["inject"],
            "is_endpoint_anomaly": [True],
            "f__a": [3.0],
            "f__b": [4.0],
        }
    )
    l2 = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a", "f__b"], agg="l2")
    mx = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a", "f__b"], agg="max")
    assert l2["score"].iloc[0] == pytest.approx(5.0)
    assert mx["score"].iloc[0] == pytest.approx(4.0)


def test_rel_pos_timestamp_shift_invariant_and_same_window_same_score():
    """c2 时间戳整体平移 +10^6：同序位仍同分（只依赖次序不依赖绝对时间）；
    同一时间窗加第二个 endpoint：两行同分（dense-rank 按时间窗而非按行）。"""
    df = _contract_frame()
    df.loc[df["case_id"] == "c2", "timestamp_window_ms"] += 10**6
    out = compute_rel_pos_scores(df)
    s = out.set_index("sample_id")["score"]
    # sample_id 不随时间戳平移改变；平移后 c2 与 c1 同序位仍同分
    assert s["c1_1000"] == s["c2_1000"]
    assert s["c1_4000"] == s["c2_4000"]

    df2 = _contract_frame()
    ep2 = df2[df2["case_id"] == "c1"].copy()
    ep2["endpoint_key"] = "ep2"
    ep2["sample_id"] = ep2["sample_id"] + "_ep2"
    out2 = compute_rel_pos_scores(pd.concat([df2, ep2], ignore_index=True))
    s2 = out2.set_index("sample_id")["score"]
    assert s2["c1_3000"] == s2["c1_3000_ep2"]


def test_zscore_missing_group_scores_zero_and_others_unchanged():
    """eval 含 fit 中不存在的 (case, endpoint) 组：该行按 0 偏离得 0 分，
    且其他行分数与不加该行时完全相同（缺席组不得改变既有行的参照）。"""
    df = _contract_frame()
    fit_df = df[df["timestamp_window_ms"] <= 2000]
    base = compute_zscore_scores(df, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    extra = pd.DataFrame(
        [
            {
                "sample_id": "c1_newep_3000",
                "case_id": "c1",
                "endpoint_key": "ep_new",
                "timestamp_window_ms": 3000,
                "phase": "inject",
                "is_endpoint_anomaly": True,
                "f__a": 999.0,
            }
        ]
    )
    df2 = pd.concat([df, extra], ignore_index=True)
    out = compute_zscore_scores(df2, fit_df=fit_df, feature_cols=["f__a"], agg="l2")
    s = out.set_index("sample_id")["score"]
    assert s["c1_newep_3000"] == 0.0
    pd.testing.assert_series_equal(
        out.loc[out["sample_id"] != "c1_newep_3000", "score"].reset_index(drop=True),
        base["score"],
        check_names=False,
    )


def test_cli_end_to_end_zscore_l2_and_rel_pos(tmp_path):
    """main() 全链路：最小 contract（含 2 个 feature_groups）→ scores → eval 直接消费。

    第二个 group（g2）独占信号：g1 特征恒定无偏离，inject 行的非零分数只能来自
    g2——若 schema 的第二个 feature_group 被漏读，inject 行分数会全 0。
    """
    rows = []
    for case in ("c1", "c2"):
        for i, ts in enumerate([1000, 2000, 3000, 4000]):
            inject = i >= 2
            rows.append(
                {
                    "sample_id": f"{case}_{ts}",
                    "case_id": case,
                    "endpoint_key": "ep1",
                    "timestamp_window_ms": ts,
                    "phase": "inject" if inject else "baseline",
                    "anomaly_type": "Lv_E_HTTPABORT_toy",
                    "anomaly_level": "endpoint",
                    "label_granularity": "endpoint",
                    "is_endpoint_anomaly": inject,
                    "g1__a": 5.0,
                    "g2__b": 2.0 if inject else 0.0,
                }
            )
    full = pd.DataFrame(rows)
    contract = tmp_path / "contract"
    contract.mkdir()
    train_df = full[full["timestamp_window_ms"] <= 2000].reset_index(drop=True)
    eval_df = full[full["timestamp_window_ms"] >= 2000].reset_index(drop=True)
    train_df.to_parquet(contract / "train.parquet", index=False)
    eval_df.to_parquet(contract / "eval_all.parquet", index=False)
    (contract / "schema.json").write_text(
        json.dumps(
            {
                "feature_groups": {
                    "g1": {"columns": ["g1__a"]},
                    "g2": {"columns": ["g2__b"]},
                }
            }
        )
    )

    z_out = tmp_path / "z.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/score_trivial_baselines.py",
            "--contract-dir",
            str(contract),
            "--baseline",
            "zscore_l2",
            "--out",
            str(z_out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    z = pd.read_parquet(z_out)
    expected_cols = [
        "sample_id",
        "score",
        "y_true",
        "case_id",
        "endpoint_key",
        "phase",
        "anomaly_type",
        "anomaly_level",
        "is_endpoint_anomaly",
        "label_granularity",
    ]
    assert list(z.columns) == expected_cols
    diag_cols = [
        "case_id",
        "endpoint_key",
        "phase",
        "anomaly_type",
        "anomaly_level",
        "is_endpoint_anomaly",
        "label_granularity",
    ]
    merged = eval_df.merge(z, on="sample_id", suffixes=("_ev", ""), validate="one_to_one")
    for col in diag_cols:
        pd.testing.assert_series_equal(merged[f"{col}_ev"], merged[col], check_names=False)
    assert (z["y_true"] == z["is_endpoint_anomaly"].astype(int)).all()
    s = z.set_index("sample_id")["score"]
    # g1 恒定 → z=0；g2 inject 行 z=(2−0)/1（fit 零方差 → 1.0 哨兵）→ l2 = 2.0
    assert s["c1_3000"] == pytest.approx(2.0)
    assert s["c1_2000"] == pytest.approx(0.0)

    r_out = tmp_path / "r.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/score_trivial_baselines.py",
            "--contract-dir",
            str(contract),
            "--baseline",
            "rel_pos",
            "--out",
            str(r_out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    r = pd.read_parquet(r_out)
    assert list(r.columns) == expected_cols
    rs = r.set_index("sample_id")["score"]
    assert rs["c1_2000"] < rs["c1_3000"] < rs["c1_4000"]

    m_out = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_baseline_v0.py",
            "--scores",
            str(z_out),
            "--out",
            str(m_out),
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    metrics = json.loads(m_out.read_text())
    assert metrics["per_case_auroc_macro"] == pytest.approx(1.0)
