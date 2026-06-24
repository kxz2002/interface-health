"""tests/test_scores_contract.py — scores contract v0 校验器测试。

这些测试是 contract 的"第二份说明书"：每条 case 对应 scores_v0.py 文档里
列出的一条规则。后续若放宽或收紧规则，必须同步改测试，否则规则就只是注释。

设计原则：
- 每个 case 只破坏一条规则，避免多重失败混淆定位
- 错误不验证完整 message，只验证关键关键词，给文案留迭代余地
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.scores_v0 import ScoresContractError, validate_scores_df


def _valid_df(n: int = 5) -> pd.DataFrame:
    """构造合法的 scores df。每个负向 case 都基于这个再单点破坏。"""
    return pd.DataFrame(
        {
            "sample_id": [f"s_{i}" for i in range(n)],
            "score": np.linspace(0.1, 5.0, n),
            "y_true": [0, 1, 0, 1, 0][:n],
        }
    )


# ---- 正向 cases ----


def test_valid_df_passes():
    validate_scores_df(_valid_df())


def test_with_optional_columns_passes():
    df = _valid_df()
    df["case_id"] = "case_a"
    df["phase"] = "inject"
    df["timestamp_window"] = "2026-06-09T02:29:45Z"
    validate_scores_df(df)


def test_y_true_as_float_with_integer_values_passes():
    # y_true 即使是 float dtype，只要值都是 0/1 就应通过
    df = _valid_df()
    df["y_true"] = df["y_true"].astype(float)
    validate_scores_df(df)


# ---- 结构性失败 ----


def test_not_dataframe_raises():
    with pytest.raises(ScoresContractError, match="DataFrame"):
        validate_scores_df([{"sample_id": "s_0", "score": 1.0, "y_true": 0}])


def test_empty_df_raises():
    df = _valid_df(0).iloc[0:0]
    with pytest.raises(ScoresContractError, match="empty"):
        validate_scores_df(df)


@pytest.mark.parametrize("missing_col", ["sample_id", "score", "y_true"])
def test_missing_required_column_raises(missing_col):
    df = _valid_df().drop(columns=[missing_col])
    with pytest.raises(ScoresContractError, match="missing required columns"):
        validate_scores_df(df)


# ---- sample_id 失败 ----


def test_sample_id_duplicated_raises():
    df = _valid_df()
    df.loc[0, "sample_id"] = df.loc[1, "sample_id"]
    with pytest.raises(ScoresContractError, match="duplicate"):
        validate_scores_df(df)


def test_sample_id_with_nan_raises():
    df = _valid_df()
    df["sample_id"] = df["sample_id"].astype(object)
    df.loc[0, "sample_id"] = None
    with pytest.raises(ScoresContractError, match="NaN"):
        validate_scores_df(df)


# ---- score 失败 ----


def test_score_with_nan_raises():
    df = _valid_df()
    df.loc[0, "score"] = np.nan
    with pytest.raises(ScoresContractError, match="NaN"):
        validate_scores_df(df)


def test_score_with_positive_inf_raises():
    df = _valid_df()
    df.loc[0, "score"] = np.inf
    with pytest.raises(ScoresContractError, match="inf"):
        validate_scores_df(df)


def test_score_with_negative_inf_raises():
    df = _valid_df()
    df.loc[0, "score"] = -np.inf
    with pytest.raises(ScoresContractError, match="inf"):
        validate_scores_df(df)


# ---- y_true 失败 ----


def test_y_true_out_of_set_raises():
    df = _valid_df()
    df.loc[0, "y_true"] = 2
    with pytest.raises(ScoresContractError, match=r"0, 1"):
        validate_scores_df(df)


def test_y_true_negative_raises():
    df = _valid_df()
    df.loc[0, "y_true"] = -1
    with pytest.raises(ScoresContractError, match=r"0, 1"):
        validate_scores_df(df)


def test_y_true_with_nan_raises():
    df = _valid_df()
    df["y_true"] = df["y_true"].astype(float)
    df.loc[0, "y_true"] = np.nan
    with pytest.raises(ScoresContractError, match="NaN"):
        validate_scores_df(df)


def test_y_true_float_with_decimal_raises():
    # 0.5 这种浮点小数应被拒（而非静默 cast 成 0）
    df = _valid_df()
    df["y_true"] = df["y_true"].astype(float)
    df.loc[0, "y_true"] = 0.5
    with pytest.raises(ScoresContractError, match="non-integer"):
        validate_scores_df(df)


# ---- 多错误聚合 ----


def test_multiple_violations_reported_together():
    """contract 校验失败时应一次列出所有问题，便于调用方一次定位。"""
    df = _valid_df()
    df.loc[0, "score"] = np.nan
    df.loc[0, "y_true"] = 5
    with pytest.raises(ScoresContractError) as exc_info:
        validate_scores_df(df)
    msg = str(exc_info.value)
    assert "NaN" in msg
    assert "0, 1" in msg
