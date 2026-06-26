import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.normalization import Normalizer


@pytest.fixture
def normal_df():
    return pd.DataFrame(
        {
            "endpoint_key": ["ep1"] * 4 + ["ep2"] * 4,
            "endpoint_red__trace_request_count": [0, 10, 20, 30, 100, 200, 300, 400],
        }
    )


def test_per_endpoint_minmax_fit_transform(normal_df):
    norm = Normalizer(
        rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")},
    )
    norm.fit(normal_df)
    out = norm.transform(normal_df)
    ep1_vals = out.loc[normal_df["endpoint_key"] == "ep1", "endpoint_red__trace_request_count"]
    assert ep1_vals.min() == pytest.approx(0.0)
    assert ep1_vals.max() == pytest.approx(1.0)


def test_anomaly_case_uses_normal_stats(normal_df):
    """故障 case 用 Normal 拟合的参数，超过 1 的值允许出现（不 clip）。"""
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    anomaly_df = pd.DataFrame(
        {
            "endpoint_key": ["ep1"],
            "endpoint_red__trace_request_count": [60],  # 超出 Normal ep1 的 max=30
        }
    )
    out = norm.transform(anomaly_df)
    # (60 - 0) / (30 - 0) = 2.0
    assert out["endpoint_red__trace_request_count"].iloc[0] == pytest.approx(2.0)


def test_stats_roundtrip_json(tmp_path, normal_df):
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    norm.save(tmp_path / "stats.json")
    norm2 = Normalizer.load(tmp_path / "stats.json")
    pd.testing.assert_frame_equal(norm.transform(normal_df), norm2.transform(normal_df))
