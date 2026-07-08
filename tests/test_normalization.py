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


def test_group_all_nan_in_fit_keeps_other_cases_untouched():
    """某 group（如某 service 的 metric 列）在 fit 集合（Normal）里全 NaN 时，
    不应把这组 nan 统计量通过减法/除法扩散到其他 case 里这个 group 本来完好的数值。
    对应 Normal 采集缺口（如 cAdvisor 掉线）曾把其他 case 的真实 metric 数值污染成 NaN 的 bug。
    """
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a"] * 3,
            "service_metric__cpu_usage_rate": [float("nan")] * 3,  # Normal 采集缺失
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    other_case_df = pd.DataFrame(
        {
            "service_name": ["svc-a", "svc-a"],
            "service_metric__cpu_usage_rate": [0.1, 0.6],  # 真实数值，来自数据完好的 case
        }
    )
    out = norm.transform(other_case_df)
    assert out["service_metric__cpu_usage_rate"].tolist() == [0.1, 0.6]


def test_group_all_nan_in_fit_does_not_affect_other_groups():
    """全 NaN 的 group 只跳过自身，同一列里其他数据完好的 group 仍正常归一化。"""
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a", "svc-a", "svc-b", "svc-b"],
            "service_metric__cpu_usage_rate": [float("nan"), float("nan"), 0.0, 10.0],
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    out = norm.transform(fit_df)
    svc_b_vals = out.loc[out["service_name"] == "svc-b", "service_metric__cpu_usage_rate"]
    assert svc_b_vals.tolist() == pytest.approx([0.0, 1.0])
    svc_a_vals = out.loc[out["service_name"] == "svc-a", "service_metric__cpu_usage_rate"]
    assert svc_a_vals.isna().all()


def test_global_scope_all_nan_in_fit_keeps_original_values():
    fit_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [float("nan"), float("nan")]})
    norm = Normalizer(rules={"endpoint_red__trace_5xx_rate": ("global", "min_max")})
    norm.fit(fit_df)

    other_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [0.2, 0.9]})
    out = norm.transform(other_df)
    assert out["endpoint_red__trace_5xx_rate"].tolist() == [0.2, 0.9]


def test_group_partially_nan_in_fit_uses_non_nan_subset():
    """group 只是部分 NaN（非全 NaN）时，fit 用 pandas 默认 skipna 语义从非 NaN 子集
    算统计量，正常参与归一化——不同于全 NaN 时的跳过路径。这里显式钉住该行为，
    避免未来重构在"部分 NaN"和"全 NaN"两条路径之间引入混淆。
    """
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a"] * 4,
            "service_metric__cpu_usage_rate": [float("nan"), 0.1, float("nan"), 0.3],
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    lo, hi = norm._stats["service_metric__cpu_usage_rate"].by_group["svc-a"]
    assert (lo, hi) == pytest.approx((0.1, 0.3))

    out = norm.transform(fit_df)
    vals = out["service_metric__cpu_usage_rate"]
    assert vals.iloc[1] == pytest.approx(0.0)  # (0.1-0.1)/(0.3-0.1)
    assert vals.iloc[3] == pytest.approx(1.0)  # (0.3-0.1)/(0.3-0.1)
    assert vals.iloc[0] != vals.iloc[0]  # 原本就是 NaN 的行仍是 NaN（NaN != NaN）
    assert vals.iloc[2] != vals.iloc[2]
