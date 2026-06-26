from pathlib import Path

import pandas as pd
import pytest

from src.preprocessors.trace_preprocessor import TracePreprocessor

REPO_ROOT = Path(__file__).parents[1]

V0_ENDPOINTS = {
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelplanservice/travelPlan/cheapest",
    "POST:/api/v1/travelplanservice/travelPlan/minStation",
    "POST:/api/v1/travelplanservice/travelPlan/quickest",
    "GET:/api/v1/routeservice/routes",
}


def test_trace_output_columns_match_contract():
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    cols = pre.get_feature_columns()
    expected = {
        "endpoint_red__trace_request_count",
        "endpoint_red__trace_latency_p50",
        "endpoint_red__trace_latency_p95",
        "endpoint_red__trace_error_rate",
        "endpoint_red__trace_5xx_rate",
    }
    assert set(cols) == expected


def test_trace_transform_filters_to_v0_endpoints():
    """非 v0 endpoint 的行必须被丢弃。"""
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(
        REPO_ROOT / "tests/fixtures/mini_tt_traces_red_15s.csv",
        case_meta={"case_id": "Normal_planA"},
    )
    assert set(df["endpoint_key"].unique()).issubset(V0_ENDPOINTS)


def test_trace_transform_preserves_window_alignment():
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(
        REPO_ROOT / "tests/fixtures/mini_tt_traces_red_15s.csv",
        case_meta={"case_id": "Normal_planA"},
    )
    # 时间戳必须是 15s 的整数倍（毫秒，15000）
    assert (df["timestamp_window_ms"] % 15_000 == 0).all()
