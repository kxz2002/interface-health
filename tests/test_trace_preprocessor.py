from pathlib import Path

import pandas as pd
import pytest

from src.preprocessors.trace_preprocessor import TracePreprocessor

REPO_ROOT = Path(__file__).parents[1]

V0_ENDPOINTS = {
    "GET:/api/v1/assuranceservice/assurances/types",
    "GET:/api/v1/contactservice/contacts/account/{uuid}",
    "POST:/api/v1/inside_pay_service/inside_payment",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/users/login",
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


def test_trace_transform_preserves_is_target_endpoint_when_present(tmp_path):
    """endpoint_raw2 类数据源的 tt_traces_red_15s.csv 带 is_target_endpoint 列，
    该列必须原样传递到输出，供后续 contract 层判断 per-endpoint 精确标签。"""
    csv_path = tmp_path / "tt_traces_red_15s.csv"
    csv_path.write_text(
        "case_id,anomaly_type,timestamp_window,endpoint_key,method,normalized_path,"
        "trace_request_count,trace_latency_mean,trace_latency_p50,trace_latency_p95,"
        "trace_latency_p99,trace_error_rate,trace_5xx_rate,trace_4xx_rate,"
        "trace_status_coverage,weak_is_anomaly,label_confidence,latency_anomaly_signal,"
        "error_anomaly_signal,5xx_anomaly_signal,phase,injection_start_ms,injection_end_ms,"
        "target_service,target_endpoint,anomaly_level,is_target_endpoint\n"
        "c1,Lv_E_HTTPABORT_assurance,1783150000000,"
        "GET:/api/v1/assuranceservice/assurances/types,GET,"
        "/api/v1/assuranceservice/assurances/types,9,22000.0,21500.0,44000.0,44000.0,"
        "0.0,0.0,0.0,1.0,0,normal,0,0,0,inject,1783150000000,1783150015000,"
        "ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,True\n"
    )
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(csv_path, case_meta={"case_id": "c1"})
    assert "is_target_endpoint" in df.columns
    assert bool(df["is_target_endpoint"].iloc[0]) is True


def test_trace_transform_fills_false_when_is_target_endpoint_absent():
    """anomod_v1 类数据源没有 is_target_endpoint 列时，不应报错，应填 False 占位。"""
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(
        REPO_ROOT / "tests/fixtures/mini_tt_traces_red_15s.csv",
        case_meta={"case_id": "Normal_planA"},
    )
    assert "is_target_endpoint" in df.columns
    assert not df["is_target_endpoint"].any()
