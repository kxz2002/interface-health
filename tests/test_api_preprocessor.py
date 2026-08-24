from pathlib import Path

import pandas as pd
import yaml

from src.preprocessors.api_preprocessor import ApiPreprocessor

REPO_ROOT = Path(__file__).parents[1]
ENDPOINT_MAPPING = REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
FIXTURE = REPO_ROOT / "tests/fixtures/mini_tt_endpoint_health_15s.csv"


TRACE_FIXTURE = REPO_ROOT / "tests/fixtures/mini_tt_traces_red_15s.csv"


def _make_preprocessor() -> ApiPreprocessor:
    return ApiPreprocessor(
        endpoint_mapping_path=ENDPOINT_MAPPING,
        trace_red_path=TRACE_FIXTURE,
    )


def test_api_output_columns():
    pre = _make_preprocessor()
    assert set(pre.get_feature_columns()) == {
        "endpoint_red__client_request_count",
        "endpoint_red__client_latency_p95",
        "endpoint_red__client_error_rate",
        "endpoint_red__client_5xx_rate",
        "endpoint_red__latency_divergence",
        "endpoint_red__client_content_length_mean",
        "endpoint_red__client_content_length_rel_shift",
        "endpoint_red__client_body_hash_mismatch_rate",
    }


def test_api_transform_returns_key_and_feature_columns():
    pre = _make_preprocessor()
    df = pre.transform(FIXTURE, case_meta={"case_id": "Normal_planA"})
    assert (
        list(df.columns)
        == [
            "endpoint_key",
            "timestamp_window_ms",
        ]
        + pre.get_feature_columns()
    )


def test_api_transform_computes_latency_divergence():
    """latency_divergence = client_latency_p95 − trace_latency_p95（join 同窗口 trace 数据）。"""
    pre = _make_preprocessor()
    df = pre.transform(FIXTURE, case_meta={"case_id": "Normal_planA"})
    assert "endpoint_red__latency_divergence" in df.columns
    assert df["endpoint_red__latency_divergence"].notna().any()

    # 至少一行可逐项验证：divergence == client_p95 - trace_p95
    matched = df[df["endpoint_red__latency_divergence"].notna()].iloc[0]
    expected = matched["endpoint_red__client_latency_p95"] - _trace_p95_for(
        matched["endpoint_key"], matched["timestamp_window_ms"]
    )
    assert matched["endpoint_red__latency_divergence"] == expected


def test_api_transform_filters_to_v0_endpoints():
    """输出只保留 v0 映射表中的 endpoint，数量 ≤ 8。"""
    pre = _make_preprocessor()
    df = pre.transform(FIXTURE, case_meta={"case_id": "Normal_planA"})
    assert df["endpoint_key"].nunique() <= 8
    # routeservice 是名单外的死条目，必须被过滤掉
    assert df["endpoint_key"].str.contains("routeservice").sum() == 0
    # inside_pay_service 是真实白名单 endpoint，必须保留
    assert df["endpoint_key"].str.contains("inside_pay_service").sum() > 0
    v0_endpoints = set(yaml.safe_load(ENDPOINT_MAPPING.read_text()).keys())
    assert set(df["endpoint_key"].unique()).issubset(v0_endpoints)


def test_api_transform_timestamp_is_epoch_ms():
    """timestamp_window_ms 必须是 15000ms 整数倍的整数（与 trace 侧 join 键一致）。"""
    pre = _make_preprocessor()
    df = pre.transform(FIXTURE, case_meta={"case_id": "Normal_planA"})
    assert pd.api.types.is_integer_dtype(df["timestamp_window_ms"])
    assert (df["timestamp_window_ms"] % 15000 == 0).all()


def _trace_p95_for(endpoint_key: str, ts_ms: int) -> float:
    trace = pd.read_csv(TRACE_FIXTURE)
    row = trace[(trace["endpoint_key"] == endpoint_key) & (trace["timestamp_window"] == ts_ms)]
    return float(row["trace_latency_p95"].iloc[0])


def test_api_transform_fills_nan_for_legacy_batch_missing_content_length_body_hash(
    tmp_path, caplog
):
    """旧批次（new_ep1/anomod_v1）的 tt_endpoint_health_15s.csv 没有 content_length/
    body_hash 三列，transform() 必须降级填 NaN 而不是 KeyError，且要有 warning 日志
    可供排查（而非静默降级）。mini fixture 本身已经带全三列，本测试单独构造一份
    缺列的 legacy 格式 CSV 来精确复现这条此前完全没有测试覆盖的路径。"""
    legacy_csv = tmp_path / "legacy_tt_endpoint_health_15s.csv"
    legacy_csv.write_text(
        "case_id,anomaly_type,timestamp_window,endpoint_key,request_count,error_rate,"
        "latency_mean,latency_p50,latency_p95,latency_p99,status_2xx_rate,status_4xx_rate,"
        "status_5xx_rate,method,normalized_path\n"
        "Normal_planA,Normal,2026-06-09T02:29:45Z,POST:/api/v1/travelservice/trips/left,"
        "12,0.0,2268.70,2100.00,3100.50,3500.00,1.0,0.0,0.0,POST,"
        "/api/v1/travelservice/trips/left\n"
    )

    pre = _make_preprocessor()
    with caplog.at_level("WARNING"):
        df = pre.transform(legacy_csv, case_meta={"case_id": "Normal_planA"})

    for col in (
        "endpoint_red__client_content_length_mean",
        "endpoint_red__client_content_length_rel_shift",
        "endpoint_red__client_body_hash_mismatch_rate",
    ):
        assert col in df.columns
        assert df[col].isna().all()
    assert any("content_length" in record.message for record in caplog.records)
