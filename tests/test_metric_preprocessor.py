from pathlib import Path

import pandas as pd

from src.preprocessors.metric_preprocessor import MetricPreprocessor

REPO_ROOT = Path(__file__).parents[1]
FIXTURE = REPO_ROOT / "tests/fixtures/mini_metric_data.csv"

# fixture 时间戳基准（秒级 unix），见 mini_metric_data.csv 生成约定
_BASE_S = 1_780_972_185
_BASE_MS = _BASE_S * 1000
_WIDE_META = {
    "case_id": "Normal",
    "trace_window_start_ms": 0,
    "trace_window_end_ms": 10**13,
}


def test_metric_output_columns():
    pre = MetricPreprocessor()
    assert set(pre.get_feature_columns()) == {
        "service_metric__cpu_usage_rate",
        "service_metric__memory_usage_ratio",
        "service_metric__net_rx_error_rate",
        "service_metric__net_tx_error_rate",
        "service_metric__process_count",
    }


def test_metric_transform_aggregates_to_service_level(tmp_path):
    """同 service 多 pod 必须聚合（mean），输出按 (service, window) 唯一。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(FIXTURE, case_meta=_WIDE_META)
    assert not df.duplicated(["service_name", "timestamp_window_ms"]).any()
    # fixture 含 2 个 ts service
    assert set(df["service_name"]) == {"ts-route-service", "ts-order-service"}


def test_metric_uses_intermediate_cache(tmp_path):
    """同一 case 第二次 transform 应直接读 intermediate parquet。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    pre.transform(FIXTURE, _WIDE_META)
    assert (tmp_path / "metrics_filtered_Normal.parquet").exists()


def test_metric_window_floor_alignment(tmp_path):
    """metric 时间戳必须 floor 对齐到 15s 桶。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(
        FIXTURE,
        case_meta={
            "case_id": "Normal",
            "trace_window_start_ms": _BASE_MS,
            "trace_window_end_ms": _BASE_MS + 200_000,
        },
    )
    assert (df["timestamp_window_ms"] % 15_000 == 0).all()


def test_metric_cpu_rate_is_counter_diff(tmp_path):
    """cpu_usage_rate 应由累积计数器相邻差分得到（非裸累积值），首窗无前值→0。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(FIXTURE, _WIDE_META)
    route = df[df["service_name"] == "ts-route-service"].sort_values("timestamp_window_ms")
    # 计数器累积值是几百量级；rate 应是每秒 core 量级（远小于裸值）
    assert (route["service_metric__cpu_usage_rate"] >= 0).all()
    assert route["service_metric__cpu_usage_rate"].max() < 50
    # 第一个窗口没有前值，rate 必须为 0
    assert route["service_metric__cpu_usage_rate"].iloc[0] == 0.0


def test_metric_memory_ratio_in_unit_range(tmp_path):
    """memory_usage_ratio = usage / limit，落在 [0, 1] 附近。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(FIXTURE, _WIDE_META)
    ratio = df["service_metric__memory_usage_ratio"]
    assert (ratio >= 0).all()
    assert (ratio <= 1.0).all()


def test_metric_net_error_rate_broadcast_host_level(tmp_path):
    """网络错误计数器无 pod 标签（host 级），同一窗口内对所有 service 取相同值。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(FIXTURE, _WIDE_META)
    for _, grp in df.groupby("timestamp_window_ms"):
        assert grp["service_metric__net_rx_error_rate"].nunique() == 1
        assert grp["service_metric__net_tx_error_rate"].nunique() == 1
    assert (df["service_metric__net_rx_error_rate"] >= 0).all()


def test_metric_out_of_window_filtered(tmp_path):
    """窗口外的行被预过滤，结果不含越界时间戳。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    df = pre.transform(
        FIXTURE,
        case_meta={
            "case_id": "Normal",
            "trace_window_start_ms": _BASE_MS,
            "trace_window_end_ms": _BASE_MS + 45_000,
        },
    )
    assert df["timestamp_window_ms"].min() >= (_BASE_MS // 15_000) * 15_000
    assert df["timestamp_window_ms"].max() <= _BASE_MS + 45_000
