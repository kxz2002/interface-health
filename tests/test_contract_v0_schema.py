"""contract v0 schema：18 维特征列 + 6 个标识列 + 8 个标签列。"""

from pathlib import Path

import pandas as pd
import pytest

from src.contracts.contract_v0 import (
    REQUIRED_ID_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
    ContractV0Error,
    validate_contract_df,
)

REPO_ROOT = Path(__file__).parents[1]


def _make_valid_row():
    return {
        # ID
        "sample_id": "Normal_planA__POST:/api/v1/preserveservice/preserve__1780972185000",
        "case_id": "Normal_planA",
        "endpoint_key": "POST:/api/v1/preserveservice/preserve",
        "service_name": "ts-preserve-service",
        "timestamp_window_ms": 1780972185000,
        "window_str": "2026-06-09T02:29:45Z",
        # endpoint_red (10)
        "endpoint_red__trace_request_count": 0.5,
        "endpoint_red__trace_latency_p50": 0.3,
        "endpoint_red__trace_latency_p95": 0.4,
        "endpoint_red__trace_error_rate": 0.0,
        "endpoint_red__trace_5xx_rate": 0.0,
        "endpoint_red__client_request_count": 0.5,
        "endpoint_red__client_latency_p95": 0.4,
        "endpoint_red__client_error_rate": 0.0,
        "endpoint_red__client_5xx_rate": 0.0,
        "endpoint_red__latency_divergence": 0.1,
        # service_metric (5)
        "service_metric__cpu_usage_rate": 0.2,
        "service_metric__memory_usage_ratio": 0.3,
        "service_metric__net_rx_error_rate": 0.0,
        "service_metric__net_tx_error_rate": 0.0,
        "service_metric__process_count": 0.5,
        # service_log (3)
        "service_log__event_rate": 0.4,
        "service_log__error_ratio": 0.0,
        "service_log__template_diversity": 0.2,
        # Label
        "phase": "normal",
        "is_anomaly": False,
        "is_train_eligible": True,
        "injection_start_ms": None,
        "injection_end_ms": None,
        "target_service": None,
        "anomaly_type": "Normal",
        "anomaly_level": "none",
    }


def test_valid_contract_passes():
    df = pd.DataFrame([_make_valid_row()])
    validated = validate_contract_df(df, REPO_ROOT / "configs/contract/v0.yaml")
    assert len(validated) == 1


def test_missing_feature_column_raises():
    row = _make_valid_row()
    del row["service_log__event_rate"]
    with pytest.raises(ContractV0Error, match="service_log__event_rate"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")


def test_score_out_of_range_for_rate_column_raises():
    """error_rate / 5xx_rate / memory_usage_ratio 必须在 [0, 1]。"""
    row = _make_valid_row()
    row["endpoint_red__trace_error_rate"] = 1.5
    with pytest.raises(ContractV0Error, match=r"trace_error_rate.*\[0"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")


def test_duplicate_sample_id_raises():
    df = pd.DataFrame([_make_valid_row(), _make_valid_row()])
    with pytest.raises(ContractV0Error, match="duplicate"):
        validate_contract_df(df, REPO_ROOT / "configs/contract/v0.yaml")


def test_inconsistent_anomaly_label_raises():
    """is_anomaly 必须等价于 phase == 'inject'。"""
    row = _make_valid_row()
    row["phase"] = "inject"
    row["is_anomaly"] = False  # 矛盾
    with pytest.raises(ContractV0Error, match="phase.*is_anomaly"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")
