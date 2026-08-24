"""contract v0 schema：18 维特征列 + 6 个标识列 + 10 个标签列。"""

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
        "label_granularity": "case",
        "is_endpoint_anomaly": False,
        "label_target_observable": True,
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


def test_service_label_granularity_is_valid():
    """entry 025 新增的 "service" 档必须被校验放行——它是三档枚举里的合法值，
    不是拼写错误。这条测试防止未来有人"收紧"枚举时把 service 档误删。"""
    row = _make_valid_row()
    row["label_granularity"] = "service"
    validated = validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")
    assert len(validated) == 1


def test_unknown_label_granularity_raises():
    """label_granularity 拼错（如 "svc"）必须报错而不是静默放行。

    下游 _write_v1 的 recover 吸收判据与 eval 分层都按具体字符串分支，非法值不会
    触发异常、只会让那批 case 悄悄走进 else 分支，行为与预期相反且无任何提示。
    """
    row = _make_valid_row()
    row["label_granularity"] = "svc"
    with pytest.raises(ContractV0Error, match="label_granularity.*非法取值"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")


def test_positive_label_outside_inject_window_raises():
    """is_endpoint_anomaly 必须是 is_anomaly 的子集：正样本只能落在 inject 窗口内。

    三档 label_granularity 的正样本判据都形如 `is_anomaly & <target 条件>`，故该
    蕴含关系恒成立。校验它是为了挡住未来某档判据被改成不带 is_anomaly 的形式——
    那会让 baseline/recover 行被标成正样本，评估协议（CLAUDE.md：正样本 = inject
    阶段内的 endpoint × 时间窗）直接失真且不报错。
    """
    row = _make_valid_row()
    row["phase"] = "recover"
    row["is_anomaly"] = False
    row["is_endpoint_anomaly"] = True  # 矛盾：recover 阶段不可能有正样本
    with pytest.raises(ContractV0Error, match="is_endpoint_anomaly.*inject 窗口外"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")
