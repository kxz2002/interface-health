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


def test_negative_rate_column_rejected_on_v1_path(tmp_path):
    """v1/v0 路径的 [0,1] 校验对负值同样生效——放宽只发生在 v2，不能顺手松掉。"""
    row = _make_valid_row()
    row["endpoint_red__trace_error_rate"] = -0.1
    with pytest.raises(ContractV0Error, match=r"trace_error_rate.*\[0, 1\]"):
        validate_contract_df(pd.DataFrame([row]), REPO_ROOT / "configs/contract/v0.yaml")


# v2 config：18 维特征声明与 v0.yaml 相同，区别只在 contract_version 与 per-case
# z-score normalization（v2×min_max 组合在配置加载期即被守卫拒绝，见
# tests/test_contract_config.py），供 v2 校验分支测试使用。
def _write_v2_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "v2.yaml"
    cfg_path.write_text(
        'contract_version: "v2"\n'
        "window_size_s: 15\n"
        "expand_train_pool: true\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features:\n"
        "      - trace_request_count\n"
        "      - trace_latency_p50\n"
        "      - trace_latency_p95\n"
        "      - trace_error_rate\n"
        "      - trace_5xx_rate\n"
        "      - client_request_count\n"
        "      - client_latency_p95\n"
        "      - client_error_rate\n"
        "      - client_5xx_rate\n"
        "      - latency_divergence\n"
        "    normalization: per_case_endpoint_z_score\n"
        "    candidates_pool: []\n"
        "  service_metric:\n"
        "    preprocessor: MetricPreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features:\n"
        "      - cpu_usage_rate\n"
        "      - memory_usage_ratio\n"
        "      - net_rx_error_rate\n"
        "      - net_tx_error_rate\n"
        "      - process_count\n"
        "    normalization: per_case_service_z_score\n"
        "    candidates_pool: []\n"
        "  service_log:\n"
        "    preprocessor: LogPreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features:\n"
        "      - event_rate\n"
        "      - error_ratio\n"
        "      - template_diversity\n"
        "    normalization: per_case_service_z_score\n"
        "    candidates_pool: []\n"
    )
    return cfg_path


def test_v2_rate_columns_allow_negative_and_above_one(tmp_path):
    """v2 per-case z-score 下 rate 列不再落 [0,1]：z 值天然有负有超 1，
    这类值必须放行，否则 v2 产物逐行过不了自己的 schema 校验。"""
    row_neg = _make_valid_row()
    row_neg["sample_id"] = row_neg["sample_id"] + "_neg"
    row_neg["endpoint_red__trace_error_rate"] = -1.5

    row_hi = _make_valid_row()
    row_hi["sample_id"] = row_hi["sample_id"] + "_hi"
    row_hi["endpoint_red__client_error_rate"] = 2.5

    cfg_path = _write_v2_config(tmp_path)
    validated = validate_contract_df(pd.DataFrame([row_neg, row_hi]), cfg_path)
    assert len(validated) == 2


def test_v2_rate_columns_reject_explosive_magnitude(tmp_path):
    """[0,1] 退役后量级 sanity 接手：entry 013 的零方差除数把差值放大 1e9~1e13
    倍的事故不能因放宽校验而重演。1e3 阈值与正常 z-score（个位数）隔着 2 个
    数量级以上，1e9 必须被拒。"""
    row = _make_valid_row()
    row["endpoint_red__trace_error_rate"] = 2e9
    cfg_path = _write_v2_config(tmp_path)
    with pytest.raises(ContractV0Error, match=r"trace_error_rate.*量级"):
        validate_contract_df(pd.DataFrame([row]), cfg_path)


def test_v2_rate_columns_accept_boundary_1e3(tmp_path):
    """量级 sanity 的边界值 ±1e3 本身必须放行（闭区间），防止手滑写成严格不等号。"""
    row_hi = _make_valid_row()
    row_hi["sample_id"] = row_hi["sample_id"] + "_hi"
    row_hi["endpoint_red__trace_error_rate"] = 1e3
    row_lo = _make_valid_row()
    row_lo["sample_id"] = row_lo["sample_id"] + "_lo"
    row_lo["endpoint_red__trace_error_rate"] = -1e3
    cfg_path = _write_v2_config(tmp_path)
    validated = validate_contract_df(pd.DataFrame([row_hi, row_lo]), cfg_path)
    assert len(validated) == 2


def test_v2_non_rate_feature_rejects_explosive_magnitude(tmp_path):
    """量级 sanity 必须覆盖全部特征列而非只守 rate 列：entry 013 实际爆值的
    client_latency_p95 / latency_divergence 都不是 rate 列，只守 rate 列会让
    承诺（防 1e9 重演）与防线错位。v2 配置守卫已强制所有特征列都是 z-score
    产出，故对全特征列做量级检查语义同样成立。"""
    row = _make_valid_row()
    row["endpoint_red__client_latency_p95"] = 2e9
    cfg_path = _write_v2_config(tmp_path)
    with pytest.raises(ContractV0Error, match=r"client_latency_p95.*量级"):
        validate_contract_df(pd.DataFrame([row]), cfg_path)


def test_v2_non_rate_feature_accepts_negative_above_one_and_boundary(tmp_path):
    """非 rate 特征列在 v2 下同样按 z-score 尺度放行：负值、超 1 值与 ±1e3
    边界都合法（v1 路径上这些列本就不受 [0,1] 约束，这里钉死的是 v2 全特征列
    量级检查的边界与方向性）。"""
    row_hi = _make_valid_row()
    row_hi["sample_id"] = row_hi["sample_id"] + "_hi"
    row_hi["endpoint_red__latency_divergence"] = 1e3
    row_lo = _make_valid_row()
    row_lo["sample_id"] = row_lo["sample_id"] + "_lo"
    row_lo["endpoint_red__latency_divergence"] = -1e3
    row_neg = _make_valid_row()
    row_neg["sample_id"] = row_neg["sample_id"] + "_neg"
    row_neg["endpoint_red__client_latency_p95"] = -3.7
    cfg_path = _write_v2_config(tmp_path)
    validated = validate_contract_df(pd.DataFrame([row_hi, row_lo, row_neg]), cfg_path)
    assert len(validated) == 3


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
