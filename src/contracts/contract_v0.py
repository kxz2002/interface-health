from __future__ import annotations

from typing import NewType

import pandas as pd

from src.contracts.contract_config import load_contract_config

ContractV0 = NewType("ContractV0", pd.DataFrame)

REQUIRED_ID_COLUMNS = [
    "sample_id",
    "case_id",
    "endpoint_key",
    "service_name",
    "timestamp_window_ms",
    "window_str",
]
REQUIRED_LABEL_COLUMNS = [
    "phase",
    "is_anomaly",
    "is_train_eligible",
    "injection_start_ms",
    "injection_end_ms",
    "target_service",
    "anomaly_type",
    "anomaly_level",
]
RATE_COLUMNS = [
    "endpoint_red__trace_error_rate",
    "endpoint_red__trace_5xx_rate",
    "endpoint_red__client_error_rate",
    "endpoint_red__client_5xx_rate",
    "service_metric__cpu_usage_rate",
    "service_metric__memory_usage_ratio",
    "service_metric__net_rx_error_rate",
    "service_metric__net_tx_error_rate",
    "service_log__event_rate",
    "service_log__error_ratio",
]


class ContractV0Error(ValueError):
    """Contract v0 校验失败时抛出。"""


def validate_contract_df(df: pd.DataFrame, config_path: str) -> ContractV0:
    cfg = load_contract_config(config_path)
    feature_cols = [
        f"{mod}__{feat}" for mod, spec in cfg.modalities.items() for feat in spec.features
    ]
    expected = REQUIRED_ID_COLUMNS + feature_cols + REQUIRED_LABEL_COLUMNS

    errors: list[str] = []

    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        # If ID/label columns missing, subsequent checks would KeyError — raise now
        critical_missing = [
            c for c in (REQUIRED_ID_COLUMNS + REQUIRED_LABEL_COLUMNS) if c not in df.columns
        ]
        if critical_missing:
            raise ContractV0Error("contract v0 violations: " + "; ".join(errors))

    if df["sample_id"].duplicated().any():
        errors.append(f"sample_id has {int(df['sample_id'].duplicated().sum())} duplicate values")

    for col in RATE_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if ((vals < 0) | (vals > 1)).any():
            errors.append(f"{col} 越界 [0, 1]，发现 {int(((vals < 0) | (vals > 1)).sum())} 行")

    inconsistent = (df["phase"] == "inject") != df["is_anomaly"]
    if inconsistent.any():
        errors.append(
            f"phase 与 is_anomaly 不一致：{int(inconsistent.sum())} 行 "
            "（is_anomaly 必须等价于 phase == 'inject'）"
        )

    if errors:
        raise ContractV0Error("contract v0 violations: " + "; ".join(errors))

    return ContractV0(df)
