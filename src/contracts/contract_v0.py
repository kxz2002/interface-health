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
    "label_granularity",
    "is_endpoint_anomaly",
    "label_target_observable",
]
# label_granularity 的合法取值（entry 025 从二元 endpoint/case 扩到三元）。校验这个
# 枚举而不是只查列存在性：下游 _write_v1 的 recover 吸收判据、eval 分层都按具体字符串
# 分支，拼错一个值（如写成 "svc"）不会报错，只会让那批 case 静默走进 else 分支、
# 行为与预期相反。
VALID_LABEL_GRANULARITY = frozenset({"endpoint", "service", "case"})
RATE_COLUMNS = [
    "endpoint_red__trace_error_rate",
    "endpoint_red__trace_5xx_rate",
    "endpoint_red__client_error_rate",
    "endpoint_red__client_5xx_rate",
    "endpoint_red__client_body_hash_mismatch_rate",
    "service_metric__cpu_usage_rate",
    "service_metric__memory_usage_ratio",
    "service_metric__net_rx_error_rate",
    "service_metric__net_tx_error_rate",
    "service_log__event_rate",
    "service_log__error_ratio",
]

# v1 专属：ReliabilityGatedFusion 查 EndpointBaselineStats 用的整数 endpoint 标识，
# 不进 v0 校验（REQUIRED_ID_COLUMNS 保持不变，v0 产物无需这一列）。
# 故意不在 validate_contract_df 里校验这个常量——v0/v1 共用同一个校验函数，不引入
# 版本分支逻辑；endpoint_id 的存在性校验由 tests/test_build_contract_v1_endpoint_id.py
# 单独覆盖。这个常量本身仅用于文档化该约定，非死代码。
REQUIRED_ID_COLUMNS_V1_EXTRA = ["endpoint_id"]

# v2 量级 sanity 上界（闭区间），对**全部特征列**生效。v2 的 per-case z-score
# 正常产出是个位数 z 值；1e3 与其隔着 2 个数量级以上，唯一用途是挡住 entry 013
# 那类零方差除数把差值放大到 1e9~1e13 的爆值——它不承担分布合理性校验，故刻意
# 放得很宽。不能只守 rate 列：entry 013 实际爆值的 client_latency_p95 /
# latency_divergence 都不是 rate 列，只守 rate 列会让"防 1e9 重演"的承诺与
# 防线错位。v2 配置守卫已强制所有特征列都是 z-score 产出，全列检查语义成立。
V2_ZSCORE_MAGNITUDE_BOUND = 1e3


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

    is_v2 = cfg.contract_version == "v2"
    if is_v2:
        # v2 per-case z-score 下 rate 列不再落 [0,1]（z 值天然有负有超 1），
        # [0,1] 校验退役；但只放宽尺度、不放弃防线——换成量级 sanity 防
        # entry 013 的 1e9 爆值重演（z-score 内部有退化回退，仍可能出意外）。
        # 检查对象是**全部特征列**而不只是 RATE_COLUMNS：entry 013 实际爆值的
        # client_latency_p95 / latency_divergence 均非 rate 列；配置加载期守卫
        # 已保证 v2 下每个特征列都是 z-score 产出，全列同一量级语义。
        for col in feature_cols:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            exploded = vals.abs() > V2_ZSCORE_MAGNITUDE_BOUND
            if exploded.any():
                errors.append(
                    f"{col} 越过量级 sanity 上界 {V2_ZSCORE_MAGNITUDE_BOUND:g}（绝对值），"
                    f"发现 {int(exploded.sum())} 行，疑似归一化爆值（参见 entry 013）"
                )
    else:
        for col in RATE_COLUMNS:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            out_of_range = (vals < 0) | (vals > 1)
            if out_of_range.any():
                errors.append(f"{col} 越界 [0, 1]，发现 {int(out_of_range.sum())} 行")

    inconsistent = (df["phase"] == "inject") != df["is_anomaly"]
    if inconsistent.any():
        errors.append(
            f"phase 与 is_anomaly 不一致：{int(inconsistent.sum())} 行 "
            "（is_anomaly 必须等价于 phase == 'inject'）"
        )

    bad_granularity = sorted(set(df["label_granularity"].dropna()) - VALID_LABEL_GRANULARITY)
    if bad_granularity:
        errors.append(
            f"label_granularity 出现非法取值 {bad_granularity}，"
            f"合法取值为 {sorted(VALID_LABEL_GRANULARITY)}"
        )

    # is_endpoint_anomaly 必须蕴含 is_anomaly：正样本只能出现在 inject 窗口内。三档
    # label_granularity 的正样本判据都是 is_anomaly 与某个 target 条件的 AND，故该
    # 蕴含关系恒成立；这里校验它是为了挡住未来某档判据被改成不带 is_anomaly 的形式
    # （那会让 baseline/recover 行被标成正样本，评估协议直接失真而不报错）。
    positive_outside_inject = df["is_endpoint_anomaly"].astype(bool) & ~df["is_anomaly"].astype(
        bool
    )
    if positive_outside_inject.any():
        errors.append(
            f"is_endpoint_anomaly 在 inject 窗口外为 True：{int(positive_outside_inject.sum())} 行"
            "（正样本必须是 is_anomaly 的子集）"
        )

    if errors:
        raise ContractV0Error("contract v0 violations: " + "; ".join(errors))

    return ContractV0(df)
