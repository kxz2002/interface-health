"""生成 v2_fit_scope_mini fixture 数据。

contract v2 的 Normalizer fit 行集泄漏断言专用（Task 12，
tests/test_build_contract_v2_fit_scope.py）：需要 (a) 每个故障 case ≥10 个
baseline 窗，使 int(10*0.2)=2 窗的分数切分非空且便于手算；(b) 单 endpoint，
使"每窗行数"恒为 1，fit 行集可以直接按时窗精确手算（多 endpoint fan-out 会
让样本数计算依赖 join 细节，模糊断言意图）；(c) 故障 case 无 inject/recover
窗——泄漏断言只关心 baseline 前 20% 的归属，inject/recover 行与此无关，
省略后 fixture 更小、手算预期更直观。

nontarget_split_mini 已有多 endpoint fan-out，但每 case 只有 5 个 baseline
窗（0.2→1 窗，且 60% 的窗是 inject/recover），不符合本 fixture 的诉求，故
参照它的生成器模式新造。手写几十行 CSV 易错，改由本脚本确定性生成；改 fixture
时重跑它而不是手改 CSV。

用法（repo 根目录）：
    conda run -n interface python tests/fixtures/v2_fit_scope_mini/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent

_EP = "POST:/api/v1/travelservice/trips/left"

_TRACE_COLUMNS = [
    "case_id",
    "anomaly_type",
    "timestamp_window",
    "window_str",
    "endpoint_key",
    "method",
    "normalized_path",
    "trace_request_count",
    "trace_latency_mean",
    "trace_latency_p50",
    "trace_latency_p95",
    "trace_latency_p99",
    "trace_error_rate",
    "trace_5xx_rate",
    "trace_4xx_rate",
    "trace_status_coverage",
    "is_target_endpoint",
    "weak_is_anomaly",
    "label_confidence",
    "latency_anomaly_signal",
    "error_anomaly_signal",
    "5xx_anomaly_signal",
    "phase",
    "injection_start_ms",
    "injection_end_ms",
    "target_service",
]

_HEALTH_COLUMNS = [
    "case_id",
    "anomaly_type",
    "timestamp_window",
    "endpoint_key",
    "request_count",
    "error_rate",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "status_2xx_rate",
    "status_4xx_rate",
    "status_5xx_rate",
    "method",
    "normalized_path",
]


def _iso(ts_ms: int) -> str:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_rows(
    case_id: str,
    anomaly_type: str,
    base_ts: int,
    n_windows: int,
    is_target: bool,
    inject_start_ms: int | None,
    inject_end_ms: int | None,
    target_service: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """单 endpoint、每窗一行。特征值随窗序号 w 单调变化，避免归一化零方差退化。"""
    trace_rows, health_rows = [], []
    for w in range(n_windows):
        ts = base_ts + w * 15_000
        phase = "normal" if inject_start_ms is None else "baseline"

        req = 8 + w
        lat_p50 = 20_000.0 + w * 500
        lat_p95 = lat_p50 * 2
        err = 0.0

        trace_rows.append(
            {
                "case_id": case_id,
                "anomaly_type": anomaly_type,
                "timestamp_window": ts,
                "window_str": _iso(ts),
                "endpoint_key": _EP,
                "method": "POST",
                "normalized_path": "/api/v1/travelservice/trips/left",
                "trace_request_count": req,
                "trace_latency_mean": lat_p50 * 1.05,
                "trace_latency_p50": lat_p50,
                "trace_latency_p95": lat_p95,
                "trace_latency_p99": lat_p95,
                "trace_error_rate": err,
                "trace_5xx_rate": err,
                "trace_4xx_rate": 0.0,
                "trace_status_coverage": 1.0,
                "is_target_endpoint": is_target,
                "weak_is_anomaly": 0,
                "label_confidence": "normal",
                "latency_anomaly_signal": 0,
                "error_anomaly_signal": 0,
                "5xx_anomaly_signal": 0,
                "phase": phase,
                "injection_start_ms": inject_start_ms,
                "injection_end_ms": inject_end_ms,
                "target_service": target_service,
            }
        )
        health_rows.append(
            {
                "case_id": case_id,
                "anomaly_type": anomaly_type,
                "timestamp_window": _iso(ts),
                "endpoint_key": _EP,
                "request_count": req,
                "error_rate": err,
                "latency_mean": lat_p50 * 1.10,
                "latency_p50": lat_p50 * 1.02,
                "latency_p95": lat_p95 + 1_500 + w * 50,
                "latency_p99": lat_p95 + 2_000 + w * 50,
                "status_2xx_rate": 1.0 - err,
                "status_4xx_rate": 0.0,
                "status_5xx_rate": err,
                "method": "POST",
                "normalized_path": "/api/v1/travelservice/trips/left",
            }
        )

    return (
        pd.DataFrame(trace_rows, columns=_TRACE_COLUMNS),
        pd.DataFrame(health_rows, columns=_HEALTH_COLUMNS),
    )


def _write_case(
    case_dir: Path, meta: dict, trace_df: pd.DataFrame, health_df: pd.DataFrame
) -> None:
    out = case_dir / "_pipeline_out"
    out.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    trace_df.to_csv(out / "tt_traces_red_15s.csv", index=False)
    health_df.to_csv(out / "tt_endpoint_health_15s.csv", index=False)


def main() -> None:
    # Normal：10 窗 × 1 endpoint = 10 行。split_normal_rows_temporal 默认
    # fit_frac=0.6 / val_frac=0.2 → int(10*0.6)=6 窗 train_fit、
    # int(10*0.2)=2 窗 train_val、剩 2 窗 holdout。泄漏断言的预期 fit 行集
    # 直接取最早 6 个窗（w=0..5），手算无歧义。真实 Normal 数据的 trace CSV
    # 无 is_target_endpoint 列，下面删掉复刻该形态。
    normal_trace, normal_health = _make_rows(
        case_id="Normal",
        anomaly_type="Normal",
        base_ts=1_796_000_000_000,
        n_windows=10,
        is_target=False,
        inject_start_ms=None,
        inject_end_ms=None,
        target_service=None,
    )
    normal_trace = normal_trace.drop(columns=["is_target_endpoint"])
    _write_case(
        _HERE / "Normal",
        {
            "case_id": "Normal",
            "anomaly_type": "Normal",
            "anomaly_level": "none",
            "target_service": None,
            "inject_start_ms": None,
            "inject_end_ms": None,
        },
        normal_trace,
        normal_health,
    )

    # 两个故障 case：各 10 个 baseline 窗、无 inject/recover 窗（inject 边界
    # 声明在全部窗口之后，故 phase 全为 baseline；泄漏断言不涉及 inject 行）。
    # fault_baseline_train_fraction=0.2 → int(10*0.2)=2 个最早窗进 fit 行集，
    # 其余 8 窗必须留在 eval_all 且不参与任何归一化统计量。
    for case_id, base_ts in (
        ("Lv_A_FAULTONE_travel", 1_796_100_000_000),
        ("Lv_B_FAULTTWO_travel", 1_796_200_000_000),
    ):
        # 边界放在最后一窗之后：所有行 ts < inject_start → 全 baseline
        inject_start = base_ts + 10 * 15_000
        inject_end = base_ts + 15 * 15_000
        trace, health = _make_rows(
            case_id=case_id,
            anomaly_type=case_id,
            base_ts=base_ts,
            n_windows=10,
            is_target=True,
            inject_start_ms=inject_start,
            inject_end_ms=inject_end,
            target_service="ts-travel-service",
        )
        _write_case(
            _HERE / case_id,
            {
                "case_id": case_id,
                "anomaly_type": case_id,
                "anomaly_level": "endpoint",
                "target_service": "ts-travel-service",
                "target_endpoint": _EP,
                "inject_start_ms": inject_start,
                "inject_end_ms": inject_end,
            },
            trace,
            health,
        )


if __name__ == "__main__":
    main()
