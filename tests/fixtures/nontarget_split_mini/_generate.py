"""生成 nontarget_split_mini fixture 数据。

验证 inject/recover 非目标 endpoint 行时序切分（见
docs/superpowers/plans/2026-07-29-inject-recover-nontarget-split.md）。手写
70+ 行 CSV 易错且难复核，改由本脚本确定性生成；脚本随 fixture 一起提交，改
fixture 时重跑它而不是手改 CSV。

用法（repo 根目录）：
    conda run -n interface python tests/fixtures/nontarget_split_mini/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent

TARGET_EP = "POST:/api/v1/travelservice/trips/left"
NONTARGET_EP = "POST:/api/v1/travel2service/trips/left"

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


def _split_ep(endpoint_key: str) -> tuple[str, str]:
    method, path = endpoint_key.split(":", 1)
    return method, path


def _make_rows(
    case_id: str,
    anomaly_type: str,
    base_ts: int,
    n_windows: int,
    inject_window_idx: set[int],
    recover_window_idx: set[int],
    target_endpoint: str | None,
    target_service: str | None,
    inject_start_ms: int | None,
    inject_end_ms: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """产出 (trace_df, health_df)。每个时间窗都含 TARGET_EP 与 NONTARGET_EP 两行，
    制造 endpoint 级 fan-out——这是本 fixture 存在的理由（现有 fixture 每个故障
    case 只有单 endpoint，非目标候选池恒为空，验证不了新切分）。

    特征值随窗序号 w 与 endpoint 变化，避免 per_endpoint_min_max 归一化在 fit
    集合上零方差退化（退化会触发跳过归一化 + 告警，干扰测试信噪比）。
    inject 窗口内只有目标 endpoint 的特征异常抬高，非目标 endpoint 维持正常量级
    ——非目标 endpoint 在 inject 阶段本来就没有故障，它就是我们要吸收进训练池的
    "正常"数据。
    """
    trace_rows, health_rows = [], []
    for w in range(n_windows):
        ts = base_ts + w * 15_000
        if w in inject_window_idx:
            phase = "inject"
        elif w in recover_window_idx:
            phase = "recover"
        elif inject_start_ms is None:
            phase = "normal"
        else:
            phase = "baseline"

        for ep in (TARGET_EP, NONTARGET_EP):
            method, path = _split_ep(ep)
            is_target = target_endpoint is not None and ep == target_endpoint
            faulty = phase == "inject" and is_target

            ep_offset = 0 if ep == TARGET_EP else 3
            req = 8 + w + ep_offset
            lat_p50 = 20_000.0 + w * 500 + ep_offset * 100
            lat_p95 = lat_p50 * 2
            err = 0.0
            if faulty:
                req += 4
                lat_p50 *= 2.6
                lat_p95 *= 2.1
                err = 0.4

            trace_rows.append(
                {
                    "case_id": case_id,
                    "anomaly_type": anomaly_type,
                    "timestamp_window": ts,
                    "window_str": _iso(ts),
                    "endpoint_key": ep,
                    "method": method,
                    "normalized_path": path,
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
                    "weak_is_anomaly": 1 if faulty else 0,
                    "label_confidence": "strong" if faulty else "normal",
                    "latency_anomaly_signal": 1 if faulty else 0,
                    "error_anomaly_signal": 1 if faulty else 0,
                    "5xx_anomaly_signal": 1 if faulty else 0,
                    "phase": phase,
                    "injection_start_ms": inject_start_ms,
                    "injection_end_ms": inject_end_ms,
                    "target_service": target_service,
                }
            )
            # client 侧（health）比 trace 侧略高，让 latency_divergence 非零非常数
            health_rows.append(
                {
                    "case_id": case_id,
                    "anomaly_type": anomaly_type,
                    "timestamp_window": _iso(ts),
                    "endpoint_key": ep,
                    "request_count": req,
                    "error_rate": err,
                    "latency_mean": lat_p50 * 1.10,
                    "latency_p50": lat_p50 * 1.02,
                    "latency_p95": lat_p95 + 1_500 + w * 50,
                    "latency_p99": lat_p95 + 2_000 + w * 50,
                    "status_2xx_rate": 1.0 - err,
                    "status_4xx_rate": 0.0,
                    "status_5xx_rate": err,
                    "method": method,
                    "normalized_path": path,
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
    # Normal：6 窗 × 2 endpoint = 12 行。split_normal_rows_temporal 默认
    # fit_frac=0.6 / val_frac=0.2 → int(6*0.6)=3 窗 train_fit（6 行）、
    # int(6*0.2)=1 窗 train_val（2 行）、剩 2 窗 eval_normal_holdout（4 行）。
    # 必须覆盖故障 case 用到的两个 endpoint，否则 Normalizer/EndpointBaselineStats
    # 的 fit 集合缺 endpoint。
    normal_trace, normal_health = _make_rows(
        case_id="Normal",
        anomaly_type="Normal",
        base_ts=1_796_000_000_000,
        n_windows=6,
        inject_window_idx=set(),
        recover_window_idx=set(),
        target_endpoint=None,
        target_service=None,
        inject_start_ms=None,
        inject_end_ms=None,
    )
    # Normal case 的 trace CSV 不应有 is_target_endpoint 列（真实 Normal 数据没有
    # 注入目标），与 split_fraction_mini/Normal 的既有形态一致
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

    # Lv_E_NTGT_travel：endpoint 级精确标签（含 target_endpoint）。
    # 15 窗 = 窗 0-4 baseline / 窗 5-9 inject / 窗 10-14 recover。
    # 每阶段 5 窗，fraction=0.2 → int(5*0.2)=1 窗进 train，能验证非零分数切分。
    ntgt_base = 1_796_100_000_000
    ntgt_trace, ntgt_health = _make_rows(
        case_id="Lv_E_NTGT_travel",
        anomaly_type="Lv_E_NTGT_travel",
        base_ts=ntgt_base,
        n_windows=15,
        inject_window_idx=set(range(5, 10)),
        recover_window_idx=set(range(10, 15)),
        target_endpoint=TARGET_EP,
        target_service="ts-travel-service",
        inject_start_ms=ntgt_base + 5 * 15_000,
        inject_end_ms=ntgt_base + 9 * 15_000,
    )
    _write_case(
        _HERE / "Lv_E_NTGT_travel",
        {
            "case_id": "Lv_E_NTGT_travel",
            "anomaly_type": "Lv_E_NTGT_travel",
            "anomaly_level": "endpoint",
            "target_service": "ts-travel-service",
            "target_endpoint": TARGET_EP,
            "inject_start_ms": ntgt_base + 5 * 15_000,
            "inject_end_ms": ntgt_base + 9 * 15_000,
        },
        ntgt_trace,
        ntgt_health,
    )

    # Lv_D_CASELVL_travel：service 级标签（有 target_service、**无** target_endpoint）。
    # anomaly_level 故意写成 "endpoint" 而与缺失的 target_endpoint 矛盾——
    # CLAUDE.md 明确 label_granularity 的判据是 target 字段而非 anomaly_level，
    # 这个 fixture 主动钉死实现不会退回用 anomaly_level 判断。
    # target_service=ts-travel-service 命中 TARGET_EP 所属 service，NONTARGET_EP
    # （travel2）不命中，构成 service 级 fan-out：inject 阶段前者是正样本、后者是
    # 可吸收的非目标行。entry 025 前两者都被 case 级 fallback 误标为正样本。
    case_base = 1_796_200_000_000
    case_trace, case_health = _make_rows(
        case_id="Lv_D_CASELVL_travel",
        anomaly_type="Lv_D_CASELVL_travel",
        base_ts=case_base,
        n_windows=15,
        inject_window_idx=set(range(5, 10)),
        recover_window_idx=set(range(10, 15)),
        target_endpoint=None,
        target_service="ts-travel-service",
        inject_start_ms=case_base + 5 * 15_000,
        inject_end_ms=case_base + 9 * 15_000,
    )
    # case 级标签的真实数据（如 anomod_v1）trace CSV 里没有 is_target_endpoint 列，
    # TracePreprocessor 会填 False。这里删掉该列以复刻真实形态。
    case_trace = case_trace.drop(columns=["is_target_endpoint"])
    _write_case(
        _HERE / "Lv_D_CASELVL_travel",
        {
            "case_id": "Lv_D_CASELVL_travel",
            "anomaly_type": "Lv_D_CASELVL_travel",
            "anomaly_level": "endpoint",
            "target_service": "ts-travel-service",
            "inject_start_ms": case_base + 5 * 15_000,
            "inject_end_ms": case_base + 9 * 15_000,
        },
        case_trace,
        case_health,
    )

    # Lv_D_UNOBSERVABLE_mysql：target_service 落在 endpoint→service 映射覆盖范围外
    # （tsdb-mysql 不是任何客户端 endpoint 的宿主 service），复刻真实数据里 Lv_D_* 与
    # Lv_S_KILLPOD_gateway 的形态。预期行为（entry 025 决策 2a）：正样本恒为 0、
    # label_target_observable=False、分层 AUROC 为 null，且其 inject 行**不**参与训练池
    # 吸收——后者是关键，若少了 label_target_observable 这道闸，该 case 全部 inject 行
    # （正样本为 0 → ~is_endpoint_anomaly 恒 True）会在 fraction=1.0 下整段被吸进训练池，
    # 把"数据库挂掉时 8 个 service 全受影响"的故障数据当成正常数据喂给 One-Class 模型。
    unobs_base = 1_796_300_000_000
    unobs_trace, unobs_health = _make_rows(
        case_id="Lv_D_UNOBSERVABLE_mysql",
        anomaly_type="Lv_D_UNOBSERVABLE_mysql",
        base_ts=unobs_base,
        n_windows=15,
        inject_window_idx=set(range(5, 10)),
        recover_window_idx=set(range(10, 15)),
        target_endpoint=None,
        target_service="tsdb-mysql",
        inject_start_ms=unobs_base + 5 * 15_000,
        inject_end_ms=unobs_base + 9 * 15_000,
    )
    unobs_trace = unobs_trace.drop(columns=["is_target_endpoint"])
    _write_case(
        _HERE / "Lv_D_UNOBSERVABLE_mysql",
        {
            "case_id": "Lv_D_UNOBSERVABLE_mysql",
            "anomaly_type": "Lv_D_UNOBSERVABLE_mysql",
            "anomaly_level": "database",
            "target_service": "tsdb-mysql",
            "inject_start_ms": unobs_base + 5 * 15_000,
            "inject_end_ms": unobs_base + 9 * 15_000,
        },
        unobs_trace,
        unobs_health,
    )


if __name__ == "__main__":
    main()
