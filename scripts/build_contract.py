#!/usr/bin/env python
"""离线 pipeline：raw data → contract v0 parquet。

流程：
1. 枚举 data_root 下所有含 _pipeline_out/ 的 case 目录
2. 每个 case 跑 4 个 preprocessor，按 join 键组装宽表
   - endpoint_red：trace + api 按 (endpoint_key, timestamp_window_ms) inner join
   - service_metric / service_log：按 (service_name, timestamp_window_ms) left join
3. 用 Normal case fit Normalizer，对全量 transform（fit 只见正常数据，符合 One-Class 约定）
4. 拼标识列 + 特征列 + 标签列 → 校验 contract → 写 train/eval parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.contracts.contract_config import ContractConfig, load_contract_config
from src.contracts.contract_v0 import RATE_COLUMNS, validate_contract_df
from src.data.normalization import Method, Normalizer, Scope
from src.preprocessors.api_preprocessor import ApiPreprocessor
from src.preprocessors.log_preprocessor import LogPreprocessor
from src.preprocessors.metric_preprocessor import MetricPreprocessor
from src.preprocessors.trace_preprocessor import TracePreprocessor
from src.utils.seed import set_seed

LOG = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EP_TO_SVC_PATH = _REPO_ROOT / "configs/contract/endpoint_to_service.yaml"

# normalization scope 配置字符串 → (Normalizer scope, method)
_SCOPE_MAP: dict[str, tuple[Scope, Method]] = {
    "per_endpoint_min_max": ("per_endpoint", "min_max"),
    "per_service_min_max": ("per_service", "min_max"),
    "global_min_max": ("global", "min_max"),
}


def _enumerate_cases(data_root: Path) -> list[Path]:
    # Anchor on _pipeline_out/ instead of case_metadata.json: works for both
    # mini-fixture layout (JSON at case root) and real data layout (JSON inside
    # trace_data/), and also picks up Normal case which has no metadata file.
    return sorted(p.parent for p in data_root.glob("*/_pipeline_out") if p.is_dir())


def _load_case_meta(case_dir: Path) -> dict:
    # Layout A (mini fixture / legacy): case_metadata.json directly in case dir
    direct = case_dir / "case_metadata.json"
    if direct.exists():
        return json.loads(direct.read_text())
    # Layout B (real anomod data): case_metadata.json inside trace_data/
    in_trace = case_dir / "trace_data" / "case_metadata.json"
    if in_trace.exists():
        return json.loads(in_trace.read_text())
    # Layout C (Normal case in real data): no metadata file; synthesize from dir name.
    return {
        "case_id": case_dir.name,
        "anomaly_type": "Normal",
        "anomaly_level": "none",
        "target_service": None,
        "inject_start_ms": None,
        "inject_end_ms": None,
    }


def _is_normal_case(case_meta: dict) -> bool:
    return str(case_meta.get("anomaly_type", "")).startswith("Normal")


def _all_feature_columns(cfg: ContractConfig) -> list[str]:
    return [f"{mod}__{feat}" for mod, spec in cfg.modalities.items() for feat in spec.features]


def _process_one_case(
    case_dir: Path,
    cfg: ContractConfig,
    ep_to_svc: dict[str, str],
    trace_pre: TracePreprocessor,
    api_pre: ApiPreprocessor,
    metric_pre: MetricPreprocessor,
    log_pre: LogPreprocessor,
) -> pd.DataFrame | None:
    """处理单个 case，返回未归一化的宽表（含标识/特征/标签列）。"""
    case_meta = _load_case_meta(case_dir)
    case_id = case_meta["case_id"]

    pipeline_out = case_dir / "_pipeline_out"
    trace_path = pipeline_out / "tt_traces_red_15s.csv"
    api_path = pipeline_out / "tt_endpoint_health_15s.csv"
    if not trace_path.exists() or not api_path.exists():
        LOG.warning("case %s 缺少 pipeline_out 文件，跳过", case_id)
        return None

    trace_df = trace_pre.transform(trace_path, case_meta)
    api_df = api_pre.transform(api_path, case_meta)

    ep_df = pd.merge(trace_df, api_df, on=["endpoint_key", "timestamp_window_ms"], how="inner")
    if ep_df.empty:
        LOG.warning("case %s endpoint join 结果为空，跳过", case_id)
        return None

    ep_df["service_name"] = ep_df["endpoint_key"].map(ep_to_svc)

    window_start = int(ep_df["timestamp_window_ms"].min())
    window_end = int(ep_df["timestamp_window_ms"].max())

    ep_df = _merge_metric(ep_df, case_dir, case_meta, metric_pre, window_start, window_end)
    ep_df = _merge_log(ep_df, case_dir, case_meta, log_pre)

    _fill_missing_feature_cols(ep_df, cfg)
    _attach_identity_columns(ep_df, case_id)
    _attach_label_columns(ep_df, case_meta)
    return ep_df


def _merge_metric(
    ep_df: pd.DataFrame,
    case_dir: Path,
    case_meta: dict,
    metric_pre: MetricPreprocessor,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    metric_dir = case_dir / "metric_data"
    if not metric_dir.exists():
        return ep_df
    metric_csvs = sorted(metric_dir.glob("*.csv"))
    if not metric_csvs:
        return ep_df

    frames: list[pd.DataFrame] = []
    for idx, csv_path in enumerate(metric_csvs):
        meta = {
            **case_meta,
            # cache key 唯一化：多 CSV 时按序号区分，避免缓存碰撞
            "case_id": f"{case_meta['case_id']}__metric{idx}",
            "trace_window_start_ms": window_start,
            "trace_window_end_ms": window_end,
        }
        try:
            frames.append(metric_pre.transform(csv_path, meta))
        except (ValueError, KeyError, TypeError, pd.errors.ParserError, OSError) as e:
            LOG.exception("metric %s transform 失败，跳过该文件: %s", csv_path.name, e)

    if not frames:
        LOG.warning(
            "case %s 全部 metric CSV 处理失败，metric 模态缺失", case_meta.get("case_id", "unknown")
        )
        return ep_df
    metric_df = pd.concat(frames, ignore_index=True)
    if metric_df.empty:
        return ep_df
    return pd.merge(ep_df, metric_df, on=["service_name", "timestamp_window_ms"], how="left")


def _merge_log(
    ep_df: pd.DataFrame,
    case_dir: Path,
    case_meta: dict,
    log_pre: LogPreprocessor,
) -> pd.DataFrame:
    log_dir = case_dir / "log_data"
    if not log_dir.exists():
        return ep_df
    try:
        log_df = log_pre.transform(log_dir, case_meta)
    except (ValueError, KeyError, TypeError, OSError) as e:
        LOG.exception("log transform 失败，跳过 log 模态: %s", e)
        return ep_df
    if log_df.empty:
        return ep_df
    merged = pd.merge(ep_df, log_df, on=["service_name", "timestamp_window_ms"], how="left")
    log_cols = [c for c in merged.columns if c.startswith("service_log__")]
    if log_cols:
        hit_rate = merged[log_cols[0]].notna().mean()
        if hit_rate < 0.5:
            LOG.warning(
                "log join 命中率 %.1f%%，log 特征可能大面积 NaN（时区或时间对齐问题）",
                hit_rate * 100,
            )
    return merged


def _fill_missing_feature_cols(ep_df: pd.DataFrame, cfg: ContractConfig) -> None:
    for col in _all_feature_columns(cfg):
        if col not in ep_df.columns:
            LOG.warning("特征列 %s 缺失（模态未产出），已填 NaN", col)
            ep_df[col] = float("nan")


def _attach_identity_columns(ep_df: pd.DataFrame, case_id: str) -> None:
    ep_df["case_id"] = case_id
    ep_df["window_str"] = pd.to_datetime(
        ep_df["timestamp_window_ms"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ep_df["sample_id"] = (
        ep_df["case_id"]
        + "__"
        + ep_df["endpoint_key"]
        + "__"
        + ep_df["timestamp_window_ms"].astype(str)
    )


def _attach_label_columns(ep_df: pd.DataFrame, case_meta: dict) -> None:
    inject_start = case_meta.get("inject_start_ms")
    inject_end = case_meta.get("inject_end_ms")
    ts = ep_df["timestamp_window_ms"]

    if inject_start is not None and inject_end is not None:
        phase = pd.Series("baseline", index=ep_df.index, dtype="object")
        phase[(ts >= inject_start) & (ts <= inject_end)] = "inject"
        phase[ts > inject_end] = "recover"
        ep_df["phase"] = phase
    else:
        ep_df["phase"] = "normal"

    ep_df["is_anomaly"] = ep_df["phase"] == "inject"
    # is_train_eligible 标记 non-inject 窗口；train.parquet 目前只取 Normal case（更严格）
    ep_df["is_train_eligible"] = ~ep_df["is_anomaly"]
    ep_df["injection_start_ms"] = inject_start
    ep_df["injection_end_ms"] = inject_end
    ep_df["target_service"] = case_meta.get("target_service")
    ep_df["anomaly_type"] = case_meta.get("anomaly_type", "Normal")
    ep_df["anomaly_level"] = case_meta.get("anomaly_level", "none")


def _build_normalizer_rules(cfg: ContractConfig) -> dict[str, tuple[Scope, Method]]:
    rules: dict[str, tuple[Scope, Method]] = {}
    for mod_name, spec in cfg.modalities.items():
        scope, method = _SCOPE_MAP[spec.normalization]
        for feat in spec.features:
            rules[f"{mod_name}__{feat}"] = (scope, method)
    return rules


def _collect_normal_log_files(cases: list[Path]) -> list[Path]:
    files: list[Path] = []
    for case_dir in cases:
        if not _is_normal_case(_load_case_meta(case_dir)):
            continue
        log_dir = case_dir / "log_data"
        if log_dir.exists():
            files.extend(sorted(log_dir.rglob("*.log")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contract v0 parquet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drain3-state", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    set_seed(args.seed)

    cfg = load_contract_config(args.config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    intermediate_dir = out / "intermediate"

    ep_to_svc = yaml.safe_load(_EP_TO_SVC_PATH.read_text())

    trace_pre = TracePreprocessor()
    api_pre = ApiPreprocessor()
    metric_pre = MetricPreprocessor(intermediate_dir=intermediate_dir)
    log_pre = LogPreprocessor(drain3_state_path=args.drain3_state or (out / "drain3.bin"))

    cases = _enumerate_cases(Path(args.data_root))
    LOG.info("发现 %d 个 case", len(cases))
    if not cases:
        raise RuntimeError(f"data_root {args.data_root} 下未找到任何含 _pipeline_out/ 的 case 目录")

    # Drain3 模板只在 Normal case 上 fit，保证模板字典不被异常日志污染
    normal_log_files = _collect_normal_log_files(cases)
    if normal_log_files:
        log_pre.fit(normal_log_files)
    else:
        LOG.warning("未找到任何 Normal case 日志文件，LogPreprocessor 将在未训练状态下运行")

    frames: list[pd.DataFrame] = []
    for case_dir in cases:
        LOG.info("处理 case: %s", case_dir.name)
        df = _process_one_case(case_dir, cfg, ep_to_svc, trace_pre, api_pre, metric_pre, log_pre)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("所有 case 处理失败，无数据")

    full = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["case_id", "endpoint_key", "timestamp_window_ms"])
        .reset_index(drop=True)
    )

    feature_cols = _all_feature_columns(cfg)
    normal_mask = full["anomaly_type"].str.startswith("Normal")
    if not normal_mask.any():
        raise RuntimeError("无 Normal case，无法 fit Normalizer")

    normalizer = Normalizer(_build_normalizer_rules(cfg))
    # 分组归一化需要 endpoint_key / service_name 列，故传完整子集而非仅特征列
    group_cols = ["endpoint_key", "service_name"]
    normalizer.fit(full[normal_mask].reset_index(drop=True))
    full[feature_cols] = normalizer.transform(full[feature_cols + group_cols])[feature_cols]
    normalizer.save(out / "normalization_stats.json")

    # min-max 在 Normal 上 fit，eval case 的 rate 列可能超出 [0,1]。
    # 这些列受 contract [0,1] 约束，饱和裁剪：超过正常上界即视为完全异常（=1）。
    rate_feature_cols = [c for c in RATE_COLUMNS if c in feature_cols]
    full[rate_feature_cols] = full[rate_feature_cols].clip(lower=0.0, upper=1.0)

    validate_contract_df(full, args.config)

    full[normal_mask].reset_index(drop=True).to_parquet(out / "train.parquet", index=False)
    full.to_parquet(out / "eval_all.parquet", index=False)

    _write_schema(out, cfg)

    LOG.info("完成！train=%d 行，eval_all=%d 行", int(normal_mask.sum()), len(full))


def _write_schema(out: Path, cfg: ContractConfig) -> None:
    (out / "schema.json").write_text(
        json.dumps(
            {
                "contract_version": cfg.contract_version,
                "window_size_s": cfg.window_size_s,
                "feature_dim": sum(len(s.features) for s in cfg.modalities.values()),
                "feature_groups": {
                    name: {
                        "columns": [f"{name}__{f}" for f in spec.features],
                        "preprocessor": f"{spec.preprocessor}@{spec.preprocessor_version}",
                        "normalization_scope": spec.normalization,
                    }
                    for name, spec in cfg.modalities.items()
                },
                "evaluation_strata": ["overall", "by_anomaly_type", "by_anomaly_level"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
