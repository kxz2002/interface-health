from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.preprocessors.base import ModalityPreprocessor


class ApiPreprocessor(ModalityPreprocessor):
    """v0：从 tt_endpoint_health_15s.csv 抽取客户端 RED，并 join 同窗口 trace p95
    派生 latency_divergence。

    与 TracePreprocessor 一样产出 endpoint_red 模态，但负责 client 侧 5 维特征。
    latency_divergence = client_latency_p95 − trace_latency_p95，反映网关/网络延迟，
    是网络类故障的高价值信号。client/trace 时间戳口径不同（health 为 ISO 字符串，
    trace 为 epoch-ms 整数），join 前统一转为 epoch-ms。
    """

    version = "v0"

    OUTPUT_COLUMNS = [
        "endpoint_red__client_request_count",
        "endpoint_red__client_latency_p95",
        "endpoint_red__client_error_rate",
        "endpoint_red__client_5xx_rate",
        "endpoint_red__latency_divergence",
    ]

    _RENAME = {
        "request_count": "endpoint_red__client_request_count",
        "latency_p95": "endpoint_red__client_latency_p95",
        "error_rate": "endpoint_red__client_error_rate",
        "status_5xx_rate": "endpoint_red__client_5xx_rate",
    }

    def __init__(
        self,
        endpoint_mapping_path: str | Path = "configs/contract/endpoint_to_service.yaml",
        trace_red_path: str | Path | None = None,
    ):
        self._v0_endpoints = set(yaml.safe_load(Path(endpoint_mapping_path).read_text()).keys())
        self._trace_red_path = Path(trace_red_path) if trace_red_path is not None else None

    def fit(self, raw_paths: list[Path]) -> None:
        return  # v0：归一化由独立 Normalization 步骤负责

    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        raw_path = Path(raw_path)
        raw = pd.read_csv(raw_path)
        df = raw[raw["endpoint_key"].isin(self._v0_endpoints)].copy()
        df = df.rename(columns=self._RENAME)

        # ISO 字符串 → epoch-ms 整数，与 trace 侧 join 键对齐
        df["timestamp_window_ms"] = (
            pd.to_datetime(df["timestamp_window"], utc=True).astype("int64") // 1_000_000
        )

        df["endpoint_red__latency_divergence"] = self._compute_divergence(df, raw_path)

        return df[["endpoint_key", "timestamp_window_ms"] + self.OUTPUT_COLUMNS].reset_index(
            drop=True
        )

    def _compute_divergence(self, df: pd.DataFrame, raw_path: Path) -> pd.Series:
        trace_path = self._resolve_trace_path(raw_path)
        if trace_path is None or not trace_path.exists():
            return pd.Series(pd.NA, index=df.index, dtype="float64")

        trace = pd.read_csv(
            trace_path, usecols=["timestamp_window", "endpoint_key", "trace_latency_p95"]
        )
        trace = trace.rename(
            columns={
                "timestamp_window": "timestamp_window_ms",
                "trace_latency_p95": "_trace_latency_p95",
            }
        )
        merged = df.merge(
            trace[["timestamp_window_ms", "endpoint_key", "_trace_latency_p95"]],
            on=["timestamp_window_ms", "endpoint_key"],
            how="left",
        )
        return (
            merged["endpoint_red__client_latency_p95"] - merged["_trace_latency_p95"]
        ).to_numpy()

    def _resolve_trace_path(self, raw_path: Path) -> Path | None:
        if self._trace_red_path is not None:
            return self._trace_red_path
        sibling = raw_path.parent / raw_path.name.replace("tt_endpoint_health_", "tt_traces_red_")
        return sibling if sibling != raw_path else None

    def get_feature_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)
