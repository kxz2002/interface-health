from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.preprocessors.base import ModalityPreprocessor

_REPO_ROOT = Path(__file__).parents[2]  # src/preprocessors/ -> src/ -> repo root


class TracePreprocessor(ModalityPreprocessor):
    """v0：从已 pipeline 化的 tt_traces_red_15s.csv 抽取 5 维 trace RED 特征。"""

    version = "v0"

    OUTPUT_COLUMNS = [
        "endpoint_red__trace_request_count",
        "endpoint_red__trace_latency_p50",
        "endpoint_red__trace_latency_p95",
        "endpoint_red__trace_error_rate",
        "endpoint_red__trace_5xx_rate",
    ]

    # tt_traces_red_15s.csv 原始列名 → contract 列名
    _RENAME = {
        "timestamp_window": "timestamp_window_ms",
        "trace_request_count": "endpoint_red__trace_request_count",
        "trace_latency_p50": "endpoint_red__trace_latency_p50",
        "trace_latency_p95": "endpoint_red__trace_latency_p95",
        "trace_error_rate": "endpoint_red__trace_error_rate",
        "trace_5xx_rate": "endpoint_red__trace_5xx_rate",
    }

    def __init__(
        self,
        endpoint_mapping_path: str | Path | None = None,
    ):
        if endpoint_mapping_path is None:
            endpoint_mapping_path = _REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
        self._v0_endpoints = set(yaml.safe_load(Path(endpoint_mapping_path).read_text()).keys())

    def fit(self, raw_paths: list[Path]) -> None:
        return  # v0：归一化由独立 Normalization 步骤负责

    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        raw = pd.read_csv(raw_path)
        df = raw[raw["endpoint_key"].isin(self._v0_endpoints)].copy()
        df = df.rename(columns=self._RENAME)
        return df[["endpoint_key", "timestamp_window_ms"] + self.OUTPUT_COLUMNS]

    def get_feature_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)
