from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import yaml

from src.preprocessors.base import ModalityPreprocessor

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[2]
_DEFAULT_INTERMEDIATE_DIR = _REPO_ROOT / "artifacts" / "intermediate"

WINDOW_MS = 15_000
WINDOW_S = WINDOW_MS / 1000

# K8s pod 名后缀 `-<replicaset-hash>-<pod-suffix(5)>`，剥离后得 canonical service 名。
# 与 LogPreprocessor 的目录名约定保持一致。
_POD_SUFFIX_RE = re.compile(r"-[a-z0-9]{6,10}-[a-z0-9]{5}$")

# cAdvisor 长格式导出中实际用到的原始 metric_name
_CPU_COUNTER = "container_cpu_usage_seconds_total"
_MEM_USAGE = "container_memory_usage_bytes"
_MEM_LIMIT = "container_spec_memory_limit_bytes"
_PROCESSES = "container_processes"
_NET_RX_ERR = "container_network_receive_errors_total"
_NET_TX_ERR = "container_network_transmit_errors_total"

_POD_METRICS = (_CPU_COUNTER, _MEM_USAGE, _MEM_LIMIT, _PROCESSES)
_HOST_METRICS = (_NET_RX_ERR, _NET_TX_ERR)
_TARGET_METRICS = frozenset(_POD_METRICS + _HOST_METRICS)

# 标签列在原始 CSV 中混入空值与非数值，强制按 string 读，避免 pyarrow 类型推断报错。
_STR_LABEL_COLS = (
    "metric_name",
    "datetime",
    "beta_kubernetes_io_arch",
    "beta_kubernetes_io_os",
    "cpu",
    "device",
    "fstype",
    "id",
    "instance",
    "interface",
    "job",
    "kubernetes_io_arch",
    "kubernetes_io_hostname",
    "kubernetes_io_os",
    "mode",
    "mountpoint",
    "namespace",
    "plugin_name",
    "pod",
    "state",
)


class MetricPreprocessor(ModalityPreprocessor):
    """v0：从 cAdvisor 长格式 metric CSV 抽取 5 维 service 级特征。

    数据现实（real-data 探查，见 CLAUDE.md Known Gotchas）：
    - container_cpu_usage_seconds_total / container_memory_usage_bytes：含 pod 标签，
      可做 service 级（多 pod mean 聚合）。cpu 为累积计数器，需相邻窗口差分求 rate。
    - container_spec_memory_limit_bytes：ts 业务 pod 多数未设 limit（=0），故
      memory_usage_ratio 在 ts service 上恒为 0（无 limit 可归一）。
    - container_processes：含 pod 标签但实测恒为 0，process_count 实际无信号。
    - container_network_*_errors_total：无 pod 标签，仅 host(instance) 级；本环境实测
      全程为 0。实现上对全部 service 广播同一 host 级 rate（保持 5 列 schema 稳定）。
    结论：本数据集下 5 维中仅 cpu_usage_rate 携带信号，其余 4 维为 schema 占位（恒 0）。
    """

    version = "v0"

    OUTPUT_COLUMNS = [
        "service_metric__cpu_usage_rate",
        "service_metric__memory_usage_ratio",
        "service_metric__net_rx_error_rate",
        "service_metric__net_tx_error_rate",
        "service_metric__process_count",
    ]

    def __init__(
        self,
        intermediate_dir: str | Path | None = None,
        endpoint_mapping_path: str | Path | None = None,
    ):
        self._intermediate_dir = (
            Path(intermediate_dir) if intermediate_dir else _DEFAULT_INTERMEDIATE_DIR
        )
        if endpoint_mapping_path is None:
            endpoint_mapping_path = _REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
        mapping = yaml.safe_load(Path(endpoint_mapping_path).read_text())
        self._v0_services = set(mapping.values())

    def fit(self, raw_paths: list[Path]) -> None:
        return  # v0：归一化由独立 Normalizer 负责

    def get_feature_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)

    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        case_id = case_meta["case_id"]
        cache_path = self._intermediate_dir / f"metrics_filtered_{case_id}.parquet"
        if cache_path.exists():
            logger.info("metric cache hit: %s", cache_path)
            return pd.read_parquet(cache_path)

        raw = self._read_filtered(Path(raw_path), case_meta)
        if raw.empty:
            result = pd.DataFrame(
                columns=["service_name", "timestamp_window_ms"] + self.OUTPUT_COLUMNS
            )
        else:
            result = self._build_features(raw)

        self._intermediate_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
        return result

    def _read_filtered(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        """pyarrow chunked reading：仅保留目标 metric 且落在 trace 窗口内的行。"""
        start_s = case_meta.get("trace_window_start_ms", 0) / 1000
        end_s = case_meta.get("trace_window_end_ms", float("inf")) / 1000

        convert = pacsv.ConvertOptions(column_types={c: pa.string() for c in _STR_LABEL_COLS})
        reader = pacsv.open_csv(
            raw_path,
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=convert,
        )
        target_arr = pa.array(_TARGET_METRICS)
        chunks: list[pd.DataFrame] = []
        for batch in reader:
            ts = batch["timestamp"]
            mask = pc.and_(
                pc.greater_equal(ts, start_s),
                pc.less_equal(ts, end_s),
            )
            mask = pc.and_(mask, pc.is_in(batch["metric_name"], value_set=target_arr))
            filtered = batch.filter(mask)
            if filtered.num_rows:
                chunks.append(
                    filtered.select(
                        ["metric_name", "timestamp", "value", "pod", "interface"]
                    ).to_pandas()
                )

        if not chunks:
            return pd.DataFrame(columns=["metric_name", "timestamp", "value", "pod", "interface"])

        df = pd.concat(chunks, ignore_index=True)
        df["timestamp_window_ms"] = (
            (df["timestamp"].astype("int64") * 1000) // WINDOW_MS
        ) * WINDOW_MS
        return df

    def _build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        pod_feats = self._pod_level_features(raw)
        host_feats = self._host_level_net_rates(raw)

        if pod_feats.empty:
            return pd.DataFrame(
                columns=["service_name", "timestamp_window_ms"] + self.OUTPUT_COLUMNS
            )

        # host 级网络 rate 按窗口广播到每个 service
        merged = pod_feats.merge(host_feats, on="timestamp_window_ms", how="left")
        merged[["service_metric__net_rx_error_rate", "service_metric__net_tx_error_rate"]] = merged[
            ["service_metric__net_rx_error_rate", "service_metric__net_tx_error_rate"]
        ].fillna(0.0)
        return merged[["service_name", "timestamp_window_ms"] + self.OUTPUT_COLUMNS]

    def _pod_level_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        pod_rows = raw[raw["metric_name"].isin(_POD_METRICS)].copy()
        pod_rows = pod_rows[pod_rows["pod"].notna() & (pod_rows["pod"] != "")]
        if pod_rows.empty:
            return pd.DataFrame(
                columns=[
                    "service_name",
                    "timestamp_window_ms",
                    "service_metric__cpu_usage_rate",
                    "service_metric__memory_usage_ratio",
                    "service_metric__process_count",
                ]
            )

        pod_rows["service_name"] = pod_rows["pod"].map(self._canonical_service_name)
        pod_rows = pod_rows[pod_rows["service_name"].isin(self._v0_services)]
        if pod_rows.empty:
            return pd.DataFrame(
                columns=[
                    "service_name",
                    "timestamp_window_ms",
                    "service_metric__cpu_usage_rate",
                    "service_metric__memory_usage_ratio",
                    "service_metric__process_count",
                ]
            )

        # 同一 (pod, window) 可能有多条采样，先在 pod 内对每窗口取均值，稳定差分基线。
        per_pod = (
            pod_rows.groupby(["service_name", "pod", "timestamp_window_ms", "metric_name"])["value"]
            .mean()
            .reset_index()
        )
        wide = per_pod.pivot_table(
            index=["service_name", "pod", "timestamp_window_ms"],
            columns="metric_name",
            values="value",
        ).reset_index()

        wide = wide.sort_values(["pod", "timestamp_window_ms"])

        # cpu 是累积计数器：每 pod 相邻窗口差分 / 窗口秒数 → core 利用率。首窗无前值→NaN→0。
        if _CPU_COUNTER in wide:
            cpu_diff = wide.groupby("pod")[_CPU_COUNTER].diff()
            wide["cpu_rate"] = (cpu_diff / WINDOW_S).clip(lower=0).fillna(0.0)
        else:
            wide["cpu_rate"] = 0.0

        if _MEM_USAGE in wide and _MEM_LIMIT in wide:
            # ts 业务 pod 多数未设 memory limit（limit=0），ratio 无定义→0。
            limit = wide[_MEM_LIMIT].where(wide[_MEM_LIMIT] > 0)
            ratio = (wide[_MEM_USAGE] / limit).clip(lower=0, upper=1.0)
            wide["mem_ratio"] = ratio.astype("float64").fillna(0.0)
        else:
            wide["mem_ratio"] = 0.0

        wide["proc_count"] = wide[_PROCESSES] if _PROCESSES in wide else 0.0

        # 跨 pod 聚合到 service 级（mean）
        agg = (
            wide.groupby(["service_name", "timestamp_window_ms"])
            .agg(
                **{
                    "service_metric__cpu_usage_rate": ("cpu_rate", "mean"),
                    "service_metric__memory_usage_ratio": ("mem_ratio", "mean"),
                    "service_metric__process_count": ("proc_count", "mean"),
                }
            )
            .reset_index()
        )
        return agg

    def _host_level_net_rates(self, raw: pd.DataFrame) -> pd.DataFrame:
        """网络错误计数器无 pod 标签：跨 interface 求和后按窗口差分求 rate（host 级）。"""
        out_cols = [
            "timestamp_window_ms",
            "service_metric__net_rx_error_rate",
            "service_metric__net_tx_error_rate",
        ]
        net = raw[raw["metric_name"].isin(_HOST_METRICS)]
        if net.empty:
            return pd.DataFrame(columns=out_cols)

        # 每窗口跨 interface 求和（counter 累积值可加），再相邻差分。
        summed = net.groupby(["metric_name", "timestamp_window_ms"])["value"].sum().reset_index()
        result: dict[str, pd.Series] = {}
        for metric, out_col in (
            (_NET_RX_ERR, "service_metric__net_rx_error_rate"),
            (_NET_TX_ERR, "service_metric__net_tx_error_rate"),
        ):
            sub = summed[summed["metric_name"] == metric].sort_values("timestamp_window_ms")
            rate = (sub["value"].diff() / WINDOW_S).clip(lower=0).fillna(0.0)
            result[out_col] = pd.Series(rate.values, index=sub["timestamp_window_ms"].values)

        windows = sorted(summed["timestamp_window_ms"].unique())
        df = pd.DataFrame({"timestamp_window_ms": windows})
        for out_col, series in result.items():
            df[out_col] = df["timestamp_window_ms"].map(series).fillna(0.0)
        return df

    @staticmethod
    def _canonical_service_name(pod: str) -> str:
        stripped = _POD_SUFFIX_RE.sub("", pod)
        return stripped or pod
