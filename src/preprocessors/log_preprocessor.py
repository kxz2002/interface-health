from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

from src.preprocessors.base import ModalityPreprocessor

logger = logging.getLogger(__name__)

# Spring Boot 日志行：`2026-06-09 10:41:43.491  INFO 1 --- [thread] logger : message`
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+" r"(?P<level>[A-Z]+)\s+\d+\s+---"
)
# K8s pod 目录名后缀：`-<replicaset-hash>-<pod-suffix(5)>`，剥离后得到 canonical service name。
# 限定长度避免误吞 service 名中的合法词段（如 `-route-service`）。
_POD_SUFFIX_RE = re.compile(r"-[a-z0-9]{6,10}-[a-z0-9]{5}$")


class LogPreprocessor(ModalityPreprocessor):
    """v0：Drain3 模板 + (service, 15s window) 粒度的频率/错误率/模板多样性特征。"""

    version = "v0"

    OUTPUT_COLUMNS = [
        "service_log__event_rate",
        "service_log__error_ratio",
        "service_log__template_diversity",
    ]
    WINDOW_MS = 15_000
    ERROR_LEVELS = frozenset({"ERROR", "FATAL"})

    def __init__(self, drain3_state_path: str | Path | None = None):
        self._state_path = Path(drain3_state_path) if drain3_state_path else None
        self._miner: TemplateMiner | None = None

    def _new_miner(self) -> TemplateMiner:
        persistence = FilePersistence(str(self._state_path)) if self._state_path else None
        return TemplateMiner(persistence_handler=persistence, config=TemplateMinerConfig())

    def fit(self, raw_paths: list[Path]) -> None:
        miner = self._new_miner()
        for path in raw_paths:
            for line in Path(path).read_text(errors="replace").splitlines():
                parsed = self._parse_line(line)
                if parsed is None:
                    continue
                _, _, content = parsed
                miner.add_log_message(content)
        # FilePersistence 在 add_log_message 内部按需落盘；显式 save 兜底空输入场景。
        if self._state_path is not None:
            miner.save_state("fit complete")
        self._miner = miner

    def transform(self, log_root_dir: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        if self._miner is None:
            logger.warning(
                "LogPreprocessor.transform 在未 fit 的 miner 上调用，"
                "template_id 将全部为 -1（log 模态信号失效）"
            )
        miner = self._miner if self._miner is not None else self._new_miner()
        log_root_dir = Path(log_root_dir)

        records: list[dict[str, Any]] = []
        for service_dir in sorted(p for p in log_root_dir.iterdir() if p.is_dir()):
            service_name = self._canonical_service_name(service_dir.name)
            for log_file in sorted(service_dir.glob("*.log")):
                for line in log_file.read_text(errors="replace").splitlines():
                    parsed = self._parse_line(line)
                    if parsed is None:
                        continue
                    ts_ms, level, content = parsed
                    cluster = miner.match(content)
                    template_id = cluster.cluster_id if cluster is not None else -1
                    records.append(
                        {
                            "service_name": service_name,
                            "timestamp_window_ms": (ts_ms // self.WINDOW_MS) * self.WINDOW_MS,
                            "level": level,
                            "template_id": template_id,
                        }
                    )

        if not records:
            return pd.DataFrame(
                columns=["service_name", "timestamp_window_ms"] + self.OUTPUT_COLUMNS
            )

        raw = pd.DataFrame.from_records(records)
        return self._aggregate(raw)

    def _aggregate(self, raw: pd.DataFrame) -> pd.DataFrame:
        grouped = raw.groupby(["service_name", "timestamp_window_ms"])
        agg = grouped.agg(
            event_count=("level", "size"),
            error_count=("level", lambda s: s.isin(self.ERROR_LEVELS).sum()),
            unique_templates=("template_id", "nunique"),
        ).reset_index()

        # 填充窗口间的空洞，使无事件窗口显式出现（event_rate=0）
        agg = self._fill_empty_windows(agg)

        agg["service_log__event_rate"] = agg["event_count"].astype(float)
        agg["service_log__error_ratio"] = (agg["error_count"] / agg["event_count"]).where(
            agg["event_count"] > 0, 0.0
        )
        agg["service_log__template_diversity"] = (
            agg["unique_templates"] / agg["event_count"]
        ).where(agg["event_count"] > 0, 0.0)

        return agg[["service_name", "timestamp_window_ms"] + self.OUTPUT_COLUMNS]

    def _fill_empty_windows(self, agg: pd.DataFrame) -> pd.DataFrame:
        filled: list[pd.DataFrame] = []
        for service_name, group in agg.groupby("service_name"):
            lo, hi = group["timestamp_window_ms"].min(), group["timestamp_window_ms"].max()
            full_index = range(int(lo), int(hi) + self.WINDOW_MS, self.WINDOW_MS)
            reindexed = (
                group.set_index("timestamp_window_ms")
                .reindex(full_index, fill_value=0)
                .reset_index()
                .rename(columns={"index": "timestamp_window_ms"})
            )
            reindexed["service_name"] = service_name
            filled.append(reindexed)
        return pd.concat(filled, ignore_index=True)

    def get_feature_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)

    @staticmethod
    def _canonical_service_name(dir_name: str) -> str:
        stripped = _POD_SUFFIX_RE.sub("", dir_name)
        return stripped or dir_name

    @staticmethod
    def _parse_line(line: str) -> tuple[int, str, str] | None:
        m = _LOG_LINE_RE.match(line)
        if m is None:
            return None
        ts = pd.Timestamp(m.group("ts"))
        ts_ms = int(ts.value // 1_000_000)
        content = line[m.end() :].strip()
        return ts_ms, m.group("level"), content
