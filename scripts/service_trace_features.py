"""Build service-level trace features and dependency edges from raw SkyWalking JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRACE_FIELDS = (
    "trace_count",
    "span_count",
    "duration_mean_ms",
    "duration_p95_ms",
    "error_rate",
)


def _span_service(span: dict[str, Any]) -> str | None:
    value = span.get("service_code")
    return str(value) if value else None


def read_trace_case(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-window service features and directed service edges.

    The raw collector stores complete trace trees rather than a flat span table.
    We derive edges from explicit ``children_node_ids`` and use span start times
    for 15-second buckets. Missing child references are retained in the quality
    frame so callers can reject incomplete cases instead of silently dropping
    topology evidence.
    """

    payload = json.loads(Path(path).read_text())
    rows: list[dict[str, Any]] = []
    edges: set[tuple[str, str]] = set()
    missing_children = 0
    for trace in payload.get("traces", []):
        spans = trace.get("spans", [])
        by_node = {span.get("node_id"): span for span in spans}
        for span in spans:
            service = _span_service(span)
            start_ms = span.get("start_timestamp_ms")
            if service is None or not isinstance(start_ms, (int, float)):
                continue
            bucket = int(start_ms) // 15_000
            rows.append(
                {
                    "bucket": bucket,
                    "service": service,
                    "trace_id": span.get("trace_id"),
                    "duration_ms": float(span.get("duration_ms") or 0.0),
                    "is_error": bool(span.get("is_error", False)),
                }
            )
            for child_id in span.get("children_node_ids") or []:
                child = by_node.get(child_id)
                if child is None:
                    missing_children += 1
                    continue
                child_service = _span_service(child)
                if child_service is not None and child_service != service:
                    edges.add((service, child_service))

    spans = pd.DataFrame(rows)
    if spans.empty:
        features = pd.DataFrame(columns=["bucket", "service", *TRACE_FIELDS])
    else:
        grouped = spans.groupby(["bucket", "service"], sort=True)
        features = grouped.agg(
            trace_count=("trace_id", "nunique"),
            span_count=("trace_id", "size"),
            duration_mean_ms=("duration_ms", "mean"),
            duration_p95_ms=("duration_ms", lambda x: float(np.percentile(x, 95))),
            error_rate=("is_error", "mean"),
        ).reset_index()
    edge_frame = pd.DataFrame(sorted(edges), columns=["source_service", "target_service"])
    edge_frame.attrs["missing_child_count"] = missing_children
    edge_frame.attrs["trace_count"] = len(payload.get("traces", []))
    return features, edge_frame


def discover_trace_cases(
    raw_root: str | Path, excluded_tokens: tuple[str, ...] = ("POOL", "DNS")
) -> dict[str, Path]:
    """Discover one canonical SkyWalking file per case without loading data."""

    root = Path(raw_root)
    cases: dict[str, Path] = {}
    for path in sorted(root.glob("*/trace_data/*skywalking*.json")):
        case = path.parent.parent.name
        if any(token in case for token in excluded_tokens):
            continue
        if case in cases:
            raise ValueError(f"Multiple trace files found for case {case}")
        cases[case] = path
    return cases
