"""Read-only resource observations; query timestamps do not establish freshness.

CPU rates estimate the interval ending at the query point, rather than exact
15 second bucket integrals. No lifecycle/phase evidence enters aggregation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src.service.protocol import (
    FEATURE_VALUE_COLUMNS,
    ServiceProtocol,
    canonical_service,
    make_grid,
    validate_features,
)

CPU = "container_cpu_usage_seconds_total"
METRICS = {
    CPU: "cpu_cores",
    "container_memory_usage_bytes": "memory_usage_bytes",
    "container_memory_working_set_bytes": "memory_working_set_bytes",
    "container_spec_memory_limit_bytes": "memory_limit_bytes",
    "container_processes": "process_count",
}
RAW_COLUMNS = [
    "metric_name",
    "timestamp",
    "value",
    "namespace",
    "pod",
    "container",
    "id",
    "instance",
    "job",
    "cpu",
]
IDENTITY = ["namespace", "pod", "container", "id"]
SERIES = ["case_id", *IDENTITY, "instance", "job", "metric_name", "cpu"]
QUALITY_COLUMNS = [
    "case_id",
    "canonical_service_id",
    "window_start_ms",
    "namespace",
    "pod",
    "container",
    "container_id",
    "exporter_instance",
    "job",
    "metric_name",
    "cpu",
    "query_time_ms",
    "value",
    "dedup_count",
    "counter_previous_time_ms",
    "counter_previous_value",
    "counter_dt_s",
    "counter_delta",
    "series_status",
    "source_path",
    "source_row_number",
]


def _case_for_path(path: Path, protocol: ServiceProtocol) -> str:
    candidates = [
        c["case_id"]
        for c in protocol.config["cases"]
        if c["case_id"] in path.parts or path.name.startswith(c["case_id"] + "_")
    ]
    if len(candidates) != 1:
        raise ValueError(f"Metric path must identify exactly one configured case: {path}")
    return candidates[0]


def iter_metric_batches(paths: list[Path], protocol: ServiceProtocol) -> Iterator[pd.DataFrame]:
    """Project necessary CSV columns and retain one-based data-row provenance.

    Only the supplied files are opened. Case intervals include a 30 second
    causal predecessor budget; unrelated metrics/entities are discarded per batch.
    """
    cfg = protocol.config
    identity = cfg["identity"]
    for path in paths:
        path = Path(path)
        case_id = _case_for_path(path, protocol)
        case = next(c for c in cfg["cases"] if c["case_id"] == case_id)
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=1 << 20),
            convert_options=pacsv.ConvertOptions(
                include_columns=RAW_COLUMNS,
                include_missing_columns=True,
                column_types={c: pa.string() for c in RAW_COLUMNS},
                strings_can_be_null=False,
            ),
        )
        offset = 0
        for batch in reader:
            frame = batch.to_pandas()
            frame["source_row_number"] = np.arange(offset + 1, offset + len(frame) + 1)
            offset += len(frame)
            frame = frame.fillna("")
            frame = frame[frame.metric_name.isin(METRICS)].copy()
            if frame.empty:
                continue
            frame["namespace"] = frame.namespace.replace("", identity["default_namespace"])
            frame["job"] = frame.job.replace("", identity["default_job"])
            frame["canonical_service_id"] = frame.pod.map(
                lambda p: canonical_service(p, cfg["registry"])
            )
            mapping = identity["application_container_mapping"]
            selected = frame.canonical_service_id.isin(cfg["services"])
            selected &= frame.namespace.isin(identity["allowed_namespaces"])
            selected &= frame.job.isin(identity["allowed_jobs"])
            selected &= frame.container.eq(frame.canonical_service_id.map(mapping))
            frame = frame[selected].copy()
            times = (pd.to_numeric(frame.timestamp, errors="coerce") * 1000).round()
            selected = (
                np.isfinite(times)
                & times.ge(case["window_start_ms"] - 30000)
                & times.lt(case["window_end_ms"])
            )
            frame = frame[selected].copy()
            frame["query_time_ms"] = times[selected].round().astype("int64")
            frame["value"] = pd.to_numeric(frame.value, errors="coerce").astype("float64")
            frame["case_id"] = case_id
            frame["source_path"] = str(path)
            if not frame.empty:
                yield frame.drop(columns="timestamp")


def _series_observations(raw: pd.DataFrame, protocol: ServiceProtocol) -> pd.DataFrame:
    """Deduplicate full series/time, then compute strictly causal counter rates."""
    result = []
    for _, group in raw.groupby(SERIES, sort=True, dropna=False):
        previous_time = None
        previous_value = np.nan
        for time, samples in group.groupby("query_time_ms", sort=True):
            values = samples.value.to_numpy()
            conflict = len(np.unique(values)) != 1
            value = values[0] if not conflict else np.nan
            record = samples.iloc[0].to_dict()
            record.update(
                value=value,
                dedup_count=len(samples),
                counter_previous_time_ms=previous_time,
                counter_previous_value=previous_value,
                counter_dt_s=np.nan,
                counter_delta=np.nan,
                series_status="OBSERVED",
                feature_value=value,
            )
            # Preserve every source for conflicts and duplicates in the diagnostic table.
            record["sources"] = list(
                zip(samples.source_path, samples.source_row_number, samples.value)
            )
            if conflict:
                record["series_status"] = "CONFLICTING_SAMPLE"
            elif not np.isfinite(value) or value < 0:
                record["series_status"] = "UNSUPPORTED"
            elif record["metric_name"] == CPU:
                if not record["id"]:
                    record["series_status"] = "UNSUPPORTED"
                elif previous_time is None or not np.isfinite(previous_value):
                    record["series_status"] = "UNOBSERVED"
                else:
                    dt = (time - previous_time) / 1000
                    delta = float(value) - float(previous_value)
                    record.update(counter_dt_s=dt, counter_delta=delta)
                    if dt <= 0:
                        record["series_status"] = "UNSUPPORTED"
                    elif dt > protocol.config["max_counter_gap_s"]:
                        record["series_status"] = "GAP"
                    elif delta < 0:
                        record["series_status"] = "COUNTER_RESET"
                    else:
                        rate = delta / dt
                        if np.isfinite(rate):
                            record["feature_value"] = rate
                        else:
                            record["series_status"] = "UNSUPPORTED"
            if record["series_status"] != "OBSERVED":
                record["feature_value"] = np.nan
            result.append(record)
            previous_time, previous_value = time, value
    return pd.DataFrame(result)


def _mark_identity_episodes(observations: pd.DataFrame, transition_ms: int) -> pd.DataFrame:
    """Keep each causal observation episode's first point fixed during overlap.

    A known identity returns in a new episode only when it has left the recent
    observation budget and a different identity was observed since its last point.
    Missing records alone never create a switch. Simultaneous samples are processed
    together, so metric/file ordering cannot alter episode starts.
    """
    marked = observations.copy()
    marked["transition_start_ms"] = pd.Series(pd.NA, index=marked.index, dtype="Int64")
    for _, service in marked.groupby("canonical_service_id", sort=False):
        last_seen: dict[tuple[str, ...], int] = {}
        starts: dict[tuple[str, ...], int | None] = {}
        for time, snapshot in service.groupby("query_time_ms", sort=True):
            time = int(time)
            identities = {
                tuple(row) for row in snapshot[IDENTITY].itertuples(index=False, name=None)
            }
            for ident in identities:
                prior = last_seen.get(ident)
                other_since = any(
                    other != ident and (prior is None or seen >= prior)
                    for other, seen in last_seen.items()
                )
                new_episode = prior is None or time - prior > transition_ms
                if new_episode and other_since:
                    starts[ident] = time
                elif ident not in starts:
                    starts[ident] = None
                indices = snapshot.index[
                    snapshot[IDENTITY].apply(tuple, axis=1).map(lambda value: value == ident)
                ]
                if starts[ident] is not None:
                    marked.loc[indices, "transition_start_ms"] = starts[ident]
            # Update only after all identities at this timestamp were considered.
            last_seen.update({ident: time for ident in identities})
    return marked


def build_metric_windows(
    paths: list[Path], meta: dict[str, object], protocol: ServiceProtocol
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build complete independent Service grid and auditable instance observations."""
    cfg = protocol.config
    grid = make_grid(meta, cfg["services"])
    grid["score_time_lower_bound_ms"] = grid.window_end_ms
    grid["schema_version"] = cfg["schema"]["version"]
    # Iterator performs frozen-case coarse filtering; this call can narrow it further.
    chunks = list(iter_metric_batches(paths, protocol))
    raw = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if not raw.empty:
        raw = raw[
            raw.case_id.eq(meta["case_id"])
            & raw.query_time_ms.ge(meta["window_start_ms"] - 30000)
            & raw.query_time_ms.lt(meta["window_end_ms"])
        ].copy()
    obs = _series_observations(raw, protocol) if not raw.empty else pd.DataFrame()
    if not obs.empty:
        obs = _mark_identity_episodes(obs, cfg["transition_ms"])
    records = []
    for _, key in grid.iterrows():
        record = key.to_dict()
        for name in FEATURE_VALUE_COLUMNS:
            record[name], record[name + "__status"] = np.nan, "UNOBSERVED"
        start, end = int(key.window_start_ms), int(key.window_end_ms)
        all_service = (
            obs[obs.canonical_service_id.eq(key.canonical_service_id)] if not obs.empty else obs
        )
        current = (
            all_service[all_service.query_time_ms.ge(start) & all_service.query_time_ms.lt(end)]
            if not all_service.empty
            else all_service
        )
        recent = (
            all_service[
                all_service.query_time_ms.ge(end - cfg["transition_ms"])
                & all_service.query_time_ms.lt(end)
            ]
            if not all_service.empty
            else all_service
        )
        count = len(current[IDENTITY].drop_duplicates()) if not current.empty else 0
        recent_count = len(recent[IDENTITY].drop_duplicates()) if not recent.empty else 0
        conflict = not current.empty and current.series_status.eq("CONFLICTING_SAMPLE").any()
        record.update(
            observed_instance_count=count,
            identity_ambiguity=bool(recent_count > 1 or conflict),
            metric_last_query_time_ms=int(current.query_time_ms.max()) if count else None,
            metric_receipt_time_ms=None,
            resource_valid=False,
            quality_reasons=[],
        )
        override = None
        if count:
            if recent_count > 1:
                override = "IDENTITY_AMBIGUOUS"
            else:
                latest = current.loc[current.query_time_ms.idxmax()]
                switch_time = latest.transition_start_ms
                if pd.notna(switch_time) and start < int(switch_time) + cfg["transition_ms"]:
                    override = "TRANSITION"
            if override:
                for name in FEATURE_VALUE_COLUMNS:
                    record[name + "__status"] = override
            else:
                candidates = current.copy()
                # Preference is causal: total observed through this window wins for
                # the same full exporter series identity, never a future total label.
                cpu = candidates.metric_name.eq(CPU)
                for _, cpu_group in candidates[cpu].groupby(
                    IDENTITY + ["instance", "job"], dropna=False
                ):
                    base = all_service[all_service.query_time_ms.lt(end)]
                    for column in IDENTITY + ["instance", "job"]:
                        base = base[base[column].eq(cpu_group.iloc[0][column])]
                    labels = set(base.loc[base.metric_name.eq(CPU), "cpu"])
                    preferred = "total" if "total" in labels else ""
                    candidates = candidates.drop(cpu_group.index[~cpu_group.cpu.eq(preferred)])
                for metric, name in METRICS.items():
                    selected = candidates[candidates.metric_name.eq(metric)]
                    if selected.empty:
                        continue
                    if selected.series_status.eq("CONFLICTING_SAMPLE").any():
                        record[name + "__status"] = "CONFLICTING_SAMPLE"
                        continue
                    latest = selected[selected.query_time_ms.eq(selected.query_time_ms.max())]
                    if (
                        len(latest) > 1
                        and len(set(zip(latest.feature_value, latest.series_status))) > 1
                    ):
                        record[name + "__status"] = "CONFLICTING_SAMPLE"
                        record["identity_ambiguity"] = True
                        continue
                    last = latest.iloc[0]
                    record[name], record[name + "__status"] = last.feature_value, last.series_status
                usage, limit = record["memory_usage_bytes"], record["memory_limit_bytes"]
                us, ls = record["memory_usage_bytes__status"], record["memory_limit_bytes__status"]
                if us != "OBSERVED":
                    record["memory_usage_ratio__status"] = us
                elif ls != "OBSERVED":
                    record["memory_usage_ratio__status"] = ls
                elif limit <= 0:
                    record["memory_usage_ratio__status"] = "NO_DENOMINATOR"
                else:
                    ratio = float(usage) / float(limit)
                    if np.isfinite(ratio):
                        record["memory_usage_ratio"] = ratio
                        record["memory_usage_ratio__status"] = "OBSERVED"
                    else:
                        record["memory_usage_ratio__status"] = "UNSUPPORTED"
        record["quality_reasons"] = sorted(
            {
                record[n + "__status"]
                for n in FEATURE_VALUE_COLUMNS
                if record[n + "__status"] != "OBSERVED"
            }
        )
        record["resource_valid"] = (
            not record["identity_ambiguity"]
            and count == 1
            and all(
                record[n + "__status"] == "OBSERVED" and np.isfinite(record[n])
                for n in cfg["feature_groups"]["R3"]
            )
        )
        records.append(record)
    features = pd.DataFrame(
        records,
        columns=list(grid.columns)
        + [c for name in FEATURE_VALUE_COLUMNS for c in (name, name + "__status")]
        + [
            "observed_instance_count",
            "identity_ambiguity",
            "metric_last_query_time_ms",
            "metric_receipt_time_ms",
            "resource_valid",
            "quality_reasons",
        ],
    )
    for name in FEATURE_VALUE_COLUMNS:
        features[name] = features[name].astype("float64")
    for name in ["observed_instance_count", "metric_last_query_time_ms", "metric_receipt_time_ms"]:
        features[name] = features[name].astype("Int64")
    for name in ["identity_ambiguity", "resource_valid"]:
        features[name] = features[name].astype(bool)
    features.attrs["partial_windows"] = {
        "leading_partial_ms": (-int(meta["window_start_ms"])) % cfg["window_ms"],
        "trailing_partial_ms": int(meta["window_end_ms"]) % cfg["window_ms"],
    }
    features.attrs["timestamp_semantics"] = (
        "Prometheus query_range timestamp; scrape freshness and receipt time unknown"
    )
    quality_records = []
    if not obs.empty:
        for _, row in obs.iterrows():
            record = row.to_dict()
            record["window_start_ms"] = (
                int(row.query_time_ms) // cfg["window_ms"] * cfg["window_ms"]
            )
            record["container_id"], record["exporter_instance"] = row["id"], row["instance"]
            for source_path, source_row_number, source_value in row.sources:
                quality_records.append(
                    {
                        **record,
                        "source_path": source_path,
                        "source_row_number": source_row_number,
                        "value": source_value,
                    }
                )
    quality = pd.DataFrame(quality_records, columns=QUALITY_COLUMNS)
    for name in [
        "window_start_ms",
        "query_time_ms",
        "dedup_count",
        "counter_previous_time_ms",
        "source_row_number",
    ]:
        quality[name] = quality[name].astype("Int64")
    validate_features(features, protocol)
    return features, quality
