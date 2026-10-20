"""Explicit, assumption-bound SN adapter; no labels, fitting or physical conversion."""

from __future__ import annotations

import codecs
import csv
import datetime as dt
import hashlib
import io
import json
import math
import re
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from src.service.modality_preparation import (
    CONTINUATION,
    GRID_COLUMNS,
    execution_windows,
    full_grid,
    log_windows,
)
from src.service.preparation import file_digest
from src.service.sn_preparation import SN_MD5, iter_json_array

RESOURCE_FIELDS = [
    "cpu_usage_system",
    "cpu_usage_total",
    "cpu_usage_user",
    "memory_usage",
    "memory_working_set",
    "rx_bytes",
    "tx_bytes",
]
OFFSET_MS = -28800000
LOG_HEAD = re.compile(r"^\[([^\]]+)\]\s*(.*)$", re.DOTALL)
MONTHS = {
    name: i
    for i, name in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1
    )
}


def validate_profile(profile: dict) -> None:
    if (
        profile.get("profile") != "sn_exploratory_v1"
        or profile.get("offset_seconds") != -28800
        or profile.get("alignment_assumption") is not True
        or profile.get("archive_md5") != SN_MD5
        or profile.get("resource_unit") != "unknown"
        or profile.get("resource_measurement_semantics") != "unknown"
        or profile.get("resource_aggregation")
        != "raw_sequence_and_pure_mathematical_mean_no_rate_no_diff"
        or profile.get("execution_object") != "Span"
        or len(profile.get("services", [])) != 12
        or len(set(profile["services"])) != 12
        or len(profile.get("runs", {})) != 7
        or len({r["container"] for r in profile["runs"].values()}) != 4
    ):
        raise ValueError("Invalid explicit SN exploratory profile")


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text())
    validate_profile(profile)
    return profile


def span_windows(grid: pd.DataFrame, traces, member: str) -> tuple[pd.DataFrame, dict]:
    references = {}
    run_id = grid.run_id.iloc[0]
    services = set(grid.canonical_service_id)
    lower, upper = int(grid.window_start_ms.min()), int(grid.window_end_ms.max())

    def observations():
        for trace_index, trace in enumerate(traces):
            for span_index, span in enumerate(trace.get("spans", [])):
                start, duration = span.get("startTime"), span.get("duration")
                valid = all(type(x) in (int, float) and math.isfinite(x) for x in [start, duration])
                start_ms = start / 1000 + OFFSET_MS if valid else None
                end_ms = (start + duration) / 1000 + OFFSET_MS if valid else None
                processes = trace.get("processes")
                process_id = span.get("processID")
                if not isinstance(processes, dict) or process_id not in processes:
                    raise ValueError(
                        f"Missing SN span process: {member} trace={trace_index} span={span_index}"
                    )
                process = processes[process_id]
                service = process.get("serviceName") if isinstance(process, dict) else None
                if not isinstance(service, str) or not service:
                    raise ValueError(
                        f"Missing SN span service: {member} trace={trace_index} span={span_index}"
                    )
                if service not in services:
                    raise ValueError(
                        f"Unknown SN span service {service!r}: {member} trace={trace_index} span={span_index}"
                    )
                trace_id, span_id = span.get("traceID", trace.get("traceID")), span.get("spanID")
                identity = json.dumps([trace_id, span_id]) if trace_id and span_id else None
                errors = [
                    tag.get("value") for tag in span.get("tags", []) if tag.get("key") == "error"
                ]
                error = errors[0] if len(errors) == 1 and type(errors[0]) is bool else None
                observation = dict(
                    run_id=run_id,
                    service=service,
                    identity=identity,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=duration / 1000 if valid else None,
                    is_error=error,
                )
                # Raw fingerprint catches changed tags/process/time under the same native key.
                observation["source_fingerprint"] = hashlib.sha256(
                    json.dumps(
                        [span, trace.get("processes", {}).get(span.get("processID"))],
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                if identity and valid and duration >= 0 and lower <= end_ms < upper:
                    window = int(end_ms // 15000 * 15000)
                    refs = references.setdefault((service, window), {})
                    ref = refs.setdefault(
                        identity,
                        dict(
                            identity=identity,
                            raw_startTime_us=start,
                            raw_duration_us=duration,
                            aligned_end_ms=end_ms,
                            references=[],
                        ),
                    )
                    ref["references"].append(
                        dict(member=member, trace_index=trace_index, span_index=span_index)
                    )
                yield observation

    table, diagnostics = execution_windows(grid, observations())
    table["statistical_object"] = "Span"
    table["alignment_assumption"] = "fixed_minus_28800_seconds_not_proven_timezone"
    table["observations_json"] = [
        json.dumps(list(references.get((r.canonical_service_id, r.window_start_ms), {}).values()))
        for r in table.itertuples()
    ]
    return table, diagnostics


def log_timestamp(stamp: str) -> int:
    # Parse months explicitly: neither host timezone nor LC_TIME affects this coordinate.
    date, clock = stamp.split(" ")
    year, month, day = date.split("-")
    hour, minute, seconds = clock.split(":")
    second, microsecond = seconds.split(".")
    parsed = dt.datetime(
        int(year),
        MONTHS[month],
        int(day),
        int(hour),
        int(minute),
        int(second),
        int(microsecond.ljust(6, "0")),
        tzinfo=dt.timezone.utc,
    )
    return int(parsed.timestamp() * 1000) + OFFSET_MS


def sn_log_windows(grid: pd.DataFrame, logs: dict, member: str) -> tuple[pd.DataFrame, dict]:
    unknown_services = set(logs) - set(grid.canonical_service_id)
    if unknown_services:
        raise ValueError(f"Unknown SN log service {sorted(unknown_services)!r}: {member}")
    events, diagnostics, collection = [], {}, {}
    lower, upper = int(grid.window_start_ms.min()), int(grid.window_end_ms.max())
    for service, messages in logs.items():
        diag, current, previous = Counter(), None, None
        collection[service] = "COLLECTION_PRESENT"
        for index, raw in enumerate(messages):
            diag["source_records"] += 1
            reference = dict(member=member, service=service, message_index=index, raw_message=raw)
            match = LOG_HEAD.match(raw) if isinstance(raw, str) else None
            if not match:
                if current is not None and isinstance(raw, str) and CONTINUATION.match(raw):
                    current["message"] += "\n" + raw
                    current["references"].append(reference)
                    diag["continuations"] += 1
                else:
                    diag["unparsed_records"] += 1
                    current = None
                continue
            stamp, message = match.groups()
            try:
                millis = log_timestamp(stamp)
            except (ValueError, KeyError, OverflowError):
                diag["invalid_timestamp"] += 1
                current = None
                continue
            if previous is not None and millis < previous:
                diag["backward_timestamps"] += 1
            previous = millis
            diag["parsed_timestamp"] += 1
            diag["outside_grid"] += int(not lower <= millis < upper)
            reference.update(
                raw_timestamp=stamp,
                aligned_timestamp_ms=millis,
                alignment_rule="naive_as_UTC_minus_28800_seconds",
            )
            current = dict(
                run_id=grid.run_id.iloc[0],
                service=service,
                timestamp_ms=millis,
                message=message,
                references=[reference],
                partial=False,
            )
            events.append(current)
        diagnostics[service] = dict(diag)
    table = log_windows(grid, events, collection)
    table["alignment_assumption"] = "naive_as_UTC_minus_28800_seconds_not_proven_timezone"
    for service, diag in diagnostics.items():
        if diag.get("invalid_timestamp") or diag.get("unparsed_records"):
            table.loc[table.canonical_service_id.eq(service), "quality_status"] = (
                "COLLECTION_PARTIAL_UNPARSED_RECORDS"
            )
    return table, diagnostics


def sn_resource_windows(
    grid: pd.DataFrame, service: str, rows, member: str
) -> tuple[pd.DataFrame, dict]:
    table = grid.copy()
    table["raw_samples_json"], table["quality_status"] = "[]", "NO_OBSERVATIONS"
    table["unit"], table["measurement_semantics"] = "unknown", "unknown"
    for field in RESOURCE_FIELDS:
        table[field + "_mean"] = float("nan")
        table[field + "_valid_count"] = float("nan")
    table["resource_sample_count"] = float("nan")
    table["resource_coverage"] = float("nan")
    groups, diag, previous = {}, Counter(), None
    lookup = {(r.canonical_service_id, r.window_start_ms): i for i, r in table.iterrows()}
    for line, row in enumerate(rows, 2):
        try:
            stamp = float(row["timestamp"])
            if not math.isfinite(stamp):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            diag["invalid_timestamp"] += 1
            continue
        if previous is not None and stamp < previous:
            diag["backward_timestamps"] += 1
        if previous is not None and stamp == previous:
            diag["duplicate_timestamps_retained"] += 1
        previous = stamp
        values = {}
        for field in RESOURCE_FIELDS:
            try:
                value = float(row[field])
                values[field] = value if math.isfinite(value) else None
            except (ValueError, TypeError, KeyError):
                values[field] = None
            if values[field] is None:
                diag["invalid_" + field] += 1
        idx = lookup.get((service, int(stamp * 1000 // 15000 * 15000)))
        if idx is None:
            diag["outside_grid"] += 1
            continue
        groups.setdefault(idx, []).append(
            dict(
                timestamp_s=stamp, values=values, raw_values=dict(row), member=member, csv_line=line
            )
        )
    for idx, samples in groups.items():
        table.loc[idx, "raw_samples_json"] = json.dumps(samples)
        for field in RESOURCE_FIELDS:
            values = [s["values"][field] for s in samples if s["values"][field] is not None]
            table.loc[idx, field + "_mean"] = sum(values) / len(values) if values else float("nan")
            table.loc[idx, field + "_valid_count"] = len(values)
        unique_seconds = {int(s["timestamp_s"]) for s in samples}
        table.loc[idx, "resource_sample_count"] = len(samples)
        table.loc[idx, "resource_coverage"] = len(unique_seconds) / 15
        table.loc[idx, "quality_status"] = (
            "OBSERVED_RAW_PARTIAL_NONFINITE_UNKNOWN_SEMANTICS"
            if any(v is None for s in samples for v in s["values"].values())
            else "OBSERVED_RAW_UNKNOWN_SEMANTICS"
        )
    diag["accepted"] = sum(map(len, groups.values()))
    return table, dict(diag)


class HashReader:
    def __init__(self, source):
        self.source, self.digest = source, hashlib.sha256()

    def read(self, size=-1):
        value = self.source.read(size)
        self.digest.update(value)
        return value


def exploratory_runs(archive: Path, inventory: dict, profile: dict):
    """One read of each exact nested container; no extraction or duplicate outer data."""
    validate_profile(profile)
    if file_digest(archive, "md5") != SN_MD5 or inventory["archive_md5"] != SN_MD5:
        raise ValueError("SN archive checksum mismatch")
    if set(inventory["runs"]) != set(profile["runs"]):
        raise ValueError("SN run allowlist mismatch")
    services = profile["services"]
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        for container in sorted({r["container"] for r in profile["runs"].values()}):
            if names.count(container) != 1:
                raise ValueError("SN container missing or duplicate: " + container)
            run_ids = [r for r, spec in profile["runs"].items() if spec["container"] == container]
            tables, evidence, seen = {}, {}, set()
            for rid in run_ids:
                run = inventory["runs"][rid]
                if run["container"] != container or set(run["metrics"]) != set(services):
                    raise ValueError("SN source service/container identity mismatch")
                grid = full_grid(
                    "eadro_sn_v1",
                    rid,
                    services,
                    math.ceil(run["metadata"]["start"] * 1000),
                    math.floor(run["metadata"]["end"] * 1000),
                )
                tables[rid] = {"grid": grid, "resource_windows": []}
                evidence[rid] = dict(
                    sources=[],
                    diagnostics={},
                    role_evidence=profile["runs"][rid],
                    qualification=(
                        "declared_fault_run_test_only"
                        if profile["runs"][rid]["is_fault_run"]
                        else "no_declared_injection_reference_distribution_not_health_gold"
                    ),
                    evaluation_label=None,
                    profile=profile,
                )
            with outer.open(container) as source, tarfile.open(fileobj=source, mode="r|*") as inner:
                for member in inner:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/")
                    if parts[0] == "data":
                        parts = parts[1:]
                    rid = parts[0]
                    if rid not in tables:
                        continue
                    kind = (
                        "resource"
                        if member.name.endswith(".csv")
                        else (
                            "logs"
                            if member.name.endswith("/logs.json")
                            else "spans" if member.name.endswith("/spans.json") else None
                        )
                    )
                    if kind is None:
                        continue
                    key = (rid, kind, Path(member.name).stem if kind == "resource" else "")
                    if key in seen:
                        raise ValueError("Duplicate SN member identity")
                    seen.add(key)
                    reader = HashReader(inner.extractfile(member))
                    grid = tables[rid]["grid"]
                    ref = container + "::" + member.name
                    if kind == "spans":
                        table, diag = span_windows(
                            grid, iter_json_array(codecs.getreader("utf-8")(reader)), ref
                        )
                        tables[rid]["execution_windows"] = table
                    elif kind == "logs":
                        table, diag = sn_log_windows(grid, json.loads(reader.read()), ref)
                        tables[rid]["log_windows"] = table
                    else:
                        service = Path(member.name).stem
                        if service not in services:
                            raise ValueError("Unknown SN resource service")
                        rows = csv.DictReader(io.StringIO(reader.read().decode("utf-8")))
                        if set(rows.fieldnames or []) != {"timestamp", *RESOURCE_FIELDS}:
                            raise ValueError("SN resource fields changed")
                        table, diag = sn_resource_windows(grid, service, rows, ref)
                        tables[rid]["resource_windows"].append(
                            table[table.canonical_service_id.eq(service)]
                        )
                    evidence[rid]["sources"].append(
                        dict(
                            container=container,
                            member=member.name,
                            sha256=reader.digest.hexdigest(),
                            size=member.size,
                        )
                    )
                    evidence[rid]["diagnostics"][member.name] = diag
            for rid, modes in tables.items():
                if len(modes["resource_windows"]) != 12 or not all(
                    k in modes for k in ["execution_windows", "log_windows"]
                ):
                    raise ValueError("SN required source member missing")
                modes["resource_windows"] = pd.concat(modes["resource_windows"], ignore_index=True)
                yield rid, modes.pop("grid"), modes, evidence[rid]
