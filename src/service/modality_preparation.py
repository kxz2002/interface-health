"""Service modality preparation; no labels, fitting, or normalization."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, TextIO

import pandas as pd

from src.service.preparation import checked_path, file_digest

KEYS = ["dataset_id", "run_id", "canonical_service_id", "window_start_ms"]
GRID_COLUMNS = KEYS + ["window_end_ms", "sample_id"]
STEP_MS = 15000


def validate_grid(grid: pd.DataFrame) -> None:
    if grid.empty or grid.duplicated(KEYS).any():
        raise ValueError("Empty or duplicate grid")
    if not (grid.window_end_ms - grid.window_start_ms).eq(STEP_MS).all():
        raise ValueError("Grid must contain complete 15 second windows")
    for _, run in grid.groupby(["dataset_id", "run_id"]):
        starts = sorted(run.window_start_ms.unique())
        if any(b - a != STEP_MS for a, b in zip(starts, starts[1:])):
            raise ValueError("Noncontiguous grid")
        if len(run) != len(starts) * run.canonical_service_id.nunique():
            raise ValueError("Incomplete service grid")


def stable_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sample_id"] = [
        hashlib.sha256("\0".join(map(str, row)).encode()).hexdigest()
        for row in frame[KEYS].itertuples(index=False, name=None)
    ]
    validate_grid(frame)
    return frame


def full_grid(
    dataset_id: str, run_id: str, services: list[str], lower_ms: int, upper_ms: int
) -> pd.DataFrame:
    lower = (lower_ms + STEP_MS - 1) // STEP_MS * STEP_MS
    upper = upper_ms // STEP_MS * STEP_MS
    rows = [
        (dataset_id, run_id, service, start, start + STEP_MS)
        for service in sorted(set(services))
        for start in range(lower, upper, STEP_MS)
    ]
    return stable_ids(pd.DataFrame(rows, columns=KEYS + ["window_end_ms"]))


def resource_windows(grid: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    validate_grid(grid)
    if set(source.case_id) != set(grid.run_id):
        raise ValueError("Resource case identity mismatch")
    source = source.copy()
    if "sample_id" in source:
        source = source.rename(columns={"sample_id": "source_sample_id"})
    source["run_id"] = source.case_id
    source["dataset_id"] = grid.dataset_id.iloc[0]
    if source.duplicated(KEYS).any() or set(map(tuple, source[KEYS].values)) != set(
        map(tuple, grid[KEYS].values)
    ):
        raise ValueError("Resource does not match full grid")
    result = grid.merge(
        source.drop(columns=["window_end_ms", "sample_id"], errors="ignore"),
        on=KEYS,
        how="left",
        validate="one_to_one",
    )
    if "window_end_ms" in source and not source.set_index(KEYS).window_end_ms.sort_index().equals(
        grid.set_index(KEYS).window_end_ms.sort_index()
    ):
        raise ValueError("Resource window boundary mismatch")
    result["quality_status"] = result.get(
        "resource_quality", pd.Series("SOURCE_QUALITY_RETAINED", index=result.index)
    )
    if "resource_valid" in result:
        result["quality_status"] = [
            (
                "RESOURCE_VALID"
                if x is True
                else "RESOURCE_INVALID" if x is False else "RESOURCE_QUALITY_UNKNOWN"
            )
            for x in result.resource_valid.tolist()
        ]
    return result


def blocked_tables(grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables = {}
    reasons = {
        "resource_windows": "BLOCKED_RESOURCE_UNITS_AND_COUNTER_GAUGE",
        "execution_windows": "BLOCKED_METADATA_SPAN_CLOCK_ALIGNMENT_AND_ENTRY_RULE",
        "log_windows": "BLOCKED_LOG_TIMEZONE",
    }
    for name, reason in reasons.items():
        table = grid.copy()
        table["quality_status"] = reason
        if name == "resource_windows":
            pass  # Exact source field names are supplied by the bound inventory.
        elif name == "execution_windows":
            for field in [
                "observation_count",
                "duration_mean_ms",
                "duration_p95_ms",
                "duration_max_ms",
                "error_known_count",
                "error_count",
                "error_coverage",
            ]:
                table[field] = None
            table["statistical_object"] = "Span_UNRESOLVED"
            table["window_assignment"] = "end_time_half_open_BLOCKED"
        else:
            table["messages_json"] = "[]"
            table["event_count"] = None
        tables[name] = table
    return tables


def iter_top_array(handle: TextIO, key: str):
    """Stream one object array and validate the entire envelope (archive parser adapted)."""
    decoder = json.JSONDecoder()
    buffer, pos, eof = "", 0, False

    def more():
        nonlocal buffer, pos, eof
        chunk = handle.read(65536)
        buffer, pos = buffer[pos:] + chunk, 0
        eof = not chunk

    def peek():
        nonlocal pos
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos < len(buffer):
                return buffer[pos]
            if eof:
                raise ValueError("Truncated JSON envelope")
            more()

    def consume(char):
        nonlocal pos
        if peek() != char:
            raise ValueError("Expected " + char)
        pos += 1

    def value():
        nonlocal pos
        peek()
        while True:
            try:
                result, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("Invalid JSON envelope") from None
                more()
                continue
            if end == len(buffer) and not eof:
                more()
                continue
            pos = end
            return result

    consume("{")
    found = False
    if peek() != "}":
        while True:
            name = value()
            consume(":")
            if name == key:
                if found:
                    raise ValueError("Duplicate target array")
                found = True
                consume("[")
                if peek() != "]":
                    while True:
                        yield value()
                        if peek() == "]":
                            break
                        consume(",")
                consume("]")
            else:
                value()
            if peek() == "}":
                break
            consume(",")
    consume("}")
    if not found or (buffer[pos:] + handle.read()).strip():
        raise ValueError("Missing target array or trailing JSON data")


def entry_observations(traces: Iterable[dict], run_id: str):
    """Yield each Entry including left-boundary-crossing observations; no start crop."""
    for trace in traces:
        for span in trace.get("spans", []):
            if span.get("type") != "Entry":
                continue
            identity = span.get("node_id")
            if not identity:
                fields = [span.get(k) for k in ("trace_id", "segment_id", "span_id")]
                identity = json.dumps(fields) if all(x is not None for x in fields) else None
            yield dict(
                run_id=run_id,
                service=span.get("service_code"),
                identity=identity,
                start_ms=span.get("start_timestamp_ms"),
                end_ms=span.get("end_timestamp_ms"),
                duration_ms=span.get("duration_ms"),
                is_error=span.get("is_error"),
            )


def execution_windows(
    grid: pd.DataFrame, observations: Iterable[dict]
) -> tuple[pd.DataFrame, dict]:
    validate_grid(grid)
    if grid.run_id.nunique() != 1:
        raise ValueError("Execution adapter accepts one run at a time")
    run = grid.run_id.iloc[0]
    lookup = {
        (r.canonical_service_id, r.window_start_ms): i
        for i, r in grid.reset_index(drop=True).iterrows()
    }
    lower = int(grid.window_start_ms.min())
    groups, seen, diag = {}, {}, Counter()
    for observation in observations:
        if observation["run_id"] != run:
            raise ValueError("Execution run mismatch")
        identity = observation["identity"]
        if identity is None:
            diag["missing_identity"] += 1
            continue
        fingerprint = json.dumps(observation, sort_keys=True)
        if identity in seen:
            if seen[identity] != fingerprint:
                raise ValueError("Execution identity has conflicting observations")
            diag["duplicates"] += 1
            continue
        seen[identity] = fingerprint
        end, start, duration = (observation[k] for k in ["end_ms", "start_ms", "duration_ms"])
        if (
            any(type(v) not in (float, int) or not math.isfinite(v) for v in [start, end, duration])
            or duration < 0
            or end < start
        ):
            diag["invalid_time_or_duration"] += 1
            continue
        window = lower + math.floor((end - lower) / STEP_MS) * STEP_MS
        idx = lookup.get((observation["service"], window))
        if idx is None:
            diag["outside_grid"] += 1
            continue
        groups.setdefault(idx, []).append(observation)
    result = grid.reset_index(drop=True).copy()
    for field in [
        "observation_count",
        "duration_mean_ms",
        "duration_p95_ms",
        "duration_max_ms",
        "error_known_count",
        "error_count",
        "error_coverage",
    ]:
        result[field] = float("nan")
    result["quality_status"] = "NO_OBSERVATIONS"
    result["statistical_object"] = "ServiceEntry"
    result["window_assignment"] = "end_time_half_open_offline"
    for idx, values in groups.items():
        durations = pd.Series([v["duration_ms"] for v in values], dtype=float)
        known = [v["is_error"] for v in values if type(v["is_error"]) is bool]
        result.loc[
            idx,
            [
                "observation_count",
                "duration_mean_ms",
                "duration_p95_ms",
                "duration_max_ms",
                "error_known_count",
                "error_count",
                "error_coverage",
            ],
        ] = [
            len(values),
            durations.mean(),
            durations.quantile(0.95),
            durations.max(),
            len(known),
            sum(known) if known else float("nan"),
            len(known) / len(values),
        ]
        result.loc[idx, "quality_status"] = (
            "OBSERVED" if len(known) == len(values) else "OBSERVED_ERROR_PARTIAL"
        )
    diag["accepted"] = sum(map(len, groups.values()))
    return result, dict(diag)


def mapped_path(root: Path, original: str, mapping: dict) -> Path:
    exact = mapping.get("exact_files", {})
    if original in exact:
        path = Path(exact[original]).resolve()
    else:
        prefix = mapping.get("relative_batch_prefix", "")
        if not prefix or not original.startswith(prefix):
            raise ValueError("No explicit path mapping: " + original)
        path = (root / original[len(prefix) :]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Mapped path outside dataset")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_manifest(root: Path) -> dict[str, dict]:
    records = {}
    with (root / "_handoff/dataset_copy_manifest.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            relative = record.get("relative_path")
            if relative in records:
                raise ValueError("Duplicate copy manifest path")
            records[relative] = record
    return records


def bound_file(root: Path, relative: str, records: dict) -> dict:
    if relative not in records:
        raise ValueError("File absent from copy manifest: " + relative)
    path = checked_path(root, relative)
    record = records[relative]
    actual = file_digest(path)
    if actual != record["sha256"] or actual != record["destination_sha256"]:
        raise ValueError("Hash mismatch: " + relative)
    return {"path": relative, "sha256": actual}


def tt_modalities(
    root: Path, run: dict, grid: pd.DataFrame, records: dict, mapping: dict
) -> tuple[dict, dict]:
    """Read exactly one allowlisted merged trace and registered JSONL per stream."""
    run_id = run["run_id"]
    provenance, diagnostics = [], {}
    trace_paths = [
        p
        for p in records
        if p
        and p.startswith(run_id + "/trace_data/")
        and "/_slices/" not in p
        and p.count("/") == 2
        and "_skywalking_traces_" in p
        and p.endswith(".json")
    ]
    if len(trace_paths) != 1:
        raise ValueError("Expected exactly one manifest-bound merged trace for " + run_id)
    provenance.append(bound_file(root, trace_paths[0], records))
    with checked_path(root, trace_paths[0]).open() as handle:
        execution, diagnostics["execution"] = execution_windows(
            grid, entry_observations(iter_top_array(handle, "traces"), run_id)
        )
    registry_relative = run_id + "/log_data/stream_registry.json"
    provenance.append(bound_file(root, registry_relative, records))
    registry = json.loads(checked_path(root, registry_relative).read_text())
    expected = {key: registry[key] for key in ["case_id", "run_id", "attempt_id", "plan_item_id"]}
    if expected["case_id"] != run_id:
        raise ValueError("Registry run mismatch")
    if run.get("collection_identity") and expected != run["collection_identity"]:
        raise ValueError("Registry collection identity mismatch")
    services = set(grid.canonical_service_id)
    collection, selected, used = {}, [], set()
    for stream in registry["streams"]:
        if any(stream.get(k) != v for k, v in expected.items()):
            raise ValueError("Registry stream identity mismatch")
        service = stream["container_name"]
        if service not in services:
            continue
        path = mapped_path(root, stream["log_file"], mapping).with_suffix(".jsonl")
        relative = str(path.relative_to(root.resolve()))
        if relative in used:
            raise ValueError("Repeated registered JSONL input")
        used.add(relative)
        if not path.is_file():
            collection[service] = "COLLECTION_MISSING"
            diagnostics[relative] = {"missing_jsonl": True}
            continue
        provenance.append(bound_file(root, relative, records))
        if collection.get(service) != "COLLECTION_MISSING":
            collection[service] = "COLLECTION_PRESENT"
        selected.append((stream, path, relative))

    def events():
        for stream, path, relative in selected:
            instance = "/".join(str(stream.get(k, "")) for k in ["pod_uid", "container_id"])
            with path.open() as handle:
                parsed, diag = log_events(
                    (json.loads(line) for line in handle),
                    stream["container_name"],
                    instance,
                    relative,
                    run_id,
                    expected,
                )
            diagnostics[relative] = diag
            if diag.get("invalid_source_time_or_raw"):
                collection[stream["container_name"]] = "COLLECTION_PARTIAL_INVALID_SOURCE_TIME"
            yield from parsed

    logs = log_windows(grid, events(), collection)
    coverage = collection_window_status(grid, [s for s, _, _ in selected])
    logs["collection_status"] = coverage
    for idx, status in enumerate(coverage):
        if status != "REGISTERED_STREAM_WINDOW_COVERED" and pd.isna(logs.loc[idx, "event_count"]):
            logs.loc[idx, "quality_status"] = status
    for service, status in collection.items():
        if status != "COLLECTION_PRESENT":
            logs.loc[logs.canonical_service_id.eq(service), "quality_status"] = status
    return {"execution_windows": execution, "log_windows": logs}, {
        "sources": provenance,
        "diagnostics": diagnostics,
        "collection_status": collection,
    }


def collection_window_status(grid: pd.DataFrame, streams: list[dict]) -> list[str]:
    """Registry overlap is collection evidence, never evidence of healthy operation."""
    result = []
    for row in grid.itertuples():
        intervals = sorted(
            (s["opened_ms"], s["closed_ms"])
            for s in streams
            if s["container_name"] == row.canonical_service_id
            and type(s.get("opened_ms")) is int
            and type(s.get("closed_ms")) is int
            and s["closed_ms"] > row.window_start_ms
            and s["opened_ms"] < row.window_end_ms
        )
        covered_until = row.window_start_ms
        for lo, hi in intervals:
            if lo <= covered_until:
                covered_until = max(covered_until, hi)
        result.append(
            "REGISTERED_STREAM_WINDOW_COVERED"
            if covered_until >= row.window_end_ms
            else "COLLECTION_PARTIAL_WINDOW" if intervals else "COLLECTION_MISSING"
        )
    return result


CONTINUATION = re.compile(r"^(?:\s|Caused by:|Suppressed:|\.\.\. \d+ more)")


def log_events(
    records: Iterable[dict],
    service: str,
    instance: str,
    source: str,
    run_id: str,
    expected_identity: dict | None = None,
) -> tuple[list[dict], dict]:
    events, diag, current, previous_segment = [], Counter(), None, None
    for line_number, record in enumerate(records, 1):
        if record.get("case_id") != run_id:
            raise ValueError("Log case identity mismatch")
        if expected_identity and any(record.get(k) != v for k, v in expected_identity.items()):
            raise ValueError("Log collection identity mismatch")
        if record.get("replay") is True:
            diag["replays"] += 1
            continue
        if record.get("segment") != previous_segment:
            current = None
        previous_segment = record.get("segment")
        stamp = record.get("source_timestamp")
        raw = record.get("raw_line")
        ref = dict(
            path=source,
            line=line_number,
            instance=instance,
            source_occurrence=record.get("source_occurrence"),
            segment=record.get("segment"),
            partial=record.get("partial"),
            replay=record.get("replay"),
        )
        try:
            ts = pd.Timestamp(stamp)
            if ts.tzinfo is None or not isinstance(raw, str) or not raw.startswith(stamp + " "):
                raise ValueError("Unproven source timestamp")
            millis = ts.value // 1000000
        except (ValueError, TypeError, AttributeError):
            diag["invalid_source_time_or_raw"] += 1
            current = None
            continue
        message = raw[len(stamp) + 1 :]
        if current is not None and (current["_partial"] or CONTINUATION.match(message)):
            current["message"] += ("" if current["_partial"] else "\n") + message
            current["references"].append(ref)
            current["_partial"] = record.get("partial") is True
        else:
            current = dict(
                run_id=run_id,
                service=service,
                timestamp_ms=millis,
                message=message,
                references=[ref],
                _partial=record.get("partial") is True,
            )
            events.append(current)
    for event in events:
        event["partial"] = event.pop("_partial")
    diag["events"] = len(events)
    return events, dict(diag)


def log_windows(
    grid: pd.DataFrame, events: Iterable[dict], collection: dict[str, str]
) -> pd.DataFrame:
    validate_grid(grid)
    result = grid.reset_index(drop=True).copy()
    result["messages_json"], result["event_count"] = "[]", float("nan")
    result["quality_status"] = [
        collection.get(s, "COLLECTION_MISSING") for s in result.canonical_service_id
    ]
    lookup = {
        (r.run_id, r.canonical_service_id, r.window_start_ms): i for i, r in result.iterrows()
    }
    lower = int(grid.window_start_ms.min())
    groups = {}
    for event in events:
        start = lower + (event["timestamp_ms"] - lower) // STEP_MS * STEP_MS
        idx = lookup.get((event["run_id"], event["service"], start))
        if idx is None:
            continue
        bucket = groups.setdefault(idx, {})
        item = bucket.setdefault(
            event["message"], dict(message=event["message"], count=0, references=[], partial=False)
        )
        item["count"] += 1
        item["references"].extend(event["references"])
        item["partial"] |= event["partial"]
    for idx, messages in groups.items():
        result.loc[idx, "messages_json"] = json.dumps(list(messages.values()), ensure_ascii=False)
        result.loc[idx, "event_count"] = sum(x["count"] for x in messages.values())
        result.loc[idx, "quality_status"] = "OBSERVED"
    result.loc[result.quality_status.eq("COLLECTION_PRESENT"), "quality_status"] = (
        "NO_OBSERVATIONS_COLLECTION_PRESENT"
    )
    return result
