"""Read Eadro SocialNetwork archives without extracting or executing contents."""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import json
import math
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import TextIO

from src.service.preparation import file_digest

SN_MD5 = "4c523faf64770b1e9a709cdd610e07c4"


def iter_json_array(handle: TextIO):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    expect_value = True
    ended = False
    first = True
    while True:
        chunk = handle.read(65536)
        buffer += chunk
        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer:
                    break
                if buffer[0] != "[":
                    raise ValueError("Expected JSON array")
                started = True
                buffer = buffer[1:]
                continue
            if ended:
                if buffer.strip():
                    raise ValueError("Trailing JSON data")
                break
            if not buffer:
                break
            if buffer[0] == "]":
                if expect_value and not first:
                    raise ValueError("Trailing comma")
                ended = True
                buffer = buffer[1:]
                continue
            if not expect_value:
                if buffer[0] != ",":
                    raise ValueError("Expected comma")
                buffer = buffer[1:]
                expect_value = True
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            yield value
            first = False
            expect_value = False
            buffer = buffer[end:]
        if not chunk:
            if not ended:
                raise ValueError("Truncated JSON array")
            return
        if not started:
            first = True


def inspect_sn(archive: Path) -> dict:
    if file_digest(archive, "md5") != SN_MD5:
        raise ValueError("SocialNetwork archive checksum mismatch")
    runs = {}
    normal_ids = {}
    normal_logs = {}
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        for name in names:
            if (
                name.startswith("__MACOSX")
                or not name.endswith(".json")
                or "/no fault/" not in name
            ):
                continue
            metadata = json.loads(outer.read(name))
            run_id = Path(name).name.replace("SN.fault-", "SN.").removesuffix(".json")
            runs.setdefault(run_id, {})["metadata"] = metadata
        containers = sorted(
            name
            for name in names
            if not name.startswith("__MACOSX")
            and (name.endswith("SN-all.tgz") or ("/no fault/" in name and name.endswith(".tar.xz")))
        )
        for container in containers:
            normal = "/no fault/" in container
            with (
                outer.open(container) as compressed,
                tarfile.open(fileobj=compressed, mode="r|*") as inner,
            ):
                for member in inner:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/")
                    if parts[0] == "data":
                        parts = parts[1:]
                    if parts[0].startswith("SN.fault-"):
                        metadata = json.load(inner.extractfile(member))
                        run_id = parts[0].replace("SN.fault-", "SN.").removesuffix(".json")
                        runs.setdefault(run_id, {})["metadata"] = metadata
                        continue
                    run_id = parts[0]
                    run = runs.setdefault(run_id, {})
                    run.update({"is_normal_archive": normal, "container": container})
                    source = inner.extractfile(member)
                    if member.name.endswith(".csv"):
                        reader = csv.DictReader(codecs.getreader("utf-8")(source))
                        times = []
                        invalid = Counter()
                        for row in reader:
                            times.append(float(row["timestamp"]))
                            for field, value in row.items():
                                try:
                                    valid = math.isfinite(float(value))
                                except (ValueError, TypeError):
                                    valid = False
                                if not valid:
                                    invalid[field] += 1
                        deltas = Counter(right - left for left, right in zip(times, times[1:]))
                        run.setdefault("metrics", {})[Path(member.name).stem] = {
                            "rows": len(times),
                            "start_s": min(times),
                            "end_s": max(times),
                            "fields": reader.fieldnames,
                            "interval_counts": dict(deltas),
                            "nonfinite": dict(invalid),
                        }
                    elif member.name.endswith("logs.json"):
                        logs = json.load(source)
                        run["logs"] = {
                            service: {
                                "messages": len(messages),
                                "first": messages[0][:180] if messages else None,
                                "last": messages[-1][:180] if messages else None,
                            }
                            for service, messages in logs.items()
                        }
                        if normal:
                            normal_logs[run_id] = {
                                hashlib.sha256((service + "\0" + message).encode()).hexdigest()
                                for service, messages in logs.items()
                                for message in messages
                            }
                    elif member.name.endswith("spans.json"):
                        identities = set()
                        services = Counter()
                        errors = Counter()
                        starts = []
                        spans = 0
                        trace_bounds = {}
                        for trace in iter_json_array(codecs.getreader("utf-8")(source)):
                            identities.add(trace["traceID"])
                            trace_starts = [span["startTime"] for span in trace.get("spans", [])]
                            if trace_starts:
                                trace_bounds[trace["traceID"]] = [
                                    min(trace_starts),
                                    max(trace_starts),
                                ]
                            processes = trace.get("processes", {})
                            for span in trace.get("spans", []):
                                spans += 1
                                services[
                                    processes.get(span.get("processID"), {}).get(
                                        "serviceName", "UNKNOWN"
                                    )
                                ] += 1
                                starts.append(span["startTime"])
                                for tag in span.get("tags", []):
                                    if tag.get("key") in ("error", "span.kind", "http.status_code"):
                                        errors[str((tag["key"], tag.get("value")))] += 1
                        run["traces"] = {
                            "trace_ids": len(identities),
                            "spans": spans,
                            "start_us": min(starts) if starts else None,
                            "end_start_us": max(starts) if starts else None,
                            "services": dict(services),
                            "tags": dict(errors),
                        }
                        if normal:
                            normal_ids[run_id] = identities
                            run["normal_trace_bounds"] = trace_bounds
    overlap = []
    normal_names = sorted(normal_ids)
    for index, left in enumerate(normal_names):
        for right in normal_names[index + 1 :]:
            shared = normal_ids[left] & normal_ids[right]
            in_both = []
            for identity in shared:
                inside = []
                for run_id in (left, right):
                    bounds = runs[run_id]["normal_trace_bounds"].get(identity)
                    meta = runs[run_id]["metadata"]
                    inside.append(
                        bounds is not None and meta["start"] <= bounds[0] / 1e6 < meta["end"]
                    )
                if all(inside):
                    in_both.append(identity)
            overlap.append(
                {
                    "left": left,
                    "right": right,
                    "trace_id_intersection": len(shared),
                    "trace_id_in_both_run_bounds": len(in_both),
                    "timestamped_log_intersection": len(normal_logs[left] & normal_logs[right]),
                }
            )
    for run in runs.values():
        run.pop("normal_trace_bounds", None)
    return {
        "archive_md5": SN_MD5,
        "runs": runs,
        "normal_overlap": overlap,
        "notes": [
            "Log samples contain no proven timezone; no timestamp correction applied.",
            "Fault metadata describes exposure, not complete Service state.",
            "Metric units and counter/gauge semantics require source provenance.",
        ],
    }
