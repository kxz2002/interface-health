"""Read-only evidence adapter. Admission is a protocol reference, never health gold.

Task consumers must use this index rather than opening raw sources themselves.
Every evidence ID binds file content and a field/line locator. No labels are made here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.service.modality_preparation import copy_manifest, stable_ids, validate_grid
from src.service.preparation import checked_path, file_digest, selected_tt_inventory

KEY_COLUMNS = ["dataset_id", "run_id", "canonical_service_id", "window_start_ms"]


@dataclass(frozen=True)
class EvidenceIndex:
    """Validated grid, case evidence and content-addressed source registry.

    cases maps case_id to metadata, plan, admission, boundary, fault, transport
    exceptions and evidence_ids. sources maps evidence_id to path/hash/locator.
    grid keeps original keys, sample IDs and modality validity, without relabeling.
    """

    grid: pd.DataFrame
    cases: dict[str, dict[str, Any]]
    sources: dict[str, dict[str, Any]]
    protocol: dict[str, Any]
    preparation_manifest_sha256: str


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_selected_case(path: Path, expected: dict, expected_hash: str) -> dict:
    if file_digest(path) != expected_hash:
        raise ValueError(f"Source hash mismatch: {path}")
    value = json.loads(path.read_text())
    _identity(value, expected)
    return value


def _identity(value: dict, expected: dict) -> None:
    for key, identity in expected.items():
        if value.get(key) != identity:
            raise ValueError(f"Evidence identity mismatch: {key}")


def validate_execution_interval(
    start: Any, end: Any, case_start: int, case_end: int
) -> tuple[int, int] | None:
    """Missing boundaries remain missing; zero duration remains instantaneous."""
    for value in (start, end):
        if value is not None and (not _integer(value) or not case_start <= value <= case_end):
            raise ValueError("Invalid execution time or case range")
    if start is None or end is None:
        return None
    if end < start:
        raise ValueError("Reversed execution interval")
    return start, end


def run_admission(quality: dict, load: dict, *, is_fault: bool) -> dict:
    reasons, warnings = [], []
    for key, value in (
        ("collection_status", "complete"),
        ("coverage_status", "passed"),
        ("processing_status", "passed"),
    ):
        if quality.get(key) != value:
            reasons.append(key)
    if quality.get("errors") != []:
        reasons.append("collection_errors")
    if load.get("errors") != [] or load.get("status") != "complete":
        reasons.append("load_collection_errors")
    if load.get("attempt_health", {}).get("healthy") is not True:
        reasons.append("attempt_unhealthy")
    if not is_fault:
        if quality.get("cleanup_status") != "passed":
            reasons.append("normal_cleanup_unreviewed")
        if quality.get("recovery", {}).get("terminal_status") != "passed":
            reasons.append("normal_recovery_unreviewed")
    if load.get("failures"):
        warnings.append("load_failures_unlocalized")
    if not _integer(load.get("failures")):
        warnings.append("load_failures_unknown")
    return {"admitted": not reasons, "reasons": sorted(reasons), "warnings": warnings}


def intervention_boundary(fault: dict, signal: dict, events: list[dict]) -> dict:
    """Conservative request boundary; never use late pause acknowledgments."""
    reasons = []
    request = fault.get("apply_requested_ms")
    if not _integer(request):
        reasons.append("missing_intervention_request")
    for event in events:
        if event.get("event_type") == "measurement_started":
            payload = event.get("payload", {})
            for key in (
                "start_ms",
                "start_mono_ns",
                "probe_pause_at_mono_ns",
                "probe_write_policy",
            ):
                if key in payload and payload[key] != signal.get(key):
                    reasons.append("pause_clock_conflict")
        if event.get("event_type") == "single_delete_requested" and not _integer(
            event.get("utc_ms")
        ):
            reasons.append("missing_intervention_request")
    # load_evidence binds these signal anchors to the metadata Case range.
    # Validate every request, even when another earlier request would win min().
    requests = [request] + [
        event.get("utc_ms")
        for event in events
        if event.get("event_type") == "single_delete_requested"
    ]
    for timestamp in requests:
        if not _integer(timestamp):
            continue  # Missing/noninteger requests already make qualification Unknown.
        case_start, case_end = signal.get("start_ms"), signal.get("end_ms")
        if (_integer(case_start) and timestamp < case_start) or (
            _integer(case_end) and timestamp > case_end
        ):
            raise ValueError("Intervention request outside case range")
        if _integer(fault.get("effective_ms")) and timestamp > fault["effective_ms"]:
            raise ValueError("Intervention request after execution start")
    pauses = [
        e.get("payload", {}).get("pause_at_mono_ns")
        for e in events
        if e.get("event_type") == "probe_write_pause_verified"
    ]
    policies = {signal.get("probe_write_policy")}
    policies.update(
        e.get("payload", {}).get("probe_write_policy")
        for e in events
        if e.get("event_type") == "measurement_started"
    )
    policies.discard(None)
    pause = signal.get("probe_pause_at_mono_ns")
    required = bool(pauses) or pause is not None or "baseline-recover-v1" in policies
    pause_ms = None
    if required:
        values = [signal.get("start_ms"), signal.get("start_mono_ns"), pause]
        if not all(_integer(v) for v in values):
            reasons.append("missing_pause_anchor")
        elif pause < signal["start_mono_ns"] or any(p != pause for p in pauses):
            reasons.append("pause_clock_conflict")
        else:
            pause_ms = signal["start_ms"] + (pause - signal["start_mono_ns"]) // 1_000_000
    elif not policies or not policies <= {"disabled", "none", "always"}:
        reasons.append("unknown_pause_policy")
    # Single-delete requests can be earlier than the generic apply request.
    boundaries = [request, pause_ms] + [
        e.get("utc_ms") for e in events if e.get("event_type") == "single_delete_requested"
    ]
    boundary = None if reasons else min(v for v in boundaries if _integer(v))
    return {
        "first_intervention_requested_ms": boundary,
        "pause_boundary_ms": pause_ms,
        "pause_required": required,
        "reasons": sorted(reasons),
    }


def load_transport_exceptions(path: Path, case_id: str, source_id: str) -> tuple[list, list]:
    """Stream requests; retain only issued load exceptions with valid end intervals."""
    exceptions, warnings = [], []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            event = json.loads(line)
            if event.get("case_id") != case_id:
                raise ValueError("Request case identity mismatch")
            if event.get("source") != "load" or event.get("event") != "end":
                continue
            if event.get("outcome") not in {"response", "exception", "suppressed"}:
                warnings.append(f"unknown_request_outcome:line:{line_number}")
            if event.get("outcome") != "exception" or event.get("transport_started") is not True:
                continue
            start, end = event.get("start_ms"), event.get("end_ms")
            if not (_integer(start) and _integer(end) and start < end):
                warnings.append(f"invalid_exception_interval:line:{line_number}")
                continue
            locator = f"line:{line_number}"
            exceptions.append(
                {
                    "start_ms": start,
                    "end_ms": end,
                    "evidence_id": f"{source_id}#{locator}",
                    "source_id": source_id,
                    "source_locator": locator,
                    "attempt_id": event.get("attempt_id"),
                    "exception_type": event.get("exception_type"),
                }
            )
    return exceptions, warnings


def load_evidence(tt_root: Path, preparation: Path, protocol: dict) -> EvidenceIndex:
    """Bind selected17 and an existing preparation package without rebuilding it.

    preparation accepts either the root package or its tt subdirectory. Hashes
    remain those recorded by the producer; historical paths are never rewritten.
    """
    tt_root, preparation = Path(tt_root).resolve(), Path(preparation).resolve()
    if preparation.name == "tt" and not (preparation / "artifact_manifest.json").exists():
        preparation = preparation.parent
    sources: dict[str, dict] = {}
    copied = copy_manifest(tt_root)

    def source(
        path: Path, *, expected_hash: str | None = None, locator: str = "$", migrated: bool = False
    ) -> str:
        digest = file_digest(path)
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"Source hash mismatch: {path}")
        if migrated:
            relative = str(path.relative_to(tt_root))
            record = copied.get(relative)
            if (
                record is None
                or record.get("sha256") != digest
                or record.get("destination_sha256") != digest
            ):
                raise ValueError(f"Migration hash mismatch: {relative}")
        identity = hashlib.sha256(f"{digest}#{locator}".encode()).hexdigest()
        sources[identity] = {
            "path": str(path),
            "sha256": digest,
            "source_locator": locator,
            "size_bytes": path.stat().st_size,
        }
        return identity

    def read(relative: str, *, expected_hash: str | None = None) -> tuple[dict, str]:
        path = checked_path(tt_root, relative)
        sid = source(path, expected_hash=expected_hash, migrated=True)
        return json.loads(path.read_text()), sid

    # Reuse checked raw supplement binding; its historical split is deliberately ignored.
    inventory = selected_tt_inventory(tt_root)
    selection_path = tt_root / "batch_control/final_acceptance/selected_cases.json"
    selection = json.loads(selection_path.read_text())
    for relative in (
        "_handoff/migration_summary.json",
        "_handoff/path_map.json",
        "_handoff/dataset_copy_manifest.jsonl",
    ):
        source(tt_root / relative)
    source(selection_path, migrated=True)
    manifest_path = preparation / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source(manifest_path)
    prepared = {}
    for relative in ("inventory.json", "tt/resource_windows.parquet"):
        entry = manifest["artifacts"][relative]
        path = checked_path(preparation, relative)
        source(path, expected_hash=entry["sha256"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("Preparation size mismatch")
        prepared[relative] = path
    original_inventory = json.loads(prepared["inventory.json"].read_text())["tt"]
    if original_inventory != inventory:
        raise ValueError("Preparation inventory binding mismatch")
    grid = pd.read_parquet(prepared["tt/resource_windows.parquet"])
    validate_grid(grid)
    if not stable_ids(grid)["sample_id"].equals(grid["sample_id"]):
        raise ValueError("Unstable sample identity")
    required = KEY_COLUMNS + ["case_id", "sample_id", "window_end_ms", "resource_valid"]
    if set(required) - set(grid):
        raise ValueError("Missing grid columns")
    if (
        grid[required].isna().any().any()
        or grid.duplicated(KEY_COLUMNS).any()
        or grid.sample_id.duplicated().any()
    ):
        raise ValueError("Invalid or duplicate stable grid keys")
    for column in ("window_start_ms", "window_end_ms"):
        if not pd.api.types.is_integer_dtype(grid[column]):
            raise ValueError("Noninteger grid time")
    if not grid.window_end_ms.eq(grid.window_start_ms + protocol["window_ms"]).all():
        raise ValueError("Invalid grid intervals")
    if (
        not grid.run_id.eq(grid.case_id).all()
        or not grid.dataset_id.eq(inventory["dataset_id"]).all()
    ):
        raise ValueError("Grid dataset/case identity mismatch")
    if set(grid.run_id) != {c["case_id"] for c in selection["cases"]}:
        raise ValueError("Grid selected17 identity mismatch")
    normal_ids = {c["case_id"] for c in selection["cases"] if not c["is_anomaly_case"]}
    if protocol.get("window_ms") != 15000 or protocol.get("fault_split") != "test":
        raise ValueError("Unsupported window or Fault split protocol")
    if (
        sorted(protocol["normal_splits"].values())
        != ["calibration", "fit", "fit", "fit", "fit", "test"]
        or set(protocol["normal_splits"]) != normal_ids
    ):
        raise ValueError("Protocol Normal split identity mismatch")
    cases = {}
    for selected, run in zip(selection["cases"], inventory["runs"]):
        cid, base = selected["case_id"], selected["case_root"]
        if base != cid:
            raise ValueError("Selected case root identity mismatch")
        expected = {**run["collection_identity"], "plan_hash": selection["plan_hash"]}
        meta_path = checked_path(tt_root, selected["source_metadata"])
        meta = validate_selected_case(meta_path, expected, selected["source_metadata_sha256"])
        evidence_ids = [
            source(meta_path, expected_hash=selected["source_metadata_sha256"], migrated=True)
        ]
        quality, sid = read(
            selected["collection_quality"], expected_hash=selected["collection_quality_sha256"]
        )
        evidence_ids.append(sid)
        plan, sid = read(f"{base}/collection_plan.json")
        evidence_ids.append(sid)
        if meta.get("planned") != plan or plan.get("plan_item_id") != expected["plan_item_id"]:
            raise ValueError("Plan identity mismatch")
        fault_run = selected["is_anomaly_case"]
        if (
            meta.get("is_anomaly_case") is not fault_run
            or (plan.get("role") == "normal") == fault_run
        ):
            raise ValueError("Normal/Fault identity mismatch")
        if not fault_run and (
            plan.get("fault") is not None or meta.get("inject_start_ms") is not None
        ):
            raise ValueError("Normal has intervention evidence")
        subset = grid[grid.run_id.eq(cid)]
        start, end = meta["window_start_ms"], meta["window_end_ms"]
        if not (_integer(start) and _integer(end) and start < end):
            raise ValueError("Invalid case range")
        if (
            len(subset) != run["expected_service_windows"]
            or set(subset.canonical_service_id) != set(run["candidate_services"])
            or not subset.window_start_ms.ge(start).all()
            or not subset.window_end_ms.le(end).all()
        ):
            raise ValueError("Grid case range/coverage mismatch")
        load, sid = read(f"{base}/control/load_status.json")
        evidence_ids.append(sid)
        _identity(load, run["collection_identity"])
        events_path = checked_path(tt_root, f"{base}/collection_events.jsonl")
        evidence_ids.append(source(events_path, migrated=True))
        with events_path.open() as handle:
            events = [json.loads(line) for line in handle]
        for event in events:
            _identity(event, run["collection_identity"])
            payload = event.get("payload", {})
            present_identity = {k: v for k, v in run["collection_identity"].items() if k in payload}
            _identity(payload, present_identity)
        for collector in quality.get("manifests", []):
            _identity(collector, run["collection_identity"])
        signal, sid = read(f"{base}/control/start_signal.json")
        evidence_ids.append(sid)
        _identity(signal, run["collection_identity"])
        if signal.get("start_ms") != start or signal.get("end_ms") != end:
            raise ValueError("Signal case range mismatch")
        fault, boundary = None, None
        if fault_run:
            fault, sid = read(f"{base}/control/fault_control.json")
            evidence_ids.append(sid)
            _identity(fault, run["collection_identity"])
            actual_start, actual_end = fault.get("effective_ms"), fault.get("removed_ms")
            for field, value in (("inject_start_ms", actual_start), ("inject_end_ms", actual_end)):
                if meta.get(field) != value:
                    raise ValueError("Fault/metadata interval mismatch")
            validate_execution_interval(actual_start, actual_end, start, end)
            boundary = intervention_boundary(fault, signal, events)

        requests = checked_path(tt_root, f"{base}/load/api_attempts.jsonl")
        request_sid = source(requests, migrated=True)
        evidence_ids.append(request_sid)
        exceptions, warnings = load_transport_exceptions(requests, cid, request_sid)
        for exception in exceptions:
            sources[exception["evidence_id"]] = {
                **sources[request_sid],
                "source_locator": exception["source_locator"],
            }
        admission = run_admission(quality, load, is_fault=fault_run)
        admission["warnings"].extend(warnings)
        cases[cid] = {
            "case_id": cid,
            "is_fault": fault_run,
            "metadata": meta,
            "plan": plan,
            "fault": fault,
            "boundary": boundary,
            "admission": admission,
            "transport_exceptions": exceptions,
            "evidence_ids": sorted(evidence_ids),
            "split": protocol["normal_splits"].get(cid, "test"),
            "candidate_services": run["candidate_services"],
        }
    return EvidenceIndex(grid, cases, sources, protocol, file_digest(manifest_path))
