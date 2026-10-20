"""Pure, complete-grid protocol reference labels; no feature or raw-data access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pandas as pd

from .evidence import KEY_COLUMNS, EvidenceIndex

IDENTITY_COLUMNS = KEY_COLUMNS + ["sample_id", "window_end_ms", "case_id"]
UNKNOWN_REASON_ORDER = (
    "run_admission_failed",
    "negative_exception",
    "instantaneous_operation",
    "target_outside_candidates",
    "partial_boundary",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class LabelTables:
    labels: pd.DataFrame
    splits: pd.DataFrame


def _validate_grid(grid: pd.DataFrame, protocol: dict) -> None:
    if set(IDENTITY_COLUMNS) - set(grid):
        raise ValueError("Missing stable grid columns")
    if grid[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("Missing stable grid identity")
    if grid.duplicated(KEY_COLUMNS).any() or grid.sample_id.duplicated().any():
        raise ValueError("duplicate grid identity")
    if not grid.run_id.eq(grid.case_id).all():
        raise ValueError("Case/run identity mismatch")
    for col in ["window_start_ms", "window_end_ms"]:
        if not pd.api.types.is_integer_dtype(grid[col]):
            raise ValueError("Noninteger grid interval")
    if not grid.window_end_ms.sub(grid.window_start_ms).eq(protocol["window_ms"]).all():
        raise ValueError("Invalid grid interval")


def split_content_version(protocol: dict) -> str:
    """Role mapping, rather than paths or timestamps, defines split identity."""
    content = {k: protocol[k] for k in ["split_version", "normal_splits", "fault_split"]}
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{protocol['split_version']}:{digest}"


def build_splits(grid: pd.DataFrame, protocol: dict) -> pd.DataFrame:
    _validate_grid(grid, protocol)
    if protocol["fault_split"] != "test" or not set(protocol["normal_splits"].values()) <= {
        "fit",
        "calibration",
        "test",
    }:
        raise ValueError("Invalid split roles")
    result = grid[IDENTITY_COLUMNS].copy()
    result["split"] = grid.case_id.map(protocol["normal_splits"]).fillna("test")
    result["split_version"] = split_content_version(protocol)
    return result


def classify_windows(grid: pd.DataFrame, evidence: EvidenceIndex, protocol: dict) -> LabelTables:
    """Assign reference labels without using modality validity or model scores.

    EvidenceIndex has already validated source completeness and identities. Cases
    and their events remain in that index even when no positive window exists.
    label_reasons retains every applicable Unknown reason in fixed priority order.
    """
    splits = build_splits(grid, protocol)
    actual = grid[IDENTITY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    expected = evidence.grid[IDENTITY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Input differs from validated grid")
    if set(grid.case_id) != set(evidence.cases):
        raise ValueError("Grid/evidence Case mismatch")
    labels = grid[IDENTITY_COLUMNS].copy()
    values, sources, reasons, reason_lists, refs = [], [], [], [], []
    for row in grid.itertuples(index=False):
        case = evidence.cases[row.case_id]
        fault_run = case["is_fault"]
        if fault_run == (row.case_id in protocol["normal_splits"]):
            raise ValueError("Normal/Fault split identity mismatch")
        admitted = case["admission"]["admitted"]
        boundary = case.get("boundary") or {}
        limit = boundary.get("first_intervention_requested_ms")
        pre = (
            fault_run
            and limit is not None
            and not boundary.get("reasons")
            and row.window_end_ms <= limit
        )
        negative = admitted and (not fault_run or pre)
        positive = False
        unknown = set()
        ids = set(case["evidence_ids"])
        if not admitted:
            unknown.add("run_admission_failed")
        if fault_run:
            fault = case.get("fault") or {}
            document = fault.get("document") or {}
            target = document.get("metadata", {}).get("labels", {}).get("target_service")
            action = document.get("spec", {}).get("action")
            start, end = fault.get("effective_ms"), fault.get("removed_ms")
            instant = action == "pod-kill" or (start is not None and start == end)
            outside = target not in case["candidate_services"]
            if instant:
                unknown.add("instantaneous_operation")
            if outside:
                unknown.add("target_outside_candidates")
            if (
                start is not None
                and end is not None
                and start < end
                and not instant
                and not outside
            ):
                contained = start <= row.window_start_ms < row.window_end_ms <= end
                positive = row.canonical_service_id == target and contained
                if (
                    row.canonical_service_id == target
                    and row.window_start_ms < end
                    and row.window_end_ms > start
                    and not contained
                ):
                    unknown.add("partial_boundary")
        if positive and negative:
            raise ValueError("Positive/negative candidate conflict")
        if negative:
            for event in case["transport_exceptions"]:
                # Only failures completed before the *earliest* intervention may
                # exclude Fault pre-windows; post-failure cannot travel backward.
                if fault_run and event["end_ms"] > limit:
                    continue
                if row.window_start_ms < event["end_ms"] and row.window_end_ms > event["start_ms"]:
                    negative = False
                    unknown.add("negative_exception")
                    ids.add(event["evidence_id"])
        value = 1 if positive else 0 if negative else pd.NA
        if positive:
            source, reason = "injection_association", "complete_direct_target_window"
        elif negative:
            source = "pre_intervention_reference" if fault_run else "normal_run_reference"
            reason = "complete_pre_intervention_window" if fault_run else "admitted_normal_run"
        else:
            source = "unknown"
            if not unknown:
                unknown.add("insufficient_evidence")
            reason = next(r for r in UNKNOWN_REASON_ORDER if r in unknown)
        values.append(value)
        sources.append(source)
        reasons.append(reason)
        reason_lists.append(
            [r for r in UNKNOWN_REASON_ORDER if r in unknown] if source == "unknown" else []
        )
        refs.append(sorted(ids))
    labels["label"] = pd.array(values, dtype="Int8")
    labels["label_source"] = sources
    labels["label_scope"] = "service_window"
    labels["label_reason"] = reasons
    labels["label_reasons"] = reason_lists
    labels["evidence_ids"] = refs
    return LabelTables(labels, splits)
