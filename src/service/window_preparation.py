"""Write separate modality tables on a complete Service/run/window grid."""

import json
import math
from pathlib import Path

import pandas as pd

from src.service.modality_preparation import (
    GRID_COLUMNS,
    KEYS,
    blocked_tables,
    copy_manifest,
    full_grid,
    resource_windows,
    stable_ids,
    tt_modalities,
    validate_grid,
)
from src.service.preparation import checked_path, file_digest


def table_schema(table: pd.DataFrame, name: str, dataset: str, profile: str = "strict") -> dict:
    units = {
        "cpu_cores": "CPU cores",
        "memory_usage_ratio": "dimensionless",
        "process_count": "processes",
        "memory_usage_bytes": "bytes",
        "memory_working_set_bytes": "bytes",
        "memory_limit_bytes": "bytes",
        "error_coverage": "fraction",
        "observation_count": "observed ServiceEntry count",
        "error_known_count": "Entry count with explicit boolean error",
        "error_count": "error Entry count among known errors",
        "event_count": "log events",
    }
    fields = []
    for column in table:
        unit = units.get(column)
        if column.endswith("_ms"):
            unit = "milliseconds" if column.startswith("duration_") else "Unix epoch milliseconds"
        if dataset == "sn":
            if name == "resource_windows" and column.endswith("_mean"):
                unit = "unknown"
            elif name == "execution_windows" and column in units:
                unit = unit.replace("ServiceEntry", "Span").replace("Entry", "Span")
            elif column not in GRID_COLUMNS and name == "resource_windows" and profile == "strict":
                unit = "UNRESOLVED"
        fields.append({"name": column, "dtype": str(table[column].dtype), "unit": unit})
    features = {
        "resource_windows": (
            ["cpu_cores", "memory_usage_ratio", "process_count"]
            if dataset == "tt"
            else [c for c in table if c.endswith("_mean")] if profile == "exploratory" else []
        ),
        "execution_windows": [
            "observation_count",
            "duration_mean_ms",
            "duration_p95_ms",
            "duration_max_ms",
            "error_known_count",
            "error_count",
            "error_coverage",
        ],
        "log_windows": ["messages_json"],
    }[name]
    return {
        "schema_version": "service-preparation-v1",
        "profile": profile if dataset == "sn" else "tt_source_verified",
        "measurement_semantics": (
            "unknown" if dataset == "sn" and name == "resource_windows" else None
        ),
        "key": KEYS,
        "fields": fields,
        "feature_fields": features,
        "source_manifest": "../modality_sources.json",
        "missing_semantics": "No observation is not zero or healthy; blocked data cannot be fitted.",
        "window_assignment": (
            "end_time_half_open_offline" if name == "execution_windows" else "source_time_half_open"
        ),
        "messages_format": (
            "JSON list of {message,count,references,partial}" if name == "log_windows" else None
        ),
        "log_reference_scope": (
            "All non-replay source rows participating in each event; replay rows only contribute diagnostic counts."
            if name == "log_windows"
            else None
        ),
        "prediction_boundary": "Use only listed feature values; log input uses message/count only. References, raw times, profile, role and source metadata are provenance, never model input.",
        "resource_coverage_definition": (
            "unique occupied one-second timestamp bins / 15; raw duplicates retained; per-field finite value counts retained"
            if dataset == "sn" and name == "resource_windows" and profile == "exploratory"
            else None
        ),
    }


def validate_tt_bound_grid(grid: pd.DataFrame, run: dict) -> None:
    """Reject tidy truncation against source-bound count and metadata full windows."""
    validate_grid(grid)
    if "expected_service_windows" not in run:
        return  # Historical low-level fixtures; production inventory always binds this.
    expected = full_grid(
        str(grid.dataset_id.iloc[0]),
        run["run_id"],
        run["candidate_services"],
        run["start_ms"],
        run["end_ms"],
    )
    if len(grid) != run["expected_service_windows"] or len(expected) != len(grid):
        raise ValueError("TT bound grid count differs from quality/metadata")
    columns = KEYS + ["window_end_ms"]
    if (
        not grid[columns]
        .sort_values(KEYS)
        .reset_index(drop=True)
        .equals(expected[columns].sort_values(KEYS).reset_index(drop=True))
    ):
        raise ValueError("TT bound grid boundaries differ from metadata")


def write_window_tables(
    inventory: dict,
    tt_root: Path,
    output: Path,
    *,
    sn_profile: str = "strict",
    sn_profile_config: dict | None = None,
    sn_archive: Path | None = None,
) -> dict:
    """Task 3 consumes windows.parquet; this writer never creates labels or splits."""
    if sn_profile not in {"strict", "exploratory"}:
        raise ValueError("Unknown SN profile")
    if sn_profile == "exploratory" and (sn_profile_config is None or sn_archive is None):
        raise ValueError("SN exploratory requires explicit profile config and archive")
    tables = {"tt": {}, "sn": {}}
    grids, provenance = [], {"tt": {}, "sn": {"status": "BLOCKED", "offset_applied": False}}
    records = copy_manifest(tt_root) if inventory["tt"]["runs"] else {}
    mapping = json.loads((tt_root / "_handoff/path_map.json").read_text()) if records else {}
    for run in inventory["tt"]["runs"]:
        relative = run["sources"]["service_features"]["path"]
        source = checked_path(tt_root, relative)
        if file_digest(source) != run["sources"]["service_features"]["sha256"]:
            raise ValueError("Resource hash changed after inventory")
        resource = pd.read_parquet(source)
        grid = resource[["canonical_service_id", "window_start_ms", "window_end_ms"]].copy()
        grid["dataset_id"], grid["run_id"] = inventory["tt"]["dataset_id"], run["run_id"]
        grid = stable_ids(grid)[GRID_COLUMNS]
        if run.get("candidate_services") and set(grid.canonical_service_id) != set(
            run["candidate_services"]
        ):
            raise ValueError("Resource service set differs from quality binding")
        validate_tt_bound_grid(grid, run)
        modes, evidence = tt_modalities(tt_root.resolve(), run, grid, records, mapping)
        modes["resource_windows"] = resource_windows(grid, resource)
        evidence["sources"].append({"path": relative, "sha256": file_digest(source)})
        provenance["tt"][run["run_id"]] = evidence
        grids.append(grid)
        for name, frame in modes.items():
            tables["tt"].setdefault(name, []).append(frame)
    strict_runs = inventory["sn"]["runs"] if sn_profile == "strict" else {}
    for run_id, run in strict_runs.items():
        metadata = run["metadata"]
        services = sorted(
            set(run.get("metrics", {}))
            | set(run.get("logs", {}))
            | set(run.get("traces", {}).get("services", {}))
        )
        grid = full_grid(
            "eadro_sn_v1",
            run_id,
            services,
            math.ceil(metadata["start"] * 1000),
            math.floor(metadata["end"] * 1000),
        )
        modes = blocked_tables(grid)
        fields = sorted(
            {
                f
                for metric in run.get("metrics", {}).values()
                for f in metric.get("fields", [])
                if f != "timestamp"
            }
        )
        if fields:
            modes["resource_windows"] = grid.assign(
                **{f: None for f in fields},
                quality_status="BLOCKED_RESOURCE_UNITS_AND_COUNTER_GAUGE",
            )
        grids.append(grid)
        for name, frame in modes.items():
            tables["sn"].setdefault(name, []).append(frame)
    if sn_profile == "exploratory":
        from src.service.sn_exploratory import exploratory_runs

        provenance["sn"] = {
            "status": "EXPLORATORY_ASSUMPTION_BOUND",
            "offset_applied": True,
            "resolved_profile": sn_profile_config,
            "runs": {},
        }
        for run_id, grid, modes, evidence in exploratory_runs(
            sn_archive, inventory["sn"], sn_profile_config
        ):
            grids.append(grid)
            provenance["sn"]["runs"][run_id] = evidence
            for name, frame in modes.items():
                tables["sn"].setdefault(name, []).append(frame)
    for dataset, modes in tables.items():
        if not modes:
            continue
        directory = output / dataset
        directory.mkdir(exist_ok=False)
        for name, frames in modes.items():
            table = pd.concat(frames, ignore_index=True)
            validate_grid(table)
            table.to_parquet(directory / f"{name}.parquet", index=False)
            (directory / f"{name}.schema.json").write_text(
                json.dumps(
                    table_schema(table, name, dataset, sn_profile), ensure_ascii=False, indent=2
                )
                + "\n"
            )
    windows = pd.concat(grids, ignore_index=True)
    validate_grid(windows)
    windows.to_parquet(output / "windows.parquet", index=False)
    (output / "modality_sources.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    )
    return {
        "window_rows": len(windows),
        "datasets": windows.groupby("dataset_id").size().to_dict(),
        "training_ready": False,
        "labels_and_splits": "PENDING_TASK_3",
        "sn_status": (
            "BLOCKED_UNRESOLVED_SOURCE_SEMANTICS"
            if sn_profile == "strict"
            else "EXPLORATORY_ASSUMPTION_BOUND_PENDING_TASK_3_4"
        ),
    }
