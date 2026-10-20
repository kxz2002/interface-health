"""Independent evaluation labels, authorized reference pools and exposure indices.

Pool eligibility is a run-level permission, never a health label or a claim that
all model inputs are valid. Consumers must additionally select explicit modality
masks from eligibility_masks and their model configuration. No fitting occurs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.service.modality_preparation import GRID_COLUMNS, KEYS, stable_ids, validate_grid
from src.service.preparation import checked_path, file_digest

ROLES = {"fit": "fit", "calibration": "calibration", "normal_test": "holdout", "fault_test": "test"}
SN_NORMAL_ROLES = {
    "SN.2022-04-20T182405D2022-04-20T184806": "fit",
    "SN.2022-04-21T105249D2022-04-21T111651": "calibration",
    "SN.2022-04-21T153302D2022-04-21T155703": "normal_test",
}


def build_label_splits(
    windows: pd.DataFrame,
    inventory: dict,
    *,
    sn_profile: str = "strict",
    sources: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve every grid row; Unknown labels never gate authorized reference pools."""
    validate_grid(windows)
    if sn_profile not in {"strict", "exploratory"}:
        raise ValueError("Unknown SN profile")
    if not stable_ids(windows)["sample_id"].equals(windows["sample_id"]):
        raise ValueError("Unstable sample identity")
    tt = inventory["tt"]
    runs = tt["runs"]
    normals = sorted(r["run_id"] for r in runs if r["is_fault_run"] is False)
    if runs and len(normals) != 6:
        raise ValueError("TT role protocol requires six Normal runs")
    expected = {
        rid: "fit" if i < 3 else "calibration" if i == 3 else "normal_test"
        for i, rid in enumerate(normals)
    }
    records = {}
    for run in runs:
        rid = run["run_id"]
        role = expected.get(rid, "fault_test")
        if run["split"] != role:
            raise ValueError("TT role differs from fixed Normal-only protocol")
        key = (tt["dataset_id"], rid)
        if key in records:
            raise ValueError("Duplicate run identity")
        records[key] = (
            role,
            not run["is_fault_run"],
            (
                "tt_authorized_normal_only_run_protocol_not_health_gold"
                if not run["is_fault_run"]
                else "declared_fault_run_test_only"
            ),
        )
    sn_sources = (sources or {}).get("sn", {})
    if sn_profile == "exploratory":
        from src.service.sn_exploratory import validate_profile

        profile = sn_sources.get("resolved_profile", {})
        validate_profile(profile)
        if sn_sources.get("status") != "EXPLORATORY_ASSUMPTION_BOUND":
            raise ValueError("SN qualification requires parsed exploratory source evidence")
        if set(profile["runs"]) != set(inventory["sn"]["runs"]):
            raise ValueError("SN qualification run allowlist mismatch")
    for rid, run in inventory["sn"]["runs"].items():
        if sn_profile == "strict":
            records[("eadro_sn_v1", rid)] = (
                (
                    SN_NORMAL_ROLES.get(rid, "normal_test")
                    if run.get("is_normal_archive")
                    else "fault_test"
                ),
                False,
                "strict_profile_unresolved_source_semantics",
            )
            continue
        spec = profile["runs"][rid]
        evidence = sn_sources.get("runs", {}).get(rid, {})
        normal = rid in SN_NORMAL_ROLES
        role = SN_NORMAL_ROLES.get(rid, "fault_test")
        if (
            spec.get("proposed_role") != role
            or spec.get("is_fault_run") is not (not normal)
            or run.get("is_normal_archive") is not normal
            or run.get("container") != spec.get("container")
            or evidence.get("role_evidence") != spec
            or not evidence.get("sources")
            or "faults" not in run["metadata"]
            or bool(run["metadata"]["faults"]) is normal
        ):
            raise ValueError("SN qualification contradicts metadata, source identity or fixed role")
        records[("eadro_sn_v1", rid)] = (
            role,
            normal,
            (
                "no_declared_injection_reference_distribution_not_health_gold"
                if normal
                else "declared_fault_run_test_only"
            ),
        )
    actual = set(map(tuple, windows[["dataset_id", "run_id"]].drop_duplicates().values))
    if actual != set(records):
        raise ValueError("Grid run identities do not match inventory")
    labels = windows[GRID_COLUMNS].copy()
    labels["label"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    labels["label_reason"] = (
        "unknown_no_independent_service_window_state_evidence; exposure_and_reference_role_are_not_state_labels"
    )
    splits = windows[GRID_COLUMNS].copy()
    values = [
        records[(d, r)]
        for d, r in windows[["dataset_id", "run_id"]].itertuples(index=False, name=None)
    ]
    splits["run_role"] = [ROLES[r] for r, _, _ in values]
    splits["reference_eligible"] = [eligible for _, eligible, _ in values]
    splits["reference_reason"] = [reason for _, _, reason in values]
    for field, role in [("training", "fit"), ("calibration", "calibration")]:
        splits[field + "_eligible"] = splits.reference_eligible & splits.run_role.eq(role)
        splits[field + "_reason"] = [
            (
                "authorized_reference_pool_requires_explicit_model_modality_mask"
                if ok
                else "excluded_by_run_role_or_reference_qualification"
            )
            for ok in splits[field + "_eligible"]
        ]
    splits["eligibility_scope"] = (
        "run_reference_pool_only; intersect_model_required_modality_masks_before_use"
    )
    return labels, splits


def modality_masks(grid: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Copy source quality/coverage without deciding a multimodal model's mask policy."""
    result = grid[GRID_COLUMNS].copy()
    for modality in ["resource", "execution", "log"]:
        source = pd.read_parquet(directory / f"{modality}_windows.parquet")
        validate_grid(source)
        identity = source[GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)
        if not identity.equals(grid[GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)):
            raise ValueError("Modality grid identity mismatch: " + modality)
        columns = [
            c
            for c in source
            if c not in GRID_COLUMNS
            and (
                c in {"resource_valid", "quality_status", "resource_coverage", "error_coverage"}
                or c.endswith("_valid_count")
                or c.endswith("_valid")
                or c.endswith("_coverage")
            )
        ]
        rename = {c: c if c.startswith(modality + "_") else modality + "_" + c for c in columns}
        result = result.merge(
            source[KEYS + columns].rename(columns=rename),
            on=KEYS,
            how="left",
            validate="one_to_one",
        )
    return result


def exposure_index(inventory: dict, dataset: str, tt_root: Path | None) -> dict:
    entries = []
    runs = (
        inventory["tt"]["runs"]
        if dataset == "tt"
        else [dict(run_id=rid, **run) for rid, run in inventory["sn"]["runs"].items()]
    )
    for run in runs:
        entry = {"run_id": run["run_id"]}
        if dataset == "tt":
            binding = run.get("sources", {}).get("source_metadata")
            entry["source_metadata"] = binding
            entry["status"] = "source_reference_only"
            if binding and tt_root is not None:
                path = checked_path(tt_root, binding["path"])
                if file_digest(path) != binding["sha256"]:
                    raise ValueError("TT exposure metadata hash mismatch")
                metadata = json.loads(path.read_text())
                if metadata["case_id"] != run["run_id"]:
                    raise ValueError("TT exposure metadata identity mismatch")
                entry["declared_exposure"] = {
                    k: metadata.get(k)
                    for k in [
                        "is_anomaly_case",
                        "anomaly_type",
                        "inject_start_ms",
                        "inject_end_ms",
                        "planned",
                    ]
                }
                entry["status"] = "source_bound_declaration_only"
        else:
            entry["declared_exposure"] = run["metadata"].get("faults")
            entry["container"] = run.get("container")
            entry["status"] = "inventory_metadata_declaration_only"
        entries.append(entry)
    return {
        "prediction_input": False,
        "interpretation": "Exposure indices are not Service state labels; non-target and recovery remain Unknown.",
        "runs": entries,
    }


def write_label_split_tables(
    inventory: dict, output: Path, *, sn_profile: str = "strict", tt_root: Path | None = None
) -> dict:
    """Consume only prepared full grids/modality tables and small bound metadata."""
    windows = pd.read_parquet(output / "windows.parquet")
    sources = json.loads((output / "modality_sources.json").read_text())
    labels, splits = build_label_splits(windows, inventory, sn_profile=sn_profile, sources=sources)
    groups = [("tt", inventory["tt"]["dataset_id"]), ("sn", "eadro_sn_v1")]
    names = [
        "labels.parquet",
        "splits.parquet",
        "eligibility_masks.parquet",
        "exposures.json",
        "labels.schema.json",
        "splits.schema.json",
        "eligibility_masks.schema.json",
    ]
    pending = []
    for dataset, dataset_id in groups:
        subset = windows[windows.dataset_id.eq(dataset_id)]
        if subset.empty:
            continue
        directory = output / dataset
        if any((directory / name).exists() for name in names):
            raise FileExistsError("Refusing to overwrite labels/splits outputs")
        tables = {
            "labels": labels[labels.dataset_id.eq(dataset_id)],
            "splits": splits[splits.dataset_id.eq(dataset_id)],
            "eligibility_masks": modality_masks(subset, directory),
        }
        pending.append((directory, tables, exposure_index(inventory, dataset, tt_root)))
    for directory, tables, exposures in pending:
        for name, table in tables.items():
            table.to_parquet(directory / f"{name}.parquet", index=False)
            schema = {
                "schema_version": "service-label-splits-v1",
                "key": KEYS,
                "fields": [{"name": c, "dtype": str(table[c].dtype)} for c in table],
                "feature_fields": [],
                "prediction_input": False,
                "evaluation_scope": "Independent Service-window evidence absent; labels Unknown, AUROC/AP unavailable.",
                "eligibility_scope": "Run reference permission only; model must intersect required modality masks. Unknown evaluation label does not veto authorized reference use.",
                "mask_semantics": "Source quality/coverage retained, not health labels; no universal multimodal validity inferred.",
                "sn_reference_scope": "Exploratory no-declared-injection reference distribution, not health gold; strict profile blocked.",
            }
            (directory / f"{name}.schema.json").write_text(json.dumps(schema, indent=2) + "\n")
        (directory / "exposures.json").write_text(
            json.dumps(exposures, ensure_ascii=False, indent=2) + "\n"
        )
    return {
        "labels_and_splits": "WRITTEN_PENDING_TASK_4_QUALITY_ACCEPTANCE",
        "labels_known": int(labels.label.notna().sum()),
        "labels_unknown": int(labels.label.isna().sum()),
        "training_ready": False,
        "reference_pool_rows": int(splits.reference_eligible.sum()),
        "training_pool_rows_before_model_masks": int(splits.training_eligible.sum()),
        "calibration_pool_rows_before_model_masks": int(splits.calibration_eligible.sum()),
    }
