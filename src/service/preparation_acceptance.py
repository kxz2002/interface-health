"""Independent, read-only table acceptance; never infer health or model validity."""

import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path

import pandas as pd

from src.service.modality_preparation import (
    GRID_COLUMNS,
    KEYS,
    full_grid,
    stable_ids,
    validate_grid,
)
from src.service.preparation import file_digest


def counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def inspect_dataset(
    directory: Path,
    *,
    expected_grid: pd.DataFrame | None = None,
    expected_splits: dict | None = None,
) -> dict:
    names = [
        "resource_windows",
        "execution_windows",
        "log_windows",
        "labels",
        "splits",
        "eligibility_masks",
    ]
    tables = {name: pd.read_parquet(directory / f"{name}.parquet") for name in names}
    baseline = tables["labels"][GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    if expected_grid is not None and not baseline.equals(
        expected_grid[GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    ):
        raise ValueError("Prepared grid differs from bound inventory full windows")
    for name, frame in tables.items():
        validate_grid(frame)
        grid = frame[GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)
        if not grid.equals(baseline) or not stable_ids(grid).sample_id.equals(grid.sample_id):
            raise ValueError(f"{directory.name}/{name}: grid or stable ID mismatch")
        schema_path = directory / f"{name}.schema.json"
        if not schema_path.is_file():
            raise ValueError(f"Missing schema: {schema_path}")
        schema = json.loads(schema_path.read_text())
        if [f["name"] for f in schema["fields"]] != list(frame):
            raise ValueError(f"{name}: schema field order mismatch")
    labels, splits = tables["labels"], tables["splits"]
    if not labels.label.dropna().isin([0, 1]).all():
        raise ValueError("Nonbinary label")
    if labels.label.notna().any():
        raise ValueError("Current preparation profile requires all labels Unknown")
    if splits.groupby("run_id").run_role.nunique().gt(1).any():
        raise ValueError("Cross-role run")
    if (
        splits.loc[splits.run_role.eq("test"), ["training_eligible", "calibration_eligible"]]
        .any()
        .any()
    ):
        raise ValueError("Fault entered reference fit/calibration")
    if expected_splits is not None:
        if set(splits.run_id) != set(expected_splits):
            raise ValueError("Run set differs from expected split authority")
        for rid, subset in splits.groupby("run_id"):
            for field, expected in expected_splits[rid].items():
                values = subset[field]
                if values.isna().any() or not values.eq(expected).all():
                    raise ValueError(f"{rid}: expected split {field}={expected!r}")
                if field.endswith("_eligible") and not pd.api.types.is_bool_dtype(values):
                    raise ValueError(f"{rid}: expected split boolean {field}")
    result = {
        "rows": len(baseline),
        "runs": {},
        "modalities": {},
        "labels_known": int(labels.label.notna().sum()),
        "labels_unknown": int(labels.label.isna().sum()),
        "label_reasons": counts(labels.label_reason),
        "training_pool_rows_before_model_masks": int(splits.training_eligible.sum()),
        "calibration_pool_rows_before_model_masks": int(splits.calibration_eligible.sum()),
        "reference_pool_rows_before_model_masks": int(splits.reference_eligible.sum()),
        "model_mask_applied": False,
        "evaluation_metrics": {
            "AUROC": None,
            "AP": None,
            "reason": "Independent Service-window labels absent",
        },
    }
    for run_id, subset in splits.groupby("run_id"):
        result["runs"][run_id] = dict(
            rows=len(subset),
            services=int(subset.canonical_service_id.nunique()),
            windows_per_service=int(subset.window_start_ms.nunique()),
            first_window_ms=int(subset.window_start_ms.min()),
            last_window_end_ms=int(subset.window_end_ms.max()),
            role=str(subset.run_role.iloc[0]),
            training_pool_rows_before_model_masks=int(subset.training_eligible.sum()),
            calibration_pool_rows_before_model_masks=int(subset.calibration_eligible.sum()),
            labels_unknown=int(labels.loc[labels.run_id.eq(run_id), "label"].isna().sum()),
            modalities={},
        )
    for name in names[:3]:
        frame = tables[name]
        report = {"quality_counts": counts(frame.quality_status), "runs": {}}
        for run_id, subset in frame.groupby("run_id"):
            data = {"quality_counts": counts(subset.quality_status)}
            for field in [
                "resource_valid",
                "resource_coverage",
                "observation_count",
                "event_count",
            ]:
                if field not in subset:
                    continue
                values = subset[field]
                data[field] = dict(
                    nonnull_rows=int(values.notna().sum()),
                    null_rows=int(values.isna().sum()),
                    sum=float(values.sum()),
                    mean=float(values.mean()) if values.notna().any() else None,
                )
            report["runs"][run_id] = data
            result["runs"][run_id]["modalities"][name] = data
        result["modalities"][name] = report
    masks = tables["eligibility_masks"]
    result["mask_columns"] = list(masks)
    result["quality_mask_counts"] = {}
    for c in masks:
        if c not in GRID_COLUMNS:
            if pd.api.types.is_bool_dtype(masks[c]):
                result["quality_mask_counts"][c] = counts(masks[c])
            elif "quality" in c or "status" in c:
                result["quality_mask_counts"][c] = counts(masks[c])
    for c in splits:
        if c.endswith("reason"):
            result[c + "_counts"] = counts(splits[c])
    result["readiness"] = (
        "exploratory_input_preparation_accepted_semantics_unproven"
        if directory.name == "sn"
        else "input_preparation_accepted_pending_model_mask"
    )
    result["training_executed"] = False
    result["state_gold_available"] = False
    return result


def expected_split_permissions(inventory: dict, *, sn_profile: str, profile: dict | None) -> dict:
    """Derive roles from source identities and frozen configuration, never split outputs."""
    mapping = {
        "fit": "fit",
        "calibration": "calibration",
        "normal_test": "holdout",
        "fault_test": "test",
    }
    roles = {"tt": {}, "sn": {}}
    normals = sorted(r["run_id"] for r in inventory["tt"]["runs"] if r["is_fault_run"] is False)
    if len(normals) != 6:
        raise ValueError("Expected six TT Normal identities")
    normal_roles = {
        rid: "fit" if i < 3 else "calibration" if i == 3 else "normal_test"
        for i, rid in enumerate(normals)
    }
    for run in inventory["tt"]["runs"]:
        role = normal_roles.get(run["run_id"], "fault_test")
        if type(run["is_fault_run"]) is not bool or run["split"] != role:
            raise ValueError("TT inventory contradicts frozen Normal-only roles")
        roles["tt"][run["run_id"]] = (mapping[role], not run["is_fault_run"])
    if sn_profile != "exploratory" or profile is None:
        raise ValueError("This production acceptance requires frozen exploratory SN configuration")
    from src.service.sn_exploratory import validate_profile

    validate_profile(profile)
    if set(profile["runs"]) != set(inventory["sn"]["runs"]):
        raise ValueError("SN role allowlist differs from inventory")
    for rid, run in inventory["sn"]["runs"].items():
        spec = profile["runs"][rid]
        fault = spec["is_fault_run"]
        role = spec["proposed_role"]
        if (
            type(fault) is not bool
            or role not in mapping
            or run["is_normal_archive"] is not (not fault)
            or run["container"] != spec["container"]
            or bool(run["metadata"]["faults"]) is not fault
            or (role == "fault_test") is not fault
        ):
            raise ValueError("SN inventory contradicts frozen role configuration")
        roles["sn"][rid] = (mapping[role], not fault)
    return {
        d: {
            rid: dict(
                run_role=role,
                reference_eligible=ref,
                training_eligible=ref and role == "fit",
                calibration_eligible=ref and role == "calibration",
            )
            for rid, (role, ref) in runs.items()
        }
        for d, runs in roles.items()
    }


def write_acceptance(
    output: Path, *, manual_evidence: Path | None = None, report_directory: Path | None = None
) -> dict:
    """Create reports after production; refuse overwrite and require explicit review evidence."""
    destination = report_directory or output
    if report_directory is not None and destination.exists():
        raise FileExistsError("Revalidation directory already exists")
    targets = [destination / "quality_report.json", destination / "artifact_manifest.json"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Acceptance outputs already exist")
    if manual_evidence is None or not manual_evidence.is_file():
        raise ValueError("Manual raw-record evidence required for acceptance")
    manual = json.loads(manual_evidence.read_text())
    checks = manual.get("checks", [])
    if (
        not checks
        or not all(c.get("passed") is True for c in checks)
        or not all(
            any(c.get("dataset") == d and c.get("role") == role for c in checks)
            for d in ["tt", "sn"]
            for role in ["Normal", "Fault"]
        )
    ):
        raise ValueError("Incomplete or failed manual source evidence")
    inventory = json.loads((output / "inventory.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    resolved = summary["resolved_config"]
    profile = resolved.get("sn_assumption_profile")
    config_path = Path(resolved["sn_profile_config"])
    if (
        file_digest(config_path) != resolved["sn_profile_config_sha256"]
        or json.loads(config_path.read_text()) != profile
    ):
        raise ValueError("Frozen SN profile hash/content changed")
    permissions = expected_split_permissions(
        inventory, sn_profile=resolved["sn_profile"], profile=profile
    )
    prior_path = output / "artifact_manifest.json"
    previous = json.loads(prior_path.read_text()) if prior_path.exists() else None
    production_sources = (
        (previous or {}).get("production_source_sha256")
        or (previous or {}).get("source_sha256")
        or summary.get("implementation_sha256")
    )
    if not production_sources:
        raise ValueError("Production source hashes missing from manifest and summary")
    if previous is not None:
        for relative, binding in previous["artifacts"].items():
            source = output / relative
            if (
                file_digest(source) != binding["sha256"]
                or source.stat().st_size != binding["bytes"]
            ):
                raise ValueError("Production artifact changed: " + relative)
    expected = {"tt": [], "sn": []}
    for run in inventory["tt"]["runs"]:
        expected["tt"].append(
            full_grid(
                inventory["tt"]["dataset_id"],
                run["run_id"],
                run["candidate_services"],
                run["start_ms"],
                run["end_ms"],
            )
        )
    for rid, run in inventory["sn"]["runs"].items():
        expected["sn"].append(
            full_grid(
                "eadro_sn_v1",
                rid,
                sorted(run["metrics"]),
                math.ceil(run["metadata"]["start"] * 1000),
                math.floor(run["metadata"]["end"] * 1000),
            )
        )
    report = {
        "datasets": {
            d: inspect_dataset(
                output / d,
                expected_grid=pd.concat(expected[d], ignore_index=True),
                expected_splits=permissions[d],
            )
            for d in ["tt", "sn"]
        },
        "manual_evidence": str(manual_evidence),
        "manual_evidence_sha256": file_digest(manual_evidence),
        "scope": "Input preparation only; reference pools precede model masks; no health gold, no training.",
    }
    sources = json.loads((output / "modality_sources.json").read_text())
    report["source_diagnostics"] = {
        "tt": {r: v["diagnostics"] for r, v in sources["tt"].items()},
        "sn": {r: v["diagnostics"] for r, v in sources["sn"]["runs"].items()},
    }
    report["expected_split_permissions"] = permissions
    report["schema_gate"] = "all_six_schemas_required_and_field_order_verified"
    report["production_directory"] = str(output)
    if report_directory is not None:
        destination.mkdir(parents=True, exist_ok=False)
    targets[0].write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    versions = {}
    for package in [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
        "torch",
        "pytest",
        "hydra-core",
        "omegaconf",
        "ijson",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "validation_source_sha256": {
            str(p): file_digest(p)
            for p in [
                *Path("src/service").glob("*preparation*.py"),
                Path("src/service/sn_exploratory.py"),
                Path("scripts/prepare_service_data.py"),
                Path("configs/service/sn_exploratory_v1.json"),
            ]
        },
        "production_manifest_sha256": file_digest(prior_path) if previous is not None else None,
        "production_source_sha256": production_sources,
        "production_schema_sha256": {
            str(p.relative_to(output)): file_digest(p)
            for p in sorted(output.glob("*/*.schema.json"))
        },
        "acceptance_artifacts": {
            "quality_report.json": {
                "sha256": file_digest(targets[0]),
                "bytes": targets[0].stat().st_size,
            }
        },
        "artifacts": {
            str(p.relative_to(output)): {"sha256": file_digest(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "input_bindings": "inventory.json, modality_sources.json and tt/sn exposures.json carry exact input hashes, source members, maps and stable IDs",
        "resolved_config": json.loads((output / "summary.json").read_text())["resolved_config"],
    }
    targets[1].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manual-evidence", type=Path, required=True)
    parser.add_argument("--report-directory", type=Path)
    args = parser.parse_args()
    write_acceptance(
        args.output, manual_evidence=args.manual_evidence, report_directory=args.report_directory
    )
