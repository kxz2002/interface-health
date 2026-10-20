"""Dataset inventory and integrity gates for Service preprocessing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside dataset: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def selected_tt_inventory(root: Path) -> dict[str, Any]:
    migration = json.loads((root / "_handoff/migration_summary.json").read_text())
    if migration.get("status") != "passed":
        raise ValueError("TT migration gate has not passed")
    selection_path = root / "batch_control/final_acceptance/selected_cases.json"
    selection = json.loads(selection_path.read_text())
    cases = selection["cases"]
    if len(cases) != 17 or len({case["case_id"] for case in cases}) != 17:
        raise ValueError("Expected selected17 allowlist")
    mapping = json.loads((root / "_handoff/path_map.json").read_text())
    ids = {case["case_id"] for case in cases}
    if ids != set(mapping["selected_case_ids"]) or ids != set(migration["selected_ids"]):
        raise ValueError("Selected17 migration identity mismatch")
    if migration["verification"]["status"] != "passed":
        raise ValueError("Migration verification has not passed")
    for key in ["run_id", "plan_hash"]:
        if selection[key] != mapping[key] or selection[key] != migration["verification"][key]:
            raise ValueError("Migration run/plan mismatch")
    normals = sorted(
        (case for case in cases if not case["is_anomaly_case"]),
        key=lambda case: case["case_id"],
    )
    if len(normals) != 6:
        raise ValueError("Expected six independent Normal cases")
    roles = {
        case["case_id"]: "fit" if index < 3 else "calibration" if index == 3 else "normal_test"
        for index, case in enumerate(normals)
    }
    runs = []
    for case in cases:
        binding = bind_tt_case(root, case, selection, mapping)
        sources = {}
        for key in ("source_metadata", "service_features"):
            path = checked_path(root, case[key])
            digest = file_digest(path)
            if digest != case[f"{key}_sha256"]:
                raise ValueError(f"Hash mismatch: {path}")
            sources[key] = {"path": case[key], "sha256": digest}
        metadata = json.loads(checked_path(root, case["source_metadata"]).read_text())
        if metadata["case_id"] != case["case_id"]:
            raise ValueError("Metadata case identity mismatch")
        runs.append(
            {
                "run_id": case["case_id"],
                "split": roles.get(case["case_id"], "fault_test"),
                "is_fault_run": case["is_anomaly_case"],
                "start_ms": metadata["window_start_ms"],
                "end_ms": metadata["window_end_ms"],
                "sources": sources,
                **binding,
                "label": None,
                "label_source": "pending_window_evidence",
            }
        )
    return {
        "dataset_id": selection["run_id"],
        "selection_sha256": file_digest(selection_path),
        "runs": runs,
    }


def bind_tt_case(root: Path, case: dict, selection: dict, mapping: dict) -> dict:
    """Bind resource values to raw supplement, quality lineage and historical code."""
    from src.service.modality_preparation import mapped_path

    metadata_path = checked_path(root, case["source_metadata"])
    metadata = json.loads(metadata_path.read_text())
    manifest_path = checked_path(
        root, f"batch_control/research_service_inputs/{case['case_id']}/source_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    quality_path = checked_path(root, case["service_quality"])
    quality = json.loads(quality_path.read_text())
    expected = {
        "case_id": case["case_id"],
        "run_id": selection["run_id"],
        "plan_hash": selection["plan_hash"],
        "plan_item_id": case["plan_item_id"],
        "attempt_id": metadata["attempt_id"],
    }
    for item in [metadata, manifest, quality]:
        if any(item.get(k) != v for k, v in expected.items()):
            raise ValueError("TT source lineage mismatch")
    if manifest["status"] != "complete":
        raise ValueError("Resource source manifest incomplete")
    csv_path = mapped_path(root, manifest["csv_path"], mapping)
    csv_hash = file_digest(csv_path)
    if csv_hash != manifest["csv_sha256"] or csv_hash != quality["source_metric_sha256"]:
        raise ValueError("Supplement hash mismatch")
    if quality["features_sha256"] != case["service_features_sha256"]:
        raise ValueError("Quality feature binding mismatch")
    snapshots = []
    for original, expected_hash in quality["implementation_sha256"].items():
        saved = mapping["exact_versioned_implementation_snapshots"][original]
        if (
            saved["sha256"] != expected_hash
            or file_digest(Path(saved["destination"])) != expected_hash
        ):
            raise ValueError("Historical implementation hash mismatch")
        snapshots.append({"path": saved["destination"], "sha256": expected_hash})
    return {
        "collection_identity": {k: v for k, v in expected.items() if k != "plan_hash"},
        "candidate_services": quality["candidate_services"],
        "expected_service_windows": quality["expected_service_windows"],
        "resource_binding": {
            "source_manifest_sha256": file_digest(manifest_path),
            "quality_sha256": file_digest(quality_path),
            "csv_path": str(csv_path.relative_to(root.resolve())),
            "csv_sha256": csv_hash,
            "historical_implementations": snapshots,
        },
    }


def cached_sn_inventory(
    path: Path, expected_sha256: str, archive: Path, *, defer_archive_digest: bool = False
) -> dict:
    """Reuse the prior exhaustive inventory only with an explicit inventory/archive binding."""
    from src.service.sn_preparation import SN_MD5

    if file_digest(path) != expected_sha256:
        raise ValueError("SN inventory hash mismatch")
    inventory = json.loads(path.read_text())["sn"]
    if inventory["archive_md5"] != SN_MD5 or (
        not defer_archive_digest and file_digest(archive, "md5") != SN_MD5
    ):
        raise ValueError("SN archive checksum mismatch")
    return inventory
