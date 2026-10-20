"""Immutable, replayable label packages and stable-key relabel differences.

Output is a new run directory containing labels/. Raw and preparation assets are
read-only; staging is validated before any success manifest becomes visible.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.service.preparation import file_digest

from . import rules
from .evidence import KEY_COLUMNS, load_evidence
from .rules import IDENTITY_COLUMNS, classify_windows, split_content_version

FILES = (
    "protocol.json",
    "evidence.json",
    "exceptions.json",
    "labels.parquet",
    "splits.parquet",
    "coverage.json",
)
LABEL_COLUMNS = [
    "label",
    "label_source",
    "label_scope",
    "label_reason",
    "label_reasons",
    "evidence_ids",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_bytes(_bytes(value) + b"\n")


def _records(frame: pd.DataFrame) -> list[dict]:
    return _jsonable(frame.sort_values(KEY_COLUMNS).reset_index(drop=True).to_dict("records"))


def _semantic(protocol: dict, evidence: dict, labels: pd.DataFrame, splits: pd.DataFrame) -> dict:
    # Only registry path is provenance. Business fields inside metadata/plan/fault
    # remain intact, including target paths, which can change the label evidence.
    sources = {
        sid: {k: v for k, v in source.items() if k != "path"}
        for sid, source in evidence["sources"].items()
    }
    cases = {
        cid: {k: v for k, v in case.items() if k != "split"}
        for cid, case in evidence["cases"].items()
    }
    label_protocol = {
        k: v
        for k, v in protocol.items()
        if k not in {"normal_splits", "split_version", "fault_split"}
    }
    return {
        "protocol": _digest(label_protocol),
        "evidence": _digest({"sources": sources, "cases": cases}),
        "grid": _digest(_records(labels[IDENTITY_COLUMNS])),
        "labels": _digest(_records(labels.drop(columns=["label_version"], errors="ignore"))),
        "splits": _digest(_records(splits)),
        "rules": evidence["rules_sha256"],
    }


def _label_version(hashes: dict) -> str:
    return "label:" + _digest({k: v for k, v in hashes.items() if k != "splits"})


def _validate_tables(labels: pd.DataFrame, splits: pd.DataFrame, sources: dict) -> None:
    for frame, columns in (
        (labels, IDENTITY_COLUMNS + LABEL_COLUMNS),
        (splits, IDENTITY_COLUMNS + ["split", "split_version"]),
    ):
        if set(columns) - set(frame):
            raise ValueError("Missing bundle columns")
        if frame[IDENTITY_COLUMNS].isna().any().any():
            raise ValueError("Missing bundle identity")
        if frame.duplicated(KEY_COLUMNS).any() or frame.sample_id.duplicated().any():
            raise ValueError("Duplicate bundle keys")
        if not frame.run_id.eq(frame.case_id).all():
            raise ValueError("Case/run mismatch")
        for col in ["window_start_ms", "window_end_ms"]:
            if not pd.api.types.is_integer_dtype(frame[col]):
                raise ValueError("Noninteger bundle times")
        if not frame.window_end_ms.gt(frame.window_start_ms).all():
            raise ValueError("Invalid bundle interval")
    if _records(labels[IDENTITY_COLUMNS]) != _records(splits[IDENTITY_COLUMNS]):
        raise ValueError("Labels/splits grid mismatch")
    if (
        not pd.api.types.is_integer_dtype(labels.label.dtype)
        or not labels.label.dropna().isin([0, 1]).all()
    ):
        raise ValueError("Invalid nullable integer labels")
    if not splits.split.isin(["fit", "calibration", "test"]).all():
        raise ValueError("Invalid split role")
    allowed = {
        "unknown",
        "injection_association",
        "normal_run_reference",
        "pre_intervention_reference",
    }
    if (
        not labels.label_source.isin(allowed).all()
        or not labels.label_scope.eq("service_window").all()
    ):
        raise ValueError("Invalid label source/scope")
    if labels[LABEL_COLUMNS[:4]].drop(columns="label").isna().any().any():
        raise ValueError("Missing label metadata")
    if not labels.label.isna().equals(labels.label_source.eq("unknown")):
        raise ValueError("Unknown source/label mismatch")
    if not labels.loc[labels.label.eq(1), "label_source"].eq("injection_association").all():
        raise ValueError("Positive source mismatch")
    if (
        not labels.loc[labels.label.eq(0), "label_source"]
        .isin(["normal_run_reference", "pre_intervention_reference"])
        .all()
    ):
        raise ValueError("Negative source mismatch")
    if splits.groupby("case_id").split.nunique().gt(1).any():
        raise ValueError("Case crosses split roles")
    for row in labels.itertuples(index=False):
        for field in ["label_reasons", "evidence_ids"]:
            value = _jsonable(getattr(row, field))
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError("Invalid bundle list column")
        if not set(row.evidence_ids) <= set(sources):
            raise ValueError("Unregistered evidence reference")


def _exceptions(cases: dict) -> dict:
    result = {}
    for cid, case in sorted(cases.items()):
        events = sorted(
            case["transport_exceptions"],
            key=lambda e: (e["start_ms"], e["end_ms"], e["evidence_id"]),
        )
        merged = []
        for event in events:
            start, end = event["start_ms"], event["end_ms"]
            if merged and start <= merged[-1]["end_ms"]:
                merged[-1]["end_ms"] = max(end, merged[-1]["end_ms"])
                merged[-1]["evidence_ids"] = sorted(
                    set(merged[-1]["evidence_ids"]) | {event["evidence_id"]}
                )
            else:
                merged.append(dict(start_ms=start, end_ms=end, evidence_ids=[event["evidence_id"]]))
        result[cid] = {
            "scope": "all_application_services",
            "events": events,
            "merged_intervals": merged,
        }
    return result


def _coverage(labels: pd.DataFrame, splits: pd.DataFrame, grid: pd.DataFrame, cases: dict) -> dict:
    frame = labels.merge(splits[KEY_COLUMNS + ["split"]], on=KEY_COLUMNS, validate="one_to_one")
    validity = [c for c in grid if c.endswith("_valid")]
    if validity:
        frame = frame.merge(grid[KEY_COLUMNS + validity], on=KEY_COLUMNS, validate="one_to_one")

    def counts(part):
        return dict(
            rows=len(part),
            negative=int(part.label.eq(0).sum()),
            positive=int(part.label.eq(1).sum()),
            unknown=int(part.label.isna().sum()),
            modality_valid={c: int(part[c].eq(True).sum()) for c in validity},
        )

    groups = {}
    for column in ["case_id", "canonical_service_id", "split", "label_reason"]:
        groups[column] = {str(key): counts(part) for key, part in frame.groupby(column, sort=True)}
    events = {
        cid: {
            "is_fault": case["is_fault"],
            "fault": case.get("fault"),
            "boundary": case.get("boundary"),
            "admission": case["admission"],
            "positive_windows": int(labels.loc[labels.case_id.eq(cid), "label"].eq(1).sum()),
            "unknown_reasons": sorted(
                set(labels.loc[labels.case_id.eq(cid) & labels.label.isna(), "label_reason"])
            ),
        }
        for cid, case in sorted(cases.items())
    }
    return {"total": counts(frame), "by": groups, "events": events}


def _protect(output: Path, protected: list[Path]) -> Path:
    output = output.resolve()
    for root in protected:
        root = root.resolve()
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output overlaps protected input asset")
    if output.exists():
        raise FileExistsError(output)
    return output


def _provenance() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["numpy", "pandas", "pyarrow", "scikit-learn"]
            },
        },
    }


def read_bundle(path: Path) -> dict:
    """Validate package bytes, structures and semantic versions without raw I/O.

    Accept a run or labels directory. Returns protocol/evidence/exceptions/coverage,
    manifest and labels/splits DataFrames for the independent checker.
    """
    path = Path(path)
    if not (path / "manifest.json").is_file():
        path = path / "labels"
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Malformed bundle manifest")
    for name in FILES:
        if file_digest(path / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Bundle hash mismatch: {name}")
    result = {
        name[:-5]: json.loads((path / name).read_text()) for name in FILES if name.endswith(".json")
    }
    labels, splits = (pd.read_parquet(path / name) for name in ["labels.parquet", "splits.parquet"])
    _validate_tables(labels, splits, result["evidence"]["sources"])
    hashes = _semantic(result["protocol"], result["evidence"], labels, splits)
    if hashes != manifest["semantic_hashes"] or _label_version(hashes) != manifest["label_version"]:
        raise ValueError("Bundle semantic version mismatch")
    version = split_content_version(result["protocol"])
    if version != manifest["split_version"] or not splits.split_version.eq(version).all():
        raise ValueError("Bundle split version mismatch")
    if not labels.label_version.eq(manifest["label_version"]).all():
        raise ValueError("Bundle label version mismatch")
    result.update(manifest=manifest, labels=labels, splits=splits)
    return result


def build_bundle(tt_root: Path, preparation: Path, protocol_path: Path, output: Path) -> dict:
    """Build an immutable run/labels package from validated prepared TT evidence."""
    tt_root, preparation, protocol_path = map(
        lambda p: Path(p).resolve(), [tt_root, preparation, protocol_path]
    )
    if preparation.name == "tt" and not (preparation / "artifact_manifest.json").exists():
        preparation = preparation.parent
    output = _protect(Path(output), [tt_root, preparation, protocol_path])
    protocol = json.loads(protocol_path.read_text())
    index = load_evidence(tt_root, preparation, protocol)
    tables = classify_windows(index.grid, index, protocol)
    labels = tables.labels.sort_values(KEY_COLUMNS).reset_index(drop=True)
    splits = tables.splits.sort_values(KEY_COLUMNS).reset_index(drop=True)
    evidence = {
        "sources": index.sources,
        "cases": index.cases,
        "preparation_manifest_sha256": index.preparation_manifest_sha256,
        "rules_sha256": file_digest(Path(rules.__file__)),
    }
    _validate_tables(labels, splits, index.sources)
    hashes = _semantic(protocol, evidence, labels, splits)
    label_version = _label_version(hashes)
    labels["label_version"] = label_version
    manifest = {
        "schema_version": 1,
        "label_version": label_version,
        "split_version": split_content_version(protocol),
        "semantic_hashes": hashes,
        "inputs": {
            "tt_root": str(tt_root),
            "preparation": str(preparation),
            "protocol_path": str(protocol_path),
            "protocol_sha256": file_digest(protocol_path),
            "preparation_manifest_sha256": index.preparation_manifest_sha256,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_provenance(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".label-staging-", dir=output.parent))
    reserved = False
    try:
        package = staging / "labels"
        package.mkdir()
        for name, value in [
            ("protocol.json", protocol),
            ("evidence.json", evidence),
            ("exceptions.json", _exceptions(index.cases)),
            ("coverage.json", _coverage(labels, splits, index.grid, index.cases)),
        ]:
            _write(package / name, value)
        labels.to_parquet(package / "labels.parquet", index=False)
        splits.to_parquet(package / "splits.parquet", index=False)
        manifest["files"] = {
            name: {"sha256": file_digest(package / name), "bytes": (package / name).stat().st_size}
            for name in FILES
        }
        _write(package / "manifest.json", manifest)
        read_bundle(package)
        output.mkdir()  # atomic reservation; refuse even a concurrent empty run
        reserved = True
        package.rename(output / "labels")
    except BaseException:
        if reserved:
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(staging)
    return manifest


def diff_bundles(old: Path, new: Path, output: Path) -> dict:
    """Outer-key comparison, including added/deleted rows and nullable labels."""
    output = _protect(Path(output), [Path(old), Path(new)])
    before, after = read_bundle(old), read_bundle(new)
    _protect(
        output,
        [
            Path(data["manifest"]["inputs"][name])
            for data in [before, after]
            for name in ["tt_root", "preparation"]
        ],
    )
    columns = IDENTITY_COLUMNS + LABEL_COLUMNS + ["label_version"]
    frames = []
    for data in [before, after]:
        frame = data["labels"][columns].merge(
            data["splits"][KEY_COLUMNS + ["split", "split_version"]],
            on=KEY_COLUMNS,
            validate="one_to_one",
        )
        frames.append({tuple(row[k] for k in KEY_COLUMNS): row for row in _records(frame)})
    rows = []
    fields = [c for c in IDENTITY_COLUMNS + LABEL_COLUMNS + ["split"] if c not in KEY_COLUMNS]
    for key in sorted(set(frames[0]) | set(frames[1])):
        a, b = frames[0].get(key), frames[1].get(key)
        if a is not None and b is not None and all(a.get(c) == b.get(c) for c in fields):
            continue
        row = dict(zip(KEY_COLUMNS, key))
        row["change_type"] = "added" if a is None else "deleted" if b is None else "changed"
        for prefix, value in [("old", a), ("new", b)]:
            row.update(
                {
                    f"{prefix}_{field}": None if value is None else value.get(field)
                    for field in fields
                }
            )
        rows.append(row)
    changes = pd.DataFrame(
        rows,
        columns=KEY_COLUMNS
        + ["change_type"]
        + [f"{p}_{f}" for p in ["old", "new"] for f in fields],
    )
    for prefix in ["old", "new"]:
        changes[f"{prefix}_label"] = pd.array(changes[f"{prefix}_label"], dtype="Int8")
    summary = {
        "changed_rows": len(changes),
        "change_types": changes.change_type.value_counts().to_dict(),
        "old_label_version": before["manifest"]["label_version"],
        "new_label_version": after["manifest"]["label_version"],
        "old_split_version": before["manifest"]["split_version"],
        "new_split_version": after["manifest"]["split_version"],
    }
    transitions = {}
    for row in rows:
        key = f"{row.get('old_label_reason')} -> {row.get('new_label_reason')}"
        transitions[key] = transitions.get(key, 0) + 1
    summary["reason_transitions"] = transitions
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".diff-staging-", dir=output.parent))
    reserved = False
    try:
        changes.to_parquet(staging / "changes.parquet", index=False)
        _write(staging / "summary.json", summary)
        output.mkdir()
        reserved = True
        for name in ["changes.parquet", "summary.json"]:
            (staging / name).rename(output / name)
    except BaseException:
        if reserved:
            shutil.rmtree(output)
        raise
    finally:
        shutil.rmtree(staging)
    return summary
