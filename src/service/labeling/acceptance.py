"""Independent evidence replay and explicit, version-bound score eligibility."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.service.preparation import file_digest

from . import bundle as package
from . import rules
from .evidence import KEY_COLUMNS, load_evidence
from .rules import IDENTITY_COLUMNS, classify_windows


def _preparation_root(path: Path) -> Path:
    path = Path(path).resolve()
    return (
        path.parent
        if path.name == "tt" and not (path / "artifact_manifest.json").exists()
        else path
    )


def check_bundle(bundle: Path, preparation: Path) -> dict:
    """Revalidate original sources and replay rules against the bound full grid.

    This accepts the A-label package; the old all-Unknown preparation gate is
    deliberately separate and unchanged. No source or bundle files are written.
    """
    data = package.read_bundle(bundle)
    manifest, protocol = data["manifest"], data["protocol"]
    inputs = manifest["inputs"]
    preparation = _preparation_root(preparation)
    if preparation != _preparation_root(Path(inputs["preparation"])):
        raise ValueError("Preparation binding mismatch")
    if file_digest(Path(inputs["protocol_path"])) != inputs["protocol_sha256"]:
        raise ValueError("Protocol source hash mismatch")
    if json.loads(Path(inputs["protocol_path"]).read_text()) != protocol:
        raise ValueError("Protocol source content mismatch")
    for source in data["evidence"]["sources"].values():
        path = Path(source["path"])
        if file_digest(path) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        if "size_bytes" in source and path.stat().st_size != source["size_bytes"]:
            raise ValueError(f"Source size mismatch: {path}")
    index = load_evidence(Path(inputs["tt_root"]), preparation, protocol)
    evidence = dict(
        sources=index.sources,
        cases=index.cases,
        preparation_manifest_sha256=index.preparation_manifest_sha256,
        rules_sha256=file_digest(Path(rules.__file__)),
    )
    if package._jsonable(evidence) != package._jsonable(data["evidence"]):
        raise ValueError("Reloaded evidence/provenance mismatch")
    if index.preparation_manifest_sha256 != inputs["preparation_manifest_sha256"]:
        raise ValueError("Preparation manifest hash mismatch")
    replay = classify_windows(index.grid, index, protocol)
    expected_labels = replay.labels.copy()
    expected_labels["label_version"] = manifest["label_version"]
    for name, expected in [("labels", expected_labels), ("splits", replay.splits)]:
        if set(expected) != set(data[name]) or package._records(expected) != package._records(
            data[name]
        ):
            raise ValueError(f"Recomputed {name} mismatch / row conservation failure")
    if package._exceptions(index.cases) != data["exceptions"]:
        raise ValueError("Recomputed exceptions mismatch")
    coverage = package._coverage(replay.labels, replay.splits, index.grid, index.cases)
    if package._jsonable(coverage) != data["coverage"]:
        raise ValueError("Recomputed coverage mismatch")
    eligibility = {}
    merged = replay.labels.merge(
        replay.splits[KEY_COLUMNS + ["split"]], on=KEY_COLUMNS, validate="one_to_one"
    )
    for modality in [c for c in index.grid if c.endswith("_valid")]:
        frame = merged.merge(
            index.grid[KEY_COLUMNS + [modality]], on=KEY_COLUMNS, validate="one_to_one"
        )
        eligibility[modality] = {
            role: int((frame.split.eq(role) & frame.label.eq(0) & frame[modality].eq(True)).sum())
            for role in ["fit", "calibration"]
        }
    return dict(
        rows=len(replay.labels),
        label_version=manifest["label_version"],
        split_version=manifest["split_version"],
        coverage=coverage,
        negative_eligibility=eligibility,
    )


def join_scores(
    labels: pd.DataFrame, splits: pd.DataFrame, scores: pd.DataFrame, expected_ids: set[str]
) -> pd.DataFrame:
    """Return only predeclared eligible known test rows, joined by sample_id.

    All three frames MUST declare nonempty label_version and split_version in
    DataFrame.attrs. Existing version columns must agree with those declarations.
    expected_ids is the fixed evaluation set after an independently chosen
    modality-validity mask; it cannot contain Unknown, fit or calibration rows.
    Extra scores (including Unknown alerts) are not counted in this result.
    No metrics or threshold algorithms are run; attrs report availability only.
    """
    versions = {key: labels.attrs.get(key) for key in ["label_version", "split_version"]}
    if any(not isinstance(v, str) or not v for v in versions.values()):
        raise ValueError("Missing version declarations")
    for frame in [labels, splits, scores]:
        for key, value in versions.items():
            if frame.attrs.get(key) != value or (key in frame and not frame[key].eq(value).all()):
                raise ValueError("Label/split version mismatch")
        if (
            "sample_id" not in frame
            or frame.sample_id.isna().any()
            or frame.sample_id.duplicated().any()
        ):
            raise ValueError("Missing or duplicate sample IDs")
    for other in [splits, scores]:
        common = [c for c in IDENTITY_COLUMNS if c != "sample_id" and c in labels and c in other]
        if common:
            compared = labels[["sample_id"] + common].merge(
                other[["sample_id"] + common],
                on="sample_id",
                suffixes=("_label", "_other"),
                validate="one_to_one",
            )
            if any(not compared[c + "_label"].eq(compared[c + "_other"]).all() for c in common):
                raise ValueError("Conflicting stable sample identity")
    if set(labels.sample_id) != set(splits.sample_id):
        raise ValueError("Label/split sample grid mismatch")
    if "label" not in labels or not labels.label.dropna().isin([0, 1]).all():
        raise ValueError("Invalid labels")
    if "split" not in splits or not splits.split.isin(["fit", "calibration", "test"]).all():
        raise ValueError("Invalid splits")
    if not isinstance(expected_ids, set) or not expected_ids <= set(labels.sample_id):
        raise ValueError("Invalid expected sample IDs")
    frame = labels.merge(splits[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    frame = frame.loc[frame.sample_id.isin(expected_ids)].copy()
    if frame.label.isna().any():
        raise ValueError("Unknown cannot enter known evaluation set")
    if not frame.split.eq("test").all():
        raise ValueError("Evaluation set contains fit/calibration samples")
    if "score" not in scores or not expected_ids <= set(scores.sample_id):
        raise ValueError("Missing required scores")
    selected = scores.loc[scores.sample_id.isin(expected_ids), ["sample_id", "score"]]
    try:
        finite = np.isfinite(selected.score.to_numpy(dtype=float)).all()
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise ValueError("Nonfinite required scores")
    result = (
        frame.merge(selected, on="sample_id", validate="one_to_one")
        .sort_values("sample_id")
        .reset_index(drop=True)
    )
    count = result.label.nunique()
    result.attrs.update(versions)
    result.attrs["metric_availability"] = dict(
        auroc=None,
        ap=None,
        reason="empty" if not count else "single_class" if count == 1 else "available_not_computed",
    )
    return result
