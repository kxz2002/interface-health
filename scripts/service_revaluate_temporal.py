"""Re-evaluate published temporal scores under a derived evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import check_run, read_prepared
from src.service.labeling.acceptance import check_bundle
from src.service.labeling.bundle import read_bundle
from src.service.modeling.data import build_sample_lists
from src.service.modeling.evaluation import QUANTILES, calibrate, evaluate_run
from src.service.preparation import file_digest


def rebind(tables, prepared, bundle):
    derived = bundle["manifest"]
    if (
        derived.get("derivation", {}).get("source_manifest_sha256")
        != prepared.hashes["label_manifest_sha256"]
    ):
        raise ValueError("Derived bundle is not bound to the original prepared labels")
    for name in ("labels", "splits"):
        before = getattr(tables, name).sort_values("sample_id").reset_index(drop=True)
        after = bundle[name].sort_values("sample_id").reset_index(drop=True)
        if name == "labels":
            before = before.drop(columns="label_version")
            after = after.drop(columns="label_version")
        pd.testing.assert_frame_equal(before, after, check_like=True, check_flags=False)
    versions = dict(tables.versions)
    versions.update(
        label_version=derived["label_version"],
        split_version=derived["split_version"],
        evaluation_version=derived["evaluation_version"],
    )
    updated = replace(
        tables,
        labels=bundle["labels"],
        splits=bundle["splits"],
        evaluation=bundle["evaluation"],
        protocol=bundle["protocol"],
        versions=versions,
    )
    lists = build_sample_lists(
        updated, prepared.numeric, pd.Series(prepared.logs["masks"]), "common"
    )
    if lists.fit != prepared.lists.fit or lists.calibration != prepared.lists.calibration:
        raise ValueError("Fit or calibration membership changed")
    return updated, lists


def evaluate_source(source, tables, prepared, lists, prepared_path):
    check_run(source)
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    provenance = manifest["provenance"]
    for name in ("scores", "calibration_scores", "thresholds"):
        if (
            manifest["artifacts"][name]["path"]
            != f"run/{name}.{'parquet' if name != 'thresholds' else 'json'}"
        ):
            raise ValueError(f"Unexpected {name} artifact path")
    parent = provenance["parents"]["prepared"]
    if parent["manifest_sha256"] != file_digest(Path(prepared_path) / "manifest.json"):
        raise ValueError("Model was fitted with a different prepared artifact")
    if (
        provenance["lists_hash"] != prepared.lists.lists_hash
        or provenance["versions"] != prepared.versions
    ):
        raise ValueError("Model training membership or versions differ")
    scores = pd.read_parquet(source / "run/scores.parquet")
    calibration = pd.read_parquet(source / "run/calibration_scores.parquet")
    thresholds = json.loads((source / "run/thresholds.json").read_text())
    computed = calibrate(calibration, set(lists.calibration), list(QUANTILES))
    if thresholds != computed:
        raise ValueError("Stored thresholds do not match original calibration scores")
    report = evaluate_run(
        tables,
        lists,
        scores,
        thresholds,
        score_scope="test",
        source_binding={"versions": prepared.versions, "lists_hash": prepared.lists.lists_hash},
    )
    return report, dict(
        source=str(source.resolve()), manifest_sha256=file_digest(source / "manifest.json")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--preparation", required=True, type=Path)
    parser.add_argument(
        "--sources", required=True, type=Path, help="JSON mapping names to published runs"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    check_bundle(args.bundle, args.preparation)
    tables, prepared, _ = read_prepared(args.prepared)
    bundle = read_bundle(args.bundle)
    tables, lists = rebind(tables, prepared, bundle)
    sources = json.loads(args.sources.read_text())
    if not sources or any(not isinstance(name, str) or not name.isidentifier() for name in sources):
        raise ValueError("Source names must be nonempty Python identifiers")
    args.output.mkdir(parents=True)
    records = {}
    for name, path in sources.items():
        report, reference = evaluate_source(path, tables, prepared, lists, args.prepared)
        report_path = args.output / f"{name}.json"
        report_path.write_text(json.dumps(report, sort_keys=True, allow_nan=False))
        records[name] = {**reference, "report_sha256": file_digest(report_path)}
        print(f"completed {name}", flush=True)
    summary = dict(
        status="success",
        sources=records,
        source_list_sha256=file_digest(args.sources),
        prepared_manifest_sha256=file_digest(args.prepared / "manifest.json"),
        bundle_manifest_sha256=file_digest(
            args.bundle / "manifest.json"
            if (args.bundle / "manifest.json").is_file()
            else args.bundle / "labels/manifest.json"
        ),
        evaluation_version=tables.versions["evaluation_version"],
        lists_hash=lists.lists_hash,
    )
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
