"""Audit three fixed postprocessing runs against the original GRU, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_gru_candidates import artifact
from scripts.service_model import check_run


def load(run):
    check_run(run)
    manifest = json.loads((run / "manifest.json").read_text())
    report = json.loads(artifact(run, manifest, "evaluation_report").read_text())
    return manifest, report


def audit(root, prefix):
    rows = []
    for seed in [1, 2, 42]:
        run = root / f"{prefix}-seed{seed}-20261024-a"
        baseline = root / f"temporal-multiseed-a7e7794-gru-seed{seed}-20261023-a"
        manifest, report = load(run)
        original_manifest, original_report = load(baseline)
        fit_ids = json.loads(artifact(run, manifest, "fit_ids").read_text())
        original_fit = json.loads(artifact(baseline, original_manifest, "fit_ids").read_text())
        inner = json.loads(artifact(run, manifest, "inner_audit").read_text())
        state = json.loads(artifact(run, manifest, "residual_scale_state").read_text())
        if sorted(fit_ids) != sorted(original_fit) or not set(state["fit_ids"]) <= set(fit_ids):
            raise ValueError("Score reference uses IDs outside original Normal fit")
        if state["fit_ids"] != sorted(inner["validation_ids"]):
            raise ValueError("Postprocessing reference differs from held-out Normal IDs")
        if manifest["training_versions"] != original_manifest["training_versions"]:
            raise ValueError("Evaluation versions differ")
        for name in ["prepared"]:
            if (
                manifest["provenance"]["parents"][name]
                != original_manifest["provenance"]["parents"][name]
            ):
                raise ValueError("Prepared lineage differs")
        for scope, name in [("test", "scores"), ("calibration", "calibration_scores")]:
            scored = (
                pd.read_parquet(artifact(run, manifest, name)).set_index("sample_id").sort_index()
            )
            previous = (
                pd.read_parquet(artifact(baseline, original_manifest, name))
                .set_index("sample_id")
                .sort_index()
            )
            if not scored.index.equals(previous.index):
                raise ValueError(f"Fixed {scope} IDs differ")
            if set(state["fit_ids"]) & set(scored.index):
                raise ValueError("Reference fitting intersects outer calibration/test")
            np.testing.assert_allclose(scored.original_score, previous.score, rtol=0, atol=1e-6)
            np.testing.assert_allclose(
                scored.availability_loss, previous.availability_loss, rtol=0, atol=1e-6
            )
        if report["main"]["sample_ids"] != original_report["main"]["sample_ids"]:
            raise ValueError("Main evaluation membership differs")
        row = dict(
            seed=seed,
            run_id=run.name,
            commit=manifest["provenance"]["environment"]["git_commit"],
            git_dirty=manifest["provenance"]["environment"]["git_dirty"],
            mlflow_run_id=manifest["mlflow_run_id"],
            state_sha256=manifest["artifacts"]["residual_scale_state"]["sha256"],
            normal_reference_count=len(state["fit_ids"]),
            normal_preprocessor_hash=inner["preprocessor_hash"],
            auroc=report["main"]["auroc"],
            ap=report["main"]["ap"],
            baseline_auroc=original_report["main"]["auroc"],
            baseline_ap=original_report["main"]["ap"],
            classification_q99=report["main"]["classification"]["0.99"],
            baseline_classification_q99=original_report["main"]["classification"]["0.99"],
            groups={
                scope: {
                    name: dict(
                        rows=group["rows"],
                        auroc=group["auroc"],
                        ap=group["ap"],
                        classification_q99=group["classification"]["0.99"],
                    )
                    for name, group in report["main"].get(scope, {}).items()
                }
                for scope in ["by_case", "by_service"]
            },
        )
        if "state_only_evaluation_report" in manifest["artifacts"]:
            state_report = json.loads(
                artifact(run, manifest, "state_only_evaluation_report").read_text()
            )
            row["state_only"] = {
                key: state_report["main"][key] for key in ["auroc", "ap", "rows", "classification"]
            }
        rows.append(row)
    summary = {}
    for metric in ["auroc", "ap"]:
        values = [row[metric] for row in rows]
        baseline = [row["baseline_" + metric] for row in rows]
        summary[metric] = dict(
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)),
            paired_delta=list(np.subtract(values, baseline)),
        )
    return dict(
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary,
        runs=rows,
        paired_metric_gate=all(
            delta > 0 for metric in summary.values() for delta in metric["paired_delta"]
        ),
        exploratory=True,
        protocol_check="same fit/cal/test/main IDs and versions; original scores and availability reproduced; held-out reference excludes outer cal/test",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.prefix), indent=2))


if __name__ == "__main__":
    main()
