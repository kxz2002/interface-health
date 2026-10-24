"""Read-only comparison of frozen GRU runs and their MLflow histories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def artifact(run, manifest, name):
    record = manifest["artifacts"][name]
    path = run / record["path"]
    payload = path.read_bytes()
    if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise ValueError(f"Artifact binding mismatch: {path}")
    return path


def residual_normal_gate(records):
    selected = {}
    for candidate in ["baseline", "residual"]:
        rows = [row for row in records if row["candidate"] == candidate]
        if sorted(row["seed"] for row in rows) != [1, 2, 42]:
            raise ValueError("Normal gate requires exactly three paired seeds")
        selected[candidate] = {row["seed"]: row for row in rows}
    deltas = []
    for seed in [1, 2, 42]:
        metrics = {}
        for metric in ["validation_loss", "value_loss"]:
            baseline = selected["baseline"][seed][metric]
            residual = selected["residual"][seed][metric]
            if not np.isfinite([baseline, residual]).all():
                raise ValueError("Nonfinite Normal gate loss")
            metrics[metric] = residual - baseline
        deltas.append(dict(seed=seed, **metrics))
    values = [row[metric] for row in deltas for metric in ["validation_loss", "value_loss"]]
    return dict(
        passed=all(value <= 0 for value in values) and any(value < 0 for value in values),
        deltas=deltas,
    )


def audit(root, residual_normal_run=None, residual_prefix=None):
    if residual_prefix is not None and residual_normal_run is None:
        raise ValueError("Residual outer audit requires its Normal validation run")
    prefixes = {
        "baseline": "temporal-multiseed-a7e7794-gru",
        "context8": "route-a-context8-a41aa70",
        "hidden64": "route-a-hidden64-da7f843",
    }
    rows, bindings = [], []
    if residual_prefix is not None:
        prefixes["residual"] = residual_prefix
    for candidate, prefix in prefixes.items():
        for seed in [1, 2, 42]:
            suffix = "20261024-a" if candidate == "residual" else "20261023-a"
            run = root / f"{prefix}-seed{seed}-{suffix}"
            manifest = json.loads((run / "manifest.json").read_text())
            if manifest["status"] != "success":
                raise ValueError(f"Unpublished run: {run}")
            history = json.loads(artifact(run, manifest, "training_history").read_text())
            database_path = Path(manifest["tracking_uri"].removeprefix("sqlite:///"))
            with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
                logged = database.execute(
                    "select step,value from metrics where run_uuid=? and key=? order by step",
                    (manifest["mlflow_run_id"], "fit_loss"),
                ).fetchall()
            expected = [(record["epoch"], record["mean_loss"]) for record in history]
            if len(expected) != 50 or logged != expected:
                raise ValueError(f"MLflow history mismatch: {run.name}")
            report = json.loads(artifact(run, manifest, "evaluation_report").read_text())
            scores = pd.read_parquet(artifact(run, manifest, "scores"))
            calibration = pd.read_parquet(artifact(run, manifest, "calibration_scores"))
            fit_ids = json.loads(artifact(run, manifest, "fit_ids").read_text())
            bindings.append(
                dict(
                    fit=sorted(fit_ids),
                    calibration=sorted(calibration.sample_id),
                    test=sorted(scores.sample_id),
                    main=sorted(report["main"]["sample_ids"]),
                    versions=manifest["training_versions"],
                    prepared=manifest["provenance"]["parents"]["prepared"],
                )
            )
            rows.append(
                dict(
                    candidate=candidate,
                    seed=seed,
                    run_id=run.name,
                    commit=manifest["provenance"]["environment"]["git_commit"],
                    git_dirty=manifest["provenance"]["environment"]["git_dirty"],
                    mlflow_run_id=manifest["mlflow_run_id"],
                    mlflow_epochs_verified=len(logged),
                    training_history=history,
                    auroc=report["main"]["auroc"],
                    ap=report["main"]["ap"],
                    classification_q99=report["main"]["classification"]["0.99"],
                    groups={
                        scope: {
                            name: {
                                key: group.get(key)
                                for key in ["rows", "auroc", "ap", "classification"]
                            }
                            for name, group in report["main"].get(scope, {}).items()
                        }
                        for scope in ["by_case", "by_service"]
                    },
                    calibration_components=calibration[["value_loss", "availability_loss"]]
                    .mean()
                    .to_dict(),
                )
            )
    if any(binding != bindings[0] for binding in bindings[1:]):
        raise ValueError("Comparison protocol or sample IDs differ")
    normal_run = root / "gru-normal-validation-0e495c0-20261023-a"
    manifest = json.loads((normal_run / "manifest.json").read_text())
    normal = json.loads(artifact(normal_run, manifest, "normal_validation_results").read_text())
    inner = json.loads(artifact(normal_run, manifest, "inner_split_and_preprocessing").read_text())
    sources = [(normal_run, manifest, normal)]
    if residual_normal_run is not None:
        extra_run = root / residual_normal_run
        extra_manifest = json.loads((extra_run / "manifest.json").read_text())
        if extra_manifest["status"] != "success":
            raise ValueError("Residual Normal validation is not published")
        extra_inner = json.loads(
            artifact(extra_run, extra_manifest, "inner_split_and_preprocessing").read_text()
        )
        for key in ["actual_fit_ids", "validation_ids", "preprocessor_hash"]:
            if extra_inner[key] != inner[key]:
                raise ValueError(f"Residual Normal validation differs: {key}")
        extra_rows = json.loads(
            artifact(extra_run, extra_manifest, "normal_validation_results").read_text()
        )
        if [(row["candidate"], row["seed"]) for row in extra_rows] != [
            ("residual", seed) for seed in [1, 2, 42]
        ]:
            raise ValueError("Residual validation must contain exactly the three frozen seeds")
        database_path = Path(extra_manifest["tracking_uri"].removeprefix("sqlite:///"))
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
            for row in extra_rows:
                logged = database.execute(
                    "select step,value from metrics where run_uuid=? and key=? order by step",
                    (extra_manifest["mlflow_run_id"], f"residual_seed{row['seed']}_fit_loss"),
                ).fetchall()
                expected = [
                    (record["epoch"], record["mean_loss"]) for record in row["training_history"]
                ]
                if len(expected) != 50 or logged != expected:
                    raise ValueError("Residual Normal MLflow history mismatch")
        sources.append((extra_run, extra_manifest, extra_rows))
        normal = normal + extra_rows
    services = []
    for source_run, source_manifest, records in sources:
        for record in records:
            candidate, seed = record["candidate"], record["seed"]
            scores = pd.read_parquet(
                artifact(source_run, source_manifest, f"{candidate}_seed{seed}_scores")
            )
            components = scores.groupby("canonical_service_id")[
                ["value_loss", "availability_loss"]
            ].mean()
            services.append(
                dict(candidate=candidate, seed=seed, services=components.to_dict("index"))
            )
    summary = {}
    for candidate in sorted(set(prefixes) | {row["candidate"] for row in normal}):
        summary[candidate] = {}
        for metric in ["auroc", "ap", "validation_loss", "value_loss", "availability_loss"]:
            source = rows if metric in {"auroc", "ap"} else normal
            selected = [row[metric] for row in source if row["candidate"] == candidate]
            if not selected:
                continue
            baseline = [row[metric] for row in source if row["candidate"] == "baseline"]
            summary[candidate][metric] = dict(
                mean=float(np.mean(selected)),
                std=float(np.std(selected, ddof=1)),
                paired_delta=list(np.subtract(selected, baseline)),
            )
    return dict(
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=summary,
        runs=rows,
        normal_validation=normal,
        normal_audit=dict(
            fit_cases=inner["fit_cases"],
            validation_case=inner["validation_case"],
            fit_count=len(inner["actual_fit_ids"]),
            validation_count=len(inner["validation_ids"]),
            preprocessor_hash=inner["preprocessor_hash"],
            runs=[
                dict(
                    run_id=run.name,
                    mlflow_run_id=record["mlflow_run_id"],
                    commit=record["provenance"]["environment"]["git_commit"],
                )
                for run, record, _ in sources
            ],
        ),
        normal_service_components=services,
        residual_normal_gate=(
            residual_normal_gate(normal) if residual_normal_run is not None else None
        ),
        comparison_bindings_sha256=hashlib.sha256(
            json.dumps(bindings[0], sort_keys=True).encode()
        ).hexdigest(),
        caveat="Outer results were inspected before inner Normal validation; exploratory, not blind confirmation.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--residual-normal-run")
    parser.add_argument("--residual-prefix")
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.residual_normal_run, args.residual_prefix), indent=2))


if __name__ == "__main__":
    main()
