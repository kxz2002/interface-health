"""Audit frozen architecture runs without fitting or model selection."""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import service_temporal_model
from scripts.audit_gru_candidates import artifact
from scripts.audit_gru_postprocessing import load
from scripts.service_model import check_run, read_prepared
from src.service.modeling.temporal import (
    restore_temporal_model,
    temporal_inputs,
    temporal_model_metadata,
)


def frozen_config(candidate):
    if candidate not in {"service_bias", "short_long"}:
        raise ValueError("Unknown independent architecture candidate")
    path = Path(service_temporal_model.__file__).resolve().parents[1] / (
        f"configs/service/tt_temporal_prior_gru_{candidate}_gpu_b_v1.json"
    )
    config = json.loads(path.read_text())
    service_temporal_model.validate_temporal_config(config)
    return config


def check_checkpoint(checkpoint, expected, fit_ids, identity, inputs, *, seed, inner=False):
    expected_config = copy.deepcopy(expected)
    settings = copy.deepcopy(expected["training"])
    settings["seed"] = seed
    if not inner:
        expected_config["training"]["seed"] = seed
        expected_config["isolation_forest"]["random_state"] = seed
        expected_config["method"] = "gru"
    if checkpoint["config"] != expected_config:
        raise ValueError("Checkpoint differs from the frozen candidate configuration")
    if inner and checkpoint.get("settings") != settings:
        raise ValueError("Inner checkpoint settings or seed differ")
    if checkpoint["fit_ids"] != sorted(fit_ids):
        raise ValueError("Checkpoint fit IDs differ from the frozen fit membership")
    metadata = checkpoint["architecture_metadata"]
    expected_services = (
        sorted(set(identity.loc[sorted(fit_ids), "canonical_service_id"].astype(str)))
        if settings["architecture"] == "service_bias_gru"
        else []
    )
    if metadata["service_vocabulary"] != expected_services:
        raise ValueError("Service conditioning vocabulary differs from fit-only services")
    model = restore_temporal_model(checkpoint, inputs)
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite parameters")
    actual = temporal_model_metadata(model)
    for name in ["architecture", "service_vocabulary", "parameter_count"]:
        if actual[name] != metadata[name]:
            raise ValueError(f"Restored architecture metadata differs: {name}")
    if actual["service_bias_norm"] is not None:
        np.testing.assert_allclose(
            actual["service_bias_norm"], metadata["service_bias_norm"], rtol=1e-5, atol=1e-6
        )
        if actual["service_bias_norms"].keys() != metadata["service_bias_norms"].keys():
            raise ValueError("Per-service bias norm membership differs")
        np.testing.assert_allclose(
            list(actual["service_bias_norms"].values()),
            list(metadata["service_bias_norms"].values()),
            rtol=1e-5,
            atol=1e-6,
        )
    return actual


def check_contexts(run, manifest, prepared, length):
    histories = json.loads(artifact(run, manifest, "context_sample_ids").read_text())
    identity = prepared.identity.set_index("sample_id")
    splits = prepared.splits.set_index("sample_id").split
    ids = sorted(prepared.sample_ids)
    if sorted(histories) != ids:
        raise ValueError("Context artifact does not cover the prepared sample membership")
    lookup = {
        (
            row.case_id,
            row.canonical_service_id,
            splits.loc[sample_id],
            int(row.window_start_ms),
        ): sample_id
        for sample_id, row in identity.iterrows()
    }
    if len(lookup) != len(ids):
        raise ValueError("Duplicate context case/service/split/time key")
    fit = set(prepared.lists.fit)
    for sample_id in ids:
        row = identity.loc[sample_id]
        expected = []
        for lag in range(1, length + 1):
            previous = lookup.get(
                (
                    row.case_id,
                    row.canonical_service_id,
                    splits.loc[sample_id],
                    int(row.window_start_ms) - lag * 15000,
                )
            )
            if previous is None or (splits.loc[sample_id] == "fit" and previous not in fit):
                break
            expected.append(previous)
        if histories[sample_id] != list(reversed(expected)):
            raise ValueError(f"Noncausal, cross-boundary or noncontiguous history: {sample_id}")


def audit(root, prefix, normal_run, candidate):
    expected = frozen_config(candidate)
    check_run(normal_run)
    normal_manifest = json.loads((normal_run / "manifest.json").read_text())
    prepared_reference = normal_manifest["provenance"]["parents"]["prepared"]
    _, prepared, _ = read_prepared(prepared_reference["path"])
    inputs = temporal_inputs(prepared, expected["context_windows"])
    identity = prepared.identity.set_index("sample_id")
    labels = prepared.labels.set_index("sample_id")
    splits = prepared.splits.set_index("sample_id")
    fit_ids = sorted(prepared.lists.fit)
    if not (
        labels.loc[fit_ids, "label"].eq(0).all()
        and labels.loc[fit_ids, "label_source"].eq("normal_run_reference").all()
        and splits.loc[fit_ids, "split"].eq("fit").all()
    ):
        raise ValueError("Outer fit contains non-Normal or non-fit samples")
    normal_rows = json.loads(
        artifact(normal_run, normal_manifest, "normal_validation_results").read_text()
    )
    if sorted((row["candidate"], row["seed"]) for row in normal_rows) != [
        (candidate, seed) for seed in [1, 2, 42]
    ]:
        raise ValueError("Normal validation candidates or seeds differ")
    inner = json.loads(
        artifact(normal_run, normal_manifest, "inner_split_and_preprocessing").read_text()
    )
    if set(inner["actual_fit_ids"]) & set(inner["validation_ids"]):
        raise ValueError("Inner Normal fit and validation overlap")
    original_normal = root / "gru-normal-validation-0e495c0-20261023-a"
    check_run(original_normal)
    original_normal_manifest = json.loads((original_normal / "manifest.json").read_text())
    original_inner = json.loads(
        artifact(
            original_normal, original_normal_manifest, "inner_split_and_preprocessing"
        ).read_text()
    )
    for key in ["actual_fit_ids", "validation_ids", "preprocessor_hash"]:
        if inner[key] != original_inner[key]:
            raise ValueError("Inner Normal preprocessing differs")
    normal_database = Path(normal_manifest["tracking_uri"].removeprefix("sqlite:///"))
    with sqlite3.connect(f"file:{normal_database}?mode=ro", uri=True) as database:
        for row in normal_rows:
            checkpoint = torch.load(
                artifact(normal_run, normal_manifest, f"{candidate}_seed{row['seed']}_checkpoint"),
                map_location="cpu",
                weights_only=True,
            )
            check_checkpoint(
                checkpoint,
                expected,
                inner["actual_fit_ids"],
                identity,
                inputs,
                seed=row["seed"],
                inner=True,
            )
            if row["architecture_metadata"] != checkpoint["architecture_metadata"]:
                raise ValueError("Normal result architecture differs from its checkpoint")
            logged = database.execute(
                "select step,value from metrics where run_uuid=? and key=? order by step",
                (normal_manifest["mlflow_run_id"], f"{candidate}_seed{row['seed']}_fit_loss"),
            ).fetchall()
            if len(logged) != 50 or logged != [
                (record["epoch"], record["mean_loss"]) for record in row["training_history"]
            ]:
                raise ValueError("Normal validation MLflow history differs")
    rows = []
    for seed in [1, 2, 42]:
        run = root / f"{prefix}-seed{seed}-20261024-a"
        baseline = root / f"temporal-multiseed-a7e7794-gru-seed{seed}-20261023-a"
        manifest, report = load(run)
        original, previous = load(baseline)
        fit = json.loads(artifact(run, manifest, "fit_ids").read_text())
        baseline_fit = json.loads(artifact(baseline, original, "fit_ids").read_text())
        if sorted(fit) != sorted(baseline_fit):
            raise ValueError("Normal fit membership differs")
        checkpoint = torch.load(
            artifact(run, manifest, "model"), map_location="cpu", weights_only=True
        )
        check_checkpoint(checkpoint, expected, fit, identity, inputs, seed=seed)
        run_config = json.loads((run / "run/config.json").read_text())
        if run_config != {
            **checkpoint["config"],
            "tracking_uri": manifest["tracking_uri"],
            "experiment_name": "service-model-infrastructure",
        }:
            raise ValueError("Published configuration differs from its checkpoint")
        if manifest["provenance"]["parents"]["prepared"] != prepared_reference:
            raise ValueError("Inner and outer prepared bindings differ")
        check_contexts(run, manifest, prepared, expected["context_windows"])
        if checkpoint["fit_ids"] != sorted(fit) or checkpoint["config"]["training"]["seed"] != seed:
            raise ValueError("Checkpoint fit or seed differs")
        if not set(inner["actual_fit_ids"] + inner["validation_ids"]) <= set(fit):
            raise ValueError("Normal validation exceeds outer fit")
        if manifest["training_versions"] != original["training_versions"]:
            raise ValueError("Training versions differ")
        if (
            manifest["provenance"]["parents"]["prepared"]
            != original["provenance"]["parents"]["prepared"]
        ):
            raise ValueError("Prepared binding differs")
        for name in ["scores", "calibration_scores"]:
            scores = pd.read_parquet(artifact(run, manifest, name))
            old = pd.read_parquet(artifact(baseline, original, name))
            if (
                sorted(scores.sample_id) != sorted(old.sample_id)
                or scores.sample_id.duplicated().any()
            ):
                raise ValueError("Score membership differs")
            if set(scores.sample_id) & set(fit):
                raise ValueError("Outer fit intersects scoring")
        if report["main"]["sample_ids"] != previous["main"]["sample_ids"]:
            raise ValueError("Main membership differs")
        history = json.loads(artifact(run, manifest, "training_history").read_text())
        database_path = Path(manifest["tracking_uri"].removeprefix("sqlite:///"))
        with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as database:
            logged = database.execute(
                "select step,value from metrics where run_uuid=? and key=? order by step",
                (manifest["mlflow_run_id"], "fit_loss"),
            ).fetchall()
        if len(history) != 50 or logged != [
            (record["epoch"], record["mean_loss"]) for record in history
        ]:
            raise ValueError("MLflow training history differs")
        rows.append(
            dict(
                seed=seed,
                run_id=run.name,
                commit=manifest["provenance"]["environment"]["git_commit"],
                git_dirty=manifest["provenance"]["environment"]["git_dirty"],
                mlflow_run_id=manifest["mlflow_run_id"],
                auroc=report["main"]["auroc"],
                ap=report["main"]["ap"],
                baseline_auroc=previous["main"]["auroc"],
                baseline_ap=previous["main"]["ap"],
                classification_q99=report["main"]["classification"]["0.99"],
                groups={
                    scope: {
                        name: {key: group[key] for key in ["rows", "auroc", "ap", "classification"]}
                        for name, group in report["main"][scope].items()
                    }
                    for scope in ["by_case", "by_service"]
                },
                fit_summary=json.loads(artifact(run, manifest, "fit_summary").read_text()),
                training_history=history,
            )
        )
    summary = {
        metric: dict(
            mean=float(np.mean([row[metric] for row in rows])),
            std=float(np.std([row[metric] for row in rows], ddof=1)),
            paired_delta=[row[metric] - row["baseline_" + metric] for row in rows],
        )
        for metric in ["auroc", "ap"]
    }
    return dict(
        exploratory=True,
        summary=summary,
        runs=rows,
        normal_validation=normal_rows,
        paired_metric_gate=all(
            delta > 0 for values in summary.values() for delta in values["paired_delta"]
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--normal-run", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.prefix, args.normal_run, args.candidate), indent=2))
