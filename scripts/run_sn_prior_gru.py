"""Run the project's prior GRU on the SN exploratory preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.service.modeling.availability import fit_availability
from src.service.modeling.sn_adapter import build_sn_prepared
from src.service.modeling.temporal import score_temporal, temporal_inputs, train_temporal


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _proxy_metrics(scores: pd.DataFrame, prepared) -> dict:
    labels = prepared.labels.set_index("sample_id").label.reindex(scores.sample_id).to_numpy()
    known = pd.notna(labels)
    y = labels[known].astype(int)
    values = scores.score.to_numpy(float)[known]
    metrics = {
        "rows": int(known.sum()),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "positive_prevalence": float((y == 1).mean()),
        "constant_score_ap_baseline": float((y == 1).mean()),
        "label_source": "controlled_exposure_proxy; not independent Service gold",
        "auroc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
        "ap": float(average_precision_score(y, values)) if len(np.unique(y)) == 2 else None,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-windows", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--tracking-uri", type=str, default="sqlite:///outputs/service/sn-gru-mlflow.db"
    )
    parser.add_argument("--experiment-name", type=str, default="sn-prior-gru")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = build_sn_prepared(args.prepared_root)
    config = json.loads(Path("configs/service/sn_temporal_prior_gpu_v1.json").read_text())
    config["training"]["seed"] = args.seed
    if args.context_windows is not None:
        config["context_windows"] = args.context_windows
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if config["context_windows"] not in {4, 8, 12}:
        raise ValueError("SN prior GRU context must be 4, 8 or 12")
    if type(config["training"]["epochs"]) is not int or config["training"]["epochs"] < 1:
        raise ValueError("SN prior GRU epochs must be positive")
    prior = fit_availability(prepared, prepared.lists.fit)
    inputs = temporal_inputs(prepared, config["context_windows"], availability_prior=prior)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.output.name) as tracking_run:
        mlflow.log_params(
            {
                "seed": args.seed,
                "context_windows": config["context_windows"],
                "epochs": config["training"]["epochs"],
                "hidden_dim": config["training"]["hidden_dim"],
                "batch_size": config["training"]["batch_size"],
                "lr": config["training"]["lr"],
                "weight_decay": config["training"]["weight_decay"],
                "prepared_root": str(args.prepared_root.resolve()),
            }
        )
        started = time.perf_counter()
        model, history = train_temporal(
            prepared,
            inputs,
            config["training"],
            progress=lambda record: mlflow.log_metric(
                "fit_loss", record["mean_loss"], step=record["epoch"]
            ),
        )
        fit_elapsed = time.perf_counter() - started
        mlflow.log_metric("fit_elapsed_seconds", fit_elapsed)
    calibration = score_temporal(model, inputs, prepared.lists.calibration)
    test = score_temporal(model, inputs, prepared.lists.main_test)
    thresholds = {
        str(q): float(np.quantile(calibration.score.to_numpy(float), q))
        for q in config["quantiles"]
    }
    test_metrics = _proxy_metrics(test, prepared)
    args.output.mkdir(parents=True)
    (args.output / "run").mkdir()
    calibration.to_parquet(args.output / "run/calibration_scores.parquet", index=False)
    test.to_parquet(args.output / "run/scores.parquet", index=False)
    torch.save(
        {"model_state": model.state_dict(), "config": config, "fit_ids": list(prepared.lists.fit)},
        args.output / "run/model.pt",
    )
    report = {
        "protocol": "sn_prior_gru_role_proxy_v1",
        "thresholds": thresholds,
        "primary_threshold": thresholds["0.99"],
        "threshold_metrics": {},
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "proxy_metrics": test_metrics,
        "fit_elapsed_seconds": fit_elapsed,
        "history": history,
        "mlflow_run_id": tracking_run.info.run_id,
        "limitations": [
            "SN source labels remain Unknown; fault-run positives are controlled exposure proxies.",
            "SN resource units and timezone alignment remain exploratory assumptions.",
        ],
    }
    labels = prepared.labels.set_index("sample_id").label.reindex(test.sample_id).to_numpy(int)
    for quantile, threshold in thresholds.items():
        predicted = (test.score.to_numpy(float) > threshold).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score

        report["threshold_metrics"][quantile] = {
            "f1": float(f1_score(labels, predicted)),
            "recall": float(recall_score(labels, predicted)),
            "precision": float(precision_score(labels, predicted)),
            "alerts": int(predicted.sum()),
        }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    manifest = {
        "stage": "sn_prior_gru",
        "seed": args.seed,
        "prepared_root": str(args.prepared_root.resolve()),
        "prepared_artifacts": {
            name: _digest(args.prepared_root / "sn" / name)
            for name in [
                "resource_windows.parquet",
                "execution_windows.parquet",
                "log_windows.parquet",
                "labels.parquet",
                "splits.parquet",
            ]
        },
        "source_label_status": "all Unknown",
        "model_label_status": "role/proxy derived; not gold",
        "lists_hash": prepared.lists.lists_hash,
        "versions": prepared.versions,
        "provenance": prepared.provenance,
        "feature_order": inputs["feature_order"],
        "config": config,
        "fit_ids": list(prepared.lists.fit),
        "calibration_ids": list(prepared.lists.calibration),
        "test_ids": list(prepared.lists.main_test),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output), "proxy_metrics": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
