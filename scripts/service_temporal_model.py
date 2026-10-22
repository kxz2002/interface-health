"""Run a fixed Normal-only temporal predictor or its identical-input lag IF control."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_downstream import fit_iforest, score_iforest
from scripts.service_model import (
    _json,
    _publish,
    _record,
    _reference,
    check_run,
    read_prepared,
    score_ids,
)
from src.service.modeling.config import content_hash
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.temporal import (
    _fit_indices,
    lag_features,
    score_temporal,
    temporal_inputs,
    train_temporal,
)


def validate_temporal_config(config):
    if (
        not isinstance(config, dict)
        or set(config)
        != {
            "schema_version",
            "context_windows",
            "window_ms",
            "value_transform",
            "score",
            "training",
            "isolation_forest",
            "quantiles",
        }
        or not isinstance(config["training"], dict)
        or set(config["training"])
        != {
            "seed",
            "device",
            "hidden_dim",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "gradient_clip",
            "checkpoint",
        }
    ):
        raise ValueError("Incomplete or unknown temporal configuration keys")
    if (
        config["schema_version"] != "service-temporal-predictor-v1"
        or config["context_windows"] != 4
        or config["window_ms"] != 15000
        or config["value_transform"] != "asinh_prepared_normal_fit_z"
        or config["score"] != "masked_value_mse_plus_modality_bce"
        or config["training"]["checkpoint"] != "last"
        or config["training"]["device"] not in {"cpu", "cuda"}
        or config["isolation_forest"] != dict(n_estimators=100, n_jobs=1, random_state=42)
        or config["quantiles"] != [0.95, 0.975, 0.99, 0.995]
    ):
        raise ValueError("Unsupported temporal experiment protocol")
    settings = config["training"]
    for name in ["hidden_dim", "epochs", "batch_size"]:
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError("Temporal dimensions, epochs and batch size must be positive integers")
    if type(settings["seed"]) is not int or settings["seed"] != 42:
        raise ValueError("Temporal prototype seed must be 42")
    for name in ["lr", "weight_decay", "gradient_clip"]:
        if (
            type(settings[name]) not in {int, float}
            or not math.isfinite(settings[name])
            or settings[name] <= 0
        ):
            raise ValueError("Temporal optimization settings must be finite and positive")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--method", choices=["gru", "lag_IF"], required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    validate_temporal_config(config)
    tables, prepared, _ = read_prepared(args.prepared)
    inputs = temporal_inputs(prepared, config["context_windows"])
    indices = _fit_indices(prepared, inputs)
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    config["method"] = args.method
    binding = content_hash(dict(config=config, prepared=_reference(args.prepared)))
    provenance = dict(
        stage="temporal_experiment",
        method=args.method,
        exploratory=True,
        versions=prepared.versions,
        parents=dict(prepared=_reference(args.prepared)),
        lists_hash=prepared.lists.lists_hash,
        model_sha256=binding,
        feature_order=inputs["feature_order"],
        context_policy="strict_past_same_case_service_split_15s_contiguous",
    )
    with _record(args.output, config, args, provenance) as recorder:
        fit_started = time.perf_counter()
        model_path = recorder.output / "run/model.pt"
        if args.method == "gru":
            model, history = train_temporal(
                prepared,
                inputs,
                config["training"],
                progress=lambda record: recorder.log_epoch(
                    {"fit_loss": record["mean_loss"], "elapsed_seconds": record["elapsed_seconds"]},
                    step=record["epoch"],
                ),
            )
            torch.save(
                dict(
                    model_state=model.state_dict(),
                    config=config,
                    fit_ids=sorted(prepared.lists.fit),
                ),
                model_path,
            )
        else:
            features = lag_features(inputs)
            model = fit_iforest(features[indices], config["training"]["seed"])
            torch.save(
                dict(estimator=model, config=config, fit_ids=sorted(prepared.lists.fit)), model_path
            )
            history = []
        fit_elapsed = time.perf_counter() - fit_started
        scored = {}
        for scope in ["calibration", "test"]:
            ids = sorted(score_ids(prepared.lists, scope))
            if args.method == "gru":
                frame = score_temporal(model, inputs, ids)
            else:
                selected = [lookup[sample_id] for sample_id in ids]
                frame = inputs["identity"].set_index("sample_id").loc[ids].reset_index()
                frame["score"] = score_iforest(model, features[selected])
            frame["model_version"] = config["schema_version"] + "-" + args.method
            frame["config_sha256"] = binding
            frame["lists_hash"] = prepared.lists.lists_hash
            for name, version in prepared.versions.items():
                frame[name] = version
            scored[scope] = frame
        calibration = annotate_calibration(
            scored["calibration"], prepared, set(prepared.lists.calibration)
        )
        thresholds = calibrate(calibration, set(prepared.lists.calibration), config["quantiles"])
        report = evaluate_run(
            tables, prepared.lists, scored["test"], thresholds, score_scope="test"
        )
        recorder.log_epoch(
            {"main_auroc": report["main"]["auroc"], "main_ap": report["main"]["ap"]},
            step=config["training"]["epochs"] + 1,
        )
        report_path = recorder.output / "evaluation/report.json"
        _json(report_path, report)
        scores_path = recorder.output / "run/scores.parquet"
        scored["test"].to_parquet(scores_path, index=False)
        calibration_path = recorder.output / "run/calibration_scores.parquet"
        calibration.to_parquet(calibration_path, index=False)
        _publish(
            recorder,
            dict(
                model=model_path,
                training_history=history,
                evaluation_report=report_path,
                thresholds=thresholds,
                scores=scores_path,
                calibration_scores=calibration_path,
                fit_ids=sorted(prepared.lists.fit),
                context_sample_ids=inputs["history_ids"],
                feature_order=inputs["feature_order"],
                fit_summary=dict(
                    method=args.method,
                    elapsed_seconds=fit_elapsed,
                    fit_windows=len(indices),
                    seed=config["training"]["seed"],
                    checkpoint="last",
                ),
            ),
        )
    print(json.dumps(check_run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
