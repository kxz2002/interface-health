"""Run a fixed Normal-only temporal predictor or its identical-input lag IF control."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
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
from src.service.modeling.availability import fit_availability
from src.service.modeling.config import content_hash
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.temporal import (
    _fit_indices,
    lag_features,
    score_temporal,
    temporal_inputs,
    temporal_model_metadata,
    train_temporal,
)
from src.service.modeling.text_embeddings import build_frozen_text_embeddings


def validate_temporal_config(config):
    expected_keys = {
        "schema_version",
        "context_windows",
        "window_ms",
        "value_transform",
        "score",
        "training",
        "isolation_forest",
        "quantiles",
    }
    prior_enabled = isinstance(config, dict) and config.get("schema_version") in {
        "service-temporal-prior-v1",
        "service-temporal-prior-bert-v1",
        "service-temporal-prior-residual-v1",
        "service-temporal-prior-service-bias-v1",
        "service-temporal-prior-short-long-v1",
        "service-temporal-prior-multiscale-v1",
        "service-temporal-prior-grouped-v1",
    }
    bert_enabled = (
        isinstance(config, dict)
        and config.get("schema_version") == "service-temporal-prior-bert-v1"
    )
    if prior_enabled:
        expected_keys.add("availability_prior")
    if (
        isinstance(config, dict)
        and config.get("schema_version") == "service-temporal-prior-multiscale-v1"
    ):
        expected_keys.add("feature_mode")
    if bert_enabled:
        expected_keys.add("text_embedding")
        if "text_control" in config:
            expected_keys.add("text_control")
    residual_enabled = (
        isinstance(config, dict)
        and config.get("schema_version") == "service-temporal-prior-residual-v1"
    )
    training_keys = {
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
    if residual_enabled:
        training_keys.add("value_prediction")
    architecture = (
        {
            "service-temporal-prior-service-bias-v1": "service_bias_gru",
            "service-temporal-prior-short-long-v1": "short_long_gru",
            "service-temporal-prior-grouped-v1": "grouped_gru",
        }.get(config.get("schema_version"))
        if isinstance(config, dict)
        else None
    )
    if architecture:
        training_keys.add("architecture")
    if (
        not isinstance(config, dict)
        or set(config) != expected_keys
        or not isinstance(config["training"], dict)
        or set(config["training"]) != training_keys
    ):
        raise ValueError("Incomplete or unknown temporal configuration keys")
    if (
        config["schema_version"]
        not in {
            "service-temporal-predictor-v1",
            "service-temporal-prior-v1",
            "service-temporal-prior-bert-v1",
            "service-temporal-prior-residual-v1",
            "service-temporal-prior-service-bias-v1",
            "service-temporal-prior-short-long-v1",
        }
        or (
            prior_enabled
            and config["availability_prior"]
            not in [
                dict(source=source, smoothing=1.0, integration="logit_offset")
                for source in ["normal_fit_per_service", "normal_fit_global"]
            ]
        )
        or config["context_windows"] not in {4, 8}
        or config.get("feature_mode", "base") not in {"base", "multiscale"}
        or config["window_ms"] != 15000
        or config["value_transform"] != "asinh_prepared_normal_fit_z"
        or config["score"] != "masked_value_mse_plus_modality_bce"
        or config.get("text_control", "actual") not in {"actual", "zero", "reverse_past"}
        or config["training"]["checkpoint"] != "last"
        or config["training"]["device"] not in {"cpu", "cuda"}
        or config["isolation_forest"]
        != dict(n_estimators=100, n_jobs=1, random_state=config["training"]["seed"])
        or config["quantiles"] != [0.95, 0.975, 0.99, 0.995]
        or (
            architecture
            and (
                config["training"]["architecture"] != architecture
                or config["context_windows"] != (8 if architecture == "short_long_gru" else 4)
                or config["training"]["hidden_dim"]
                != (16 if architecture in {"short_long_gru", "grouped_gru"} else 32)
            )
        )
        or (
            config["schema_version"] == "service-temporal-prior-multiscale-v1"
            and config.get("feature_mode") != "multiscale"
        )
        or (
            config["schema_version"] == "service-temporal-prior-grouped-v1"
            and config["training"].get("architecture") != "grouped_gru"
        )
        or (
            residual_enabled
            and (
                config["training"]["value_prediction"] != "past_observed_residual"
                or config["context_windows"] != 4
            )
        )
        or (
            bert_enabled
            and config["text_embedding"]
            != dict(
                model="google/bert_uncased_L-4_H-256_A-4",
                revision="387825ce42dbb39b87911cdf8e383ee3b25184f8",
                pooling="body_token_then_group_mean",
                batch_size=32,
            )
        )
    ):
        raise ValueError("Unsupported temporal experiment protocol")
    settings = config["training"]
    for name in ["hidden_dim", "epochs", "batch_size"]:
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError("Temporal dimensions, epochs and batch size must be positive integers")
    if type(settings["seed"]) is not int or settings["seed"] not in {1, 2, 42}:
        raise ValueError("Temporal seed must be one of 1, 2, 42")
    for name in ["lr", "weight_decay", "gradient_clip"]:
        if (
            type(settings[name]) not in {int, float}
            or not math.isfinite(settings[name])
            or settings[name] <= 0
        ):
            raise ValueError("Temporal optimization settings must be finite and positive")


def load_text_cache(run, prepared_run, prepared, embedding_config):
    """Reuse only a published embedding cache bound to the identical prepared input."""
    check_run(run)
    root = Path(run)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("stage") != "temporal_experiment":
        raise ValueError("Text cache must come from a temporal experiment")
    if manifest["provenance"]["parents"].get("prepared") != _reference(prepared_run):
        raise ValueError("Text cache prepared parent mismatch")
    cached_config = json.loads((root / "run/config.json").read_text())
    if (
        cached_config.get("text_embedding") != embedding_config
        or manifest["provenance"].get("text_embedding") != embedding_config
    ):
        raise ValueError("Text cache model configuration mismatch")
    records = manifest["artifacts"]
    for name, expected in [
        ("text_embeddings", "run/text_embeddings.npy"),
        ("text_embedding_metadata", "run/text_embedding_metadata.json"),
    ]:
        if name not in records or records[name]["path"] != expected:
            raise ValueError("Missing declared text cache artifacts")
    metadata = json.loads((root / "run/text_embedding_metadata.json").read_text())
    expected_metadata = dict(
        sample_ids=sorted(prepared.sample_ids),
        embedding_dim=256,
        config=embedding_config,
        prepared_log_cache_key=prepared.logs["cache"]["cache_key"],
    )
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("Text cache metadata/IDs/config mismatch")
    vectors = np.load(root / "run/text_embeddings.npy", allow_pickle=False)
    if (
        vectors.dtype != np.float32
        or vectors.shape != (len(prepared.sample_ids), 256)
        or not np.isfinite(vectors).all()
        or hashlib.sha256(vectors.tobytes(order="C")).hexdigest() != metadata.get("vectors_sha256")
    ):
        raise ValueError("Text cache vectors shape/dtype/hash mismatch")
    return vectors, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--method", choices=["gru", "lag_IF", "service_lag_IF"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42])
    parser.add_argument("--text-cache-run", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    validate_temporal_config(config)
    if "architecture" in config["training"] and args.method != "gru":
        raise ValueError("Independent GRU architectures require method gru")
    if args.text_cache_run is not None and "text_embedding" not in config:
        raise ValueError("Text cache requires the BERT architecture")
    if args.seed is not None:
        config["training"]["seed"] = args.seed
        config["isolation_forest"]["random_state"] = args.seed
    validate_temporal_config(config)
    tables, prepared, _ = read_prepared(args.prepared)
    prior = (
        fit_availability(prepared, prepared.lists.fit) if "availability_prior" in config else None
    )
    text_embeddings = None
    text_embedding_elapsed = None
    text_metadata = None
    if "text_embedding" in config:
        log_config = tables.config["log"]
        if (
            log_config["model"] != config["text_embedding"]["model"]
            or log_config["tokenizer_revision"] != config["text_embedding"]["revision"]
        ):
            raise ValueError("Frozen text model must match the prepared tokenizer revision")
        text_started = time.perf_counter()
        if args.text_cache_run is not None:
            text_embeddings, _ = load_text_cache(
                args.text_cache_run, args.prepared, prepared, config["text_embedding"]
            )
        else:
            text_embeddings = build_frozen_text_embeddings(
                prepared,
                sorted(prepared.sample_ids),
                model_name=config["text_embedding"]["model"],
                revision=config["text_embedding"]["revision"],
                device=config["training"]["device"],
                batch_size=config["text_embedding"]["batch_size"],
            )
        text_embedding_elapsed = time.perf_counter() - text_started
        text_metadata = dict(
            sample_ids=sorted(prepared.sample_ids),
            embedding_dim=text_embeddings.shape[1],
            config=config["text_embedding"],
            prepared_log_cache_key=prepared.logs["cache"]["cache_key"],
            vectors_sha256=hashlib.sha256(text_embeddings.tobytes(order="C")).hexdigest(),
        )
    inputs = temporal_inputs(
        prepared,
        config["context_windows"],
        availability_prior=prior,
        text_embeddings=text_embeddings,
        text_control=config.get("text_control", "actual"),
        availability_prior_scope=(
            "global"
            if config.get("availability_prior", {}).get("source") == "normal_fit_global"
            else "per_service"
        ),
        feature_mode=config.get("feature_mode", "base"),
    )
    indices = _fit_indices(prepared, inputs)
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    config["method"] = args.method
    payload = dict(config=config, prepared=_reference(args.prepared))
    if prior is not None:
        payload["availability_prior"] = prior
    if text_metadata is not None:
        payload["text_embedding_metadata"] = text_metadata
    binding = content_hash(payload)
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
    if "text_embedding" in config:
        provenance["text_embedding"] = config["text_embedding"]
    if args.text_cache_run is not None:
        provenance["parents"]["text_cache"] = _reference(args.text_cache_run)
    with _record(args.output, config, args, provenance) as recorder:
        fit_started = time.perf_counter()
        model_path = recorder.output / "run/model.pt"
        if text_embeddings is not None:
            np.save(recorder.output / "run/text_embeddings.npy", text_embeddings)
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
                    architecture_metadata=temporal_model_metadata(model),
                    **(dict(availability_prior=prior) if prior is not None else {}),
                    **(
                        dict(text_embedding=config["text_embedding"])
                        if "text_embedding" in config
                        else {}
                    ),
                ),
                model_path,
            )
        else:
            features = lag_features(inputs)
            if args.method == "service_lag_IF":
                services = inputs["identity"].canonical_service_id.astype(str).to_numpy()
                model = {
                    service: fit_iforest(
                        features[indices[services[indices] == service]],
                        config["isolation_forest"]["random_state"],
                    )
                    for service in sorted(set(services[indices]))
                }
            else:
                model = fit_iforest(features[indices], config["isolation_forest"]["random_state"])
            torch.save(
                dict(
                    estimator=model,
                    config=config,
                    fit_ids=sorted(prepared.lists.fit),
                    **(dict(availability_prior=prior) if prior is not None else {}),
                ),
                model_path,
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
                if args.method == "service_lag_IF":
                    frame["score"] = 0.0
                    for service, group in frame.groupby("canonical_service_id"):
                        if str(service) not in model:
                            raise ValueError("Missing service-specific lag IF model")
                        group_indices = [lookup[sample_id] for sample_id in group.sample_id]
                        frame.loc[group.index, "score"] = score_iforest(
                            model[str(service)], features[group_indices]
                        )
                else:
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
                **(
                    dict(
                        text_embeddings=recorder.output / "run/text_embeddings.npy",
                        text_embedding_metadata=text_metadata,
                        text_control_summary=inputs["text_control_summary"],
                    )
                    if text_embeddings is not None
                    else {}
                ),
                fit_ids=sorted(prepared.lists.fit),
                context_sample_ids=inputs["history_ids"],
                feature_order=inputs["feature_order"],
                **(dict(availability_prior=prior) if prior is not None else {}),
                fit_summary=dict(
                    method=args.method,
                    elapsed_seconds=fit_elapsed,
                    fit_windows=len(indices),
                    seed=config["training"]["seed"],
                    checkpoint="last",
                    **(
                        dict(architecture_metadata=temporal_model_metadata(model))
                        if args.method == "gru"
                        else {}
                    ),
                    **(
                        dict(text_embedding_elapsed_seconds=text_embedding_elapsed)
                        if text_metadata is not None
                        else {}
                    ),
                ),
            ),
        )
    print(json.dumps(check_run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
