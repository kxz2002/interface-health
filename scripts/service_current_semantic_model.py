"""Score current-window frozen BERT semantics and optionally fuse with temporal GRU scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import (
    _json,
    _publish,
    _record,
    _reference,
    check_run,
    read_prepared,
    score_ids,
)
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run


def _config(path: Path) -> dict:
    value = json.loads(path.read_text())
    required = {
        "schema_version",
        "context_windows",
        "window_ms",
        "semantic_score",
        "fusion",
        "semantic_weight",
        "temporal_weight",
        "text_embedding",
        "training",
        "quantiles",
    }
    if set(value) != required or value["schema_version"] != "service-temporal-current-semantic-v1":
        raise ValueError("Unsupported current semantic configuration")
    if value["context_windows"] != 4 or value["window_ms"] != 15000:
        raise ValueError("Current semantic branch requires the fixed temporal protocol")
    if (
        value["semantic_score"] != "per_service_normal_fit_cosine_distance"
        or value["fusion"] != "normal_fit_robust_z_sum"
    ):
        raise ValueError("Unsupported current semantic scoring protocol")
    if value["text_embedding"]["revision"] != "387825ce42dbb39b87911cdf8e383ee3b25184f8":
        raise ValueError("Unexpected frozen text revision")
    if value["quantiles"] != [0.95, 0.975, 0.99, 0.995]:
        raise ValueError("Unexpected calibration quantiles")
    return value


def _load_embeddings(run: Path, prepared_path: Path, prepared, config: dict) -> np.ndarray:
    check_run(run)
    root = run.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["provenance"]["parents"].get("prepared") != _reference(prepared_path):
        raise ValueError("Embedding cache prepared parent mismatch")
    if (
        json.loads((root / "run/config.json").read_text()).get("text_embedding")
        != config["text_embedding"]
    ):
        raise ValueError("Embedding cache configuration mismatch")
    metadata = json.loads((root / "run/text_embedding_metadata.json").read_text())
    vectors = np.load(root / "run/text_embeddings.npy", allow_pickle=False)
    if (
        metadata["sample_ids"] != sorted(prepared.sample_ids)
        or vectors.shape != (len(metadata["sample_ids"]), 256)
        or vectors.dtype != np.float32
    ):
        raise ValueError("Embedding cache identity or shape mismatch")
    if hashlib.sha256(vectors.tobytes(order="C")).hexdigest() != metadata["vectors_sha256"]:
        raise ValueError("Embedding cache hash mismatch")
    return vectors


def _cosine_distance(
    vectors: np.ndarray, references: dict[str, np.ndarray], services: np.ndarray
) -> np.ndarray:
    output = np.empty(len(vectors), dtype=np.float64)
    for service in sorted(set(services)):
        selected = services == service
        reference = references.get(str(service))
        if reference is None:
            raise ValueError(f"Missing Normal semantic reference for service {service}")
        numerator = vectors[selected] @ reference
        denominator = np.linalg.norm(vectors[selected], axis=1) * np.linalg.norm(reference)
        output[selected] = 1.0 - numerator / np.maximum(denominator, 1e-12)
    return output


def _robust_z(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    median = float(np.median(reference))
    scale = float(np.median(np.abs(reference - median))) * 1.4826
    return (values - median) / max(scale, 1e-6)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--text-cache-run", type=Path, required=True)
    parser.add_argument("--temporal-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42], required=True)
    parser.add_argument("--mode", choices=["semantic", "fusion"], required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = _config(args.config)
    tables, prepared, _ = read_prepared(args.prepared)
    embeddings = _load_embeddings(args.text_cache_run, args.prepared, prepared, config)
    ids = sorted(prepared.sample_ids)
    identity = prepared.identity.set_index("sample_id").loc[ids]
    services = identity.canonical_service_id.astype(str).to_numpy()
    lookup = {sample_id: index for index, sample_id in enumerate(ids)}
    fit_indices = np.asarray([lookup[sample_id] for sample_id in prepared.lists.fit])
    references = {}
    for service in sorted(set(services[fit_indices])):
        service_vectors = embeddings[fit_indices][services[fit_indices] == service]
        reference = service_vectors.mean(axis=0)
        references[service] = reference / max(np.linalg.norm(reference), 1e-12)
    semantic = _cosine_distance(embeddings, references, services)
    temporal_run = args.temporal_run.resolve()
    check_run(temporal_run)
    temporal_scores = pd.read_parquet(temporal_run / "run/scores.parquet").set_index("sample_id")
    temporal_calibration = pd.read_parquet(
        temporal_run / "run/calibration_scores.parquet"
    ).set_index("sample_id")
    fit_temporal = temporal_calibration.score.to_numpy(float)
    fit_semantic = semantic[fit_indices]
    fit_map = dict(zip(prepared.lists.fit, fit_semantic))
    temporal_map = dict(zip(prepared.lists.fit, fit_temporal))
    rows = []
    for scope in ["calibration", "test"]:
        selected_ids = sorted(score_ids(prepared.lists, scope))
        selected = np.asarray([lookup[sample_id] for sample_id in selected_ids])
        semantic_score = semantic[selected]
        if args.mode == "semantic":
            score = semantic_score
        else:
            temporal_source = temporal_calibration if scope == "calibration" else temporal_scores
            temporal = temporal_source.reindex(selected_ids).score.to_numpy(float)
            score = config["semantic_weight"] * _robust_z(semantic_score, fit_semantic) + config[
                "temporal_weight"
            ] * _robust_z(temporal, fit_temporal)
        frame = identity.loc[selected_ids].reset_index()
        frame["score"] = score
        frame["semantic_score"] = semantic_score
        frame["value_loss"] = 0.0
        frame["availability_loss"] = 0.0
        rows.append(frame)
    scores = pd.concat(rows, ignore_index=True)
    binding = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    calibration_scores = scores[scores.sample_id.isin(prepared.lists.calibration)].copy()
    evaluation_scores = scores[scores.sample_id.isin(score_ids(prepared.lists, "test"))].copy()
    for frame in [scores, calibration_scores, evaluation_scores]:
        frame["model_version"] = config["schema_version"] + "-" + args.mode
        frame["config_sha256"] = binding
        frame["lists_hash"] = prepared.lists.lists_hash
        frame["label"] = (
            prepared.labels.set_index("sample_id").loc[frame.sample_id].label.to_numpy()
        )
        frame["label_source"] = (
            prepared.labels.set_index("sample_id").loc[frame.sample_id].label_source.to_numpy()
        )
        frame["split"] = (
            prepared.splits.set_index("sample_id").loc[frame.sample_id].split.to_numpy()
        )
        for name, version in prepared.versions.items():
            frame[name] = version
    calibration_scores = annotate_calibration(
        calibration_scores, prepared, set(prepared.lists.calibration)
    )
    thresholds = calibrate(calibration_scores, set(prepared.lists.calibration), config["quantiles"])
    evaluation_scores = scores[scores.sample_id.isin(score_ids(prepared.lists, "test"))].copy()
    evaluation = evaluate_run(
        tables, prepared.lists, evaluation_scores, thresholds, score_scope="test"
    )
    provenance = dict(
        stage="current_window_semantic_experiment",
        method=args.mode,
        exploratory=True,
        versions=prepared.versions,
        parents=dict(
            prepared=_reference(args.prepared),
            text_cache=_reference(args.text_cache_run),
            temporal=_reference(args.temporal_run),
        ),
        lists_hash=prepared.lists.lists_hash,
        context_policy="current_window_frozen_bert_per_service_normal_fit_reference",
    )
    with _record(args.output, config, args, provenance) as recorder:
        _json(
            recorder.output / "run/semantic_reference_summary.json",
            {
                service: {
                    "fit_count": int((services[fit_indices] == service).sum()),
                    "reference_sha256": hashlib.sha256(references[service].tobytes()).hexdigest(),
                }
                for service in references
            },
        )
        scores.to_parquet(recorder.output / "run/scores.parquet", index=False)
        calibration_scores.to_parquet(
            recorder.output / "run/calibration_scores.parquet", index=False
        )
        _json(recorder.output / "evaluation/report.json", evaluation)
        _publish(
            recorder,
            dict(
                scores=recorder.output / "run/scores.parquet",
                calibration_scores=recorder.output / "run/calibration_scores.parquet",
                evaluation_report=recorder.output / "evaluation/report.json",
                semantic_reference_summary=recorder.output / "run/semantic_reference_summary.json",
            ),
        )
    print(
        json.dumps(
            {
                "run_id": args.output.name,
                "mode": args.mode,
                "auroc": evaluation["main"]["auroc"],
                "ap": evaluation["main"]["ap"],
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
