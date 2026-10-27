"""Train and evaluate the pinned external PatchAD model on prepared service data."""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

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
from src.service.modeling.config import content_hash
from src.service.modeling.evaluation import QUANTILES, annotate_calibration, calibrate, evaluate_run
from src.service.modeling.neural_baseline_data import closed_window_sequences
from src.service.modeling.patchad_adapter import score_patchad, train_patchad


def upstream_factory(path, config):
    root = Path(path).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if revision != config["upstream_commit"] or dirty:
        raise ValueError("PatchAD source must be the pinned clean upstream checkout")
    sys.path.insert(0, str(root))
    from patchad_model import models

    if not Path(models.__file__).resolve().is_relative_to(root):
        raise ValueError("Imported PatchAD model is not from the pinned checkout")
    return lambda: models.PatchMLPAD(
        win_size=config["sequence_length"],
        e_layer=config["e_layers"],
        patch_sizes=config["patch_sizes"],
        channel=27,
        d_model=config["d_model"],
        dropout=config["dropout"],
        activation="relu",
        norm="n",
    )


def validate_config(config):
    expected = {
        "schema_version",
        "upstream_commit",
        "sequence_length",
        "patch_sizes",
        "d_model",
        "e_layers",
        "dropout",
        "training",
        "quantiles",
    }
    if set(config) != expected or config["schema_version"] != "service-patchad-contrast-v1":
        raise ValueError("Invalid PatchAD configuration")
    length = config["sequence_length"]
    patches = config["patch_sizes"]
    training = config["training"]
    if (
        type(length) is not int
        or length < 2
        or not isinstance(patches, list)
        or not patches
        or any(type(size) is not int or size < 2 or length % size for size in patches)
        or type(config["d_model"]) is not int
        or config["d_model"] < 2
        or type(config["e_layers"]) is not int
        or config["e_layers"] < 1
        or config["quantiles"] != list(QUANTILES)
        or not isinstance(training, dict)
        or set(training) != {"seed", "device", "epochs", "batch_size", "lr", "patch_weight"}
        or type(training["seed"]) is not int
        or training["seed"] not in {1, 2, 42}
        or training["device"] not in {"cpu", "cuda"}
        or any(
            type(training[name]) is not int or training[name] < 1
            for name in ["epochs", "batch_size"]
        )
        or any(
            type(value) not in {int, float} or not math.isfinite(value)
            for value in [training["lr"], training["patch_weight"], config["dropout"]]
        )
        or training["lr"] <= 0
        or not 0 <= training["patch_weight"] <= 1
        or not 0 <= config["dropout"] < 1
    ):
        raise ValueError("Unsupported PatchAD configuration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    validate_config(config)
    factory = upstream_factory(args.upstream, config)
    tables, prepared, _ = read_prepared(args.prepared)
    inputs = closed_window_sequences(prepared, config["sequence_length"])
    parent = _reference(args.prepared)
    binding = content_hash(dict(config=config, prepared=parent))
    provenance = dict(
        stage="temporal_experiment",
        method="patchad_contrast",
        exploratory=True,
        versions=prepared.versions,
        parents=dict(prepared=parent),
        lists_hash=prepared.lists.lists_hash,
        model_sha256=binding,
        feature_order=inputs["feature_order"],
        upstream_commit=config["upstream_commit"],
        context_policy="closed_same_case_service_split_15s_contiguous",
    )
    with _record(args.output, config, args, provenance) as recorder:
        model, evidence = train_patchad(prepared, inputs, config["training"], factory)
        model_path = recorder.output / "run/model.pt"
        torch.save(
            dict(model_state=model.state_dict(), config=config, fit_ids=evidence["fit_ids"]),
            model_path,
        )
        scored = {}
        for scope in ["calibration", "test"]:
            ids = score_ids(prepared.lists, scope)
            frame = score_patchad(model, inputs, ids, config["training"]["batch_size"])
            frame["model_version"] = config["schema_version"]
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
                training_history=evidence["history"],
                fit_ids=evidence["fit_ids"],
                excluded_incomplete_fit_ids=evidence["excluded_incomplete_fit_ids"],
                context_sample_ids=inputs["history_ids"],
                feature_order=inputs["feature_order"],
                evaluation_report=report_path,
                thresholds=thresholds,
                scores=scores_path,
                calibration_scores=calibration_path,
            ),
        )
    print(json.dumps(check_run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
