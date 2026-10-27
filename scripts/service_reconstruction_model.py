"""Run pinned multivariate neural reconstruction models on prepared services."""

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
from scripts.service_revaluate_temporal import accepted_bundle, rebind
from src.service.modeling.config import content_hash
from src.service.modeling.evaluation import QUANTILES, annotate_calibration, calibrate, evaluate_run
from src.service.modeling.neural_baseline_data import closed_window_sequences
from src.service.modeling.reconstruction_baseline import score_reconstruction, train_reconstruction
from src.service.preparation import file_digest


def validate_config(config):
    keys = {
        "schema_version",
        "model",
        "upstream_commit",
        "sequence_length",
        "architecture",
        "training",
        "quantiles",
    }
    if (
        not isinstance(config, dict)
        or set(config) != keys
        or config["schema_version"] != "service-reconstruction-v1"
    ):
        raise ValueError("Invalid reconstruction configuration")
    model = config["model"]
    architecture = config["architecture"]
    training = config["training"]
    architecture_keys = {
        "KANAD": {"d_model"},
        "TimesNet": {"d_model", "d_ff", "e_layers", "top_k", "num_kernels", "dropout"},
    }
    if (
        model not in architecture_keys
        or not isinstance(architecture, dict)
        or set(architecture) != architecture_keys[model]
        or type(config["sequence_length"]) is not int
        or config["sequence_length"] < 2
        or any(
            type(value) is not int or value < 1
            for key, value in architecture.items()
            if key != "dropout"
        )
        or (
            model == "TimesNet"
            and (
                architecture["top_k"] >= config["sequence_length"] // 2 + 1
                or type(architecture["dropout"]) not in {int, float}
                or not 0 <= architecture["dropout"] < 1
            )
        )
        or not isinstance(training, dict)
        or set(training) != {"seed", "device", "epochs", "batch_size", "lr"}
        or type(training["seed"]) is not int
        or training["seed"] not in {1, 2, 42}
        or training["device"] not in {"cpu", "cuda"}
        or any(
            type(training[name]) is not int or training[name] < 1
            for name in ["epochs", "batch_size"]
        )
        or type(training["lr"]) not in {int, float}
        or not math.isfinite(training["lr"])
        or training["lr"] <= 0
        or config["quantiles"] != list(QUANTILES)
    ):
        raise ValueError("Unsupported reconstruction configuration")


def upstream_factory(path, config):
    root = Path(path).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    if revision != config["upstream_commit"] or dirty:
        raise ValueError("Source must be the pinned clean Time-Series-Library checkout")
    sys.path.insert(0, str(root))
    sys.dont_write_bytecode = True
    module = importlib.import_module("models." + config["model"])
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ValueError("Model is not from the pinned checkout")
    parameters = SimpleNamespace(
        task_name="anomaly_detection",
        seq_len=config["sequence_length"],
        label_len=0,
        pred_len=0,
        enc_in=27,
        c_out=27,
        embed="timeF",
        freq="h",
        **config["architecture"],
    )
    return lambda: module.Model(parameters)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42])
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args(argv)
    if (args.bundle is None) != (args.acceptance is None):
        parser.error("--bundle and --acceptance must be provided together")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    validate_config(config)
    factory = upstream_factory(args.upstream, config)
    tables, prepared, _ = read_prepared(args.prepared)
    lists = prepared.lists
    if args.bundle:
        bundle, bundle_manifest = accepted_bundle(args.bundle, args.acceptance)
        tables, lists = rebind(tables, prepared, bundle)
    inputs = closed_window_sequences(prepared, config["sequence_length"])
    parent = _reference(args.prepared)
    binding = content_hash(dict(config=config, prepared=parent))
    provenance = dict(
        stage="temporal_experiment",
        method=config["model"],
        exploratory=True,
        versions=prepared.versions,
        parents=dict(prepared=parent),
        lists_hash=prepared.lists.lists_hash,
        model_sha256=binding,
        feature_order=inputs["feature_order"],
        upstream_commit=config["upstream_commit"],
        context_policy="closed_same_case_service_split_15s_contiguous",
    )
    if args.bundle:
        provenance.update(
            evaluation_bundle_sha256=file_digest(bundle_manifest),
            acceptance_sha256=file_digest(args.acceptance),
            evaluation_versions=tables.versions,
        )
    with _record(args.output, config, args, provenance) as recorder:
        model, evidence = train_reconstruction(prepared, inputs, config["training"], factory)
        model_path = recorder.output / "run/model.pt"
        torch.save(
            dict(model_state=model.state_dict(), config=config, fit_ids=evidence["fit_ids"]),
            model_path,
        )
        scored = {}
        for scope in ["calibration", "test"]:
            ids = score_ids(lists, scope)
            score_batch = 1 if config["model"] == "TimesNet" else config["training"]["batch_size"]
            frame = score_reconstruction(model, inputs, ids, score_batch)
            frame["model_version"] = config["schema_version"] + "-" + config["model"]
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
            tables,
            lists,
            scored["test"],
            thresholds,
            score_scope="test",
            source_binding=(
                {"versions": prepared.versions, "lists_hash": prepared.lists.lists_hash}
                if args.bundle
                else None
            ),
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
