"""Run the frozen eight-direction comparison with whole-Case Normal validation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import (
    LocalRuntime,
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
from src.service.modeling.data import load_inputs
from src.service.modeling.eight_directions import (
    DIRECTIONS,
    ExplorationPredictor,
    fit_prototypes,
    fused_score,
    fusion_scales,
    mechanism_components,
    prototype_distances,
    score_candidate,
    train_candidate,
)
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.temporal import temporal_inputs
from src.service.modeling.temporal_tuning import prepare_inner


def validate_config(config):
    expected = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/service/gru_eight_directions_v1.json"
        ).read_text()
    )
    if config != expected or config["directions"] != list(DIRECTIONS):
        raise ValueError("Exploration settings must match the frozen eight-direction protocol")


def restore_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ExplorationPredictor(checkpoint["model_direction"], checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def components(model, inputs, ids, direction, memory=None):
    base = score_candidate(model, inputs, ids)
    auxiliary = (
        prototype_distances(model, inputs, ids, memory)[:, None]
        if direction == "prototype"
        else mechanism_components(inputs, ids)
    )
    return base, np.column_stack([base.score.to_numpy(), auxiliary])


def run_candidate(args, config, direction, seed, tables, outer, inner, audit, inputs):
    output = args.output / f"{direction}-seed{seed}"
    if output.exists():
        check_run(output)
        manifest = json.loads((output / "manifest.json").read_text())
        stored = json.loads((output / "run/config.json").read_text())
        if (
            stored["direction"] != direction
            or stored["seed"] != seed
            or stored["protocol"] != config
        ):
            raise ValueError("Existing candidate configuration mismatch")
        if manifest["provenance"]["parents"]["prepared"] != _reference(args.prepared):
            raise ValueError("Existing candidate prepared parent mismatch")
        return
    run_config = dict(
        schema_version=config["schema_version"], protocol=config, direction=direction, seed=seed
    )
    parents = dict(prepared=_reference(args.prepared))
    postprocessing = direction in {"prototype", "mechanism"}
    baseline = args.output / f"baseline-seed{seed}"
    if postprocessing:
        check_run(baseline)
        parents["baseline"] = _reference(baseline)
    provenance = dict(
        stage="gru_eight_direction_experiment",
        exploratory=True,
        selection_scope="frozen_all_candidates_no_fault_selection",
        versions=outer.versions,
        lists_hash=outer.lists.lists_hash,
        parents=parents,
        source_sha256=args.source_sha256,
        context_policy="strict_past_same_case_service_split_15s_contiguous",
        priority=list(DIRECTIONS),
    )
    with _record(output, run_config, args, provenance) as recorder:
        models = {}
        histories = {}
        for scope, prepared in [("inner", inner), ("outer", outer)]:
            if postprocessing:
                model, checkpoint = restore_checkpoint(
                    baseline / f"run/{scope}-model.pt", config["training"]["device"]
                )
                if (
                    checkpoint["fit_ids"] != sorted(prepared.lists.fit)
                    or checkpoint["seed"] != seed
                ):
                    raise ValueError("Baseline seed or fit membership mismatch")
                history = []
            else:

                def progress(record, phase=scope):
                    step = record["epoch"] + (
                        config["pretraining_epochs"]
                        if record["phase"] == "prediction" and direction == "masked_pretraining"
                        else 0
                    )
                    if phase == "outer":
                        step += 100
                    recorder.log_epoch(
                        {f"{phase}_{record['phase']}_fit_loss": record["mean_loss"]}, step=step
                    )

                model, history = train_candidate(
                    prepared, inputs[scope], direction, config, seed, progress
                )
            models[scope] = model
            histories[scope] = history
        memory = {}
        scales = None
        validation_ids = audit["validation_ids"]
        if postprocessing:
            if direction == "prototype":
                for scope, prepared in [("inner", inner), ("outer", outer)]:
                    memory[scope] = fit_prototypes(
                        models[scope], prepared, inputs[scope], config, seed
                    )
            _, reference = components(
                models["inner"], inputs["inner"], validation_ids, direction, memory.get("inner")
            )
            scales = fusion_scales(reference, config["fusion_scale_floor"])

        def score(scope, ids):
            if not postprocessing:
                return score_candidate(models[scope], inputs[scope], ids)
            frame, values = components(
                models[scope], inputs[scope], ids, direction, memory.get(scope)
            )
            frame["original_gru_score"] = frame.score
            frame["score"] = fused_score(values, scales, config["fusion_auxiliary_weight"])
            for index in range(1, values.shape[1]):
                frame[f"auxiliary_component_{index}"] = values[:, index]
            return frame

        validation = score("inner", validation_ids)
        scored = {}
        for scope in ["calibration", "test"]:
            frame = score("outer", sorted(score_ids(outer.lists, scope)))
            frame["model_version"] = config["schema_version"] + ":" + direction
            frame["config_sha256"] = content_hash(run_config)
            frame["lists_hash"] = outer.lists.lists_hash
            for name, version in outer.versions.items():
                frame[name] = version
            scored[scope] = frame
        calibration = annotate_calibration(
            scored["calibration"], outer, set(outer.lists.calibration)
        )
        thresholds = calibrate(calibration, set(outer.lists.calibration), config["quantiles"])
        report = evaluate_run(tables, outer.lists, scored["test"], thresholds, score_scope="test")
        recorder.log_epoch(
            dict(
                main_auroc=report["main"]["auroc"],
                main_ap=report["main"]["ap"],
                normal_validation_value_loss=float(validation.value_loss.mean()),
            ),
            step=250,
        )
        report_path = recorder.output / "evaluation/report.json"
        _json(report_path, report)
        artifacts = dict(
            evaluation_report=report_path,
            thresholds=thresholds,
            inner_split_and_preprocessing=audit,
            training_history=histories,
            fit_ids=sorted(outer.lists.fit),
            context_sample_ids=inputs["outer"]["history_ids"],
            inner_context_sample_ids=inputs["inner"]["history_ids"],
            feature_order=inputs["outer"]["feature_order"],
            fusion_reference=dict(
                scales=scales,
                source_ids=validation_ids,
                source="whole_heldout_inner_normal_case",
                weight=config["fusion_auxiliary_weight"],
                diagnostic_uses_reference_case=postprocessing,
            ),
            prototype_memory=memory,
            normal_validation_summary=dict(
                windows=len(validation),
                value_loss=float(validation.value_loss.mean()),
                availability_loss=float(validation.availability_loss.mean()),
                score_mean=float(validation.score.mean()),
                directly_comparable_score=direction
                not in {"heteroscedastic", "prototype", "mechanism"},
            ),
            fit_summary=dict(
                direction=direction,
                seed=seed,
                parameter_count=sum(p.numel() for p in models["outer"].parameters()),
                fit_windows=len(outer.lists.fit),
                checkpoint="last",
            ),
        )
        for scope, model in models.items():
            path = recorder.output / f"run/{scope}-model.pt"
            prepared = inner if scope == "inner" else outer
            torch.save(
                dict(
                    model_state=model.state_dict(),
                    config=config,
                    seed=seed,
                    direction=direction,
                    model_direction=model.direction,
                    fit_ids=sorted(prepared.lists.fit),
                    availability_prior=inputs[scope]["availability_prior"],
                    prototype_memory=memory.get(scope),
                    fusion_scales=scales,
                ),
                path,
            )
            artifacts[f"{scope}_checkpoint"] = path
        for name, frame in [
            ("scores", scored["test"]),
            ("calibration_scores", calibration),
            ("normal_validation_scores", validation),
        ]:
            path = recorder.output / f"run/{name}.parquet"
            frame.to_parquet(path, index=False)
            artifacts[name] = path
        _publish(recorder, artifacts)
    checked = check_run(output)
    print(
        json.dumps(
            dict(
                direction=direction,
                seed=seed,
                checked=checked,
                auroc=report["main"]["auroc"],
                ap=report["main"]["ap"],
            )
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "prepared", "preparation", "label-bundle", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    from src.service.preparation import file_digest

    paths = [
        "src/service/modeling/eight_directions.py",
        "src/service/modeling/temporal.py",
        "scripts/service_eight_directions.py",
        "configs/service/gru_eight_directions_v1.json",
    ]
    args.source_sha256 = {path: file_digest(Path(path)) for path in paths}
    args.output.mkdir(parents=True, exist_ok=True)
    suite = dict(
        protocol=config, prepared=_reference(args.prepared), source_sha256=args.source_sha256
    )
    suite_path = args.output / "suite.json"
    if suite_path.exists():
        if json.loads(suite_path.read_text()) != suite:
            raise ValueError("Suite source/configuration changed; use a new output directory")
    else:
        _json(suite_path, suite)
    outer_tables, outer, _ = read_prepared(args.prepared)
    tables = load_inputs(args.preparation, args.label_bundle, outer_tables.config)
    if tables.hashes != outer_tables.hashes or tables.versions != outer_tables.versions:
        raise ValueError("Inner preprocessing source binding mismatch")
    inner, audit = prepare_inner(tables, outer, LocalRuntime().tokenizer(tables.config))
    inputs = {
        scope: temporal_inputs(
            prepared,
            config["context_windows"],
            availability_prior=fit_availability(prepared, prepared.lists.fit),
        )
        for scope, prepared in [("inner", inner), ("outer", outer)]
    }
    for direction in ["baseline", *DIRECTIONS]:
        for seed in config["seeds"]:
            run_candidate(args, config, direction, seed, outer_tables, outer, inner, audit, inputs)
    _json(
        args.output / "completed.json",
        dict(status="success", directions=list(DIRECTIONS), seeds=config["seeds"]),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
