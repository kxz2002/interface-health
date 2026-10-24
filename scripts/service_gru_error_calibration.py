"""Fit held-out Normal residual scales and rescore an immutable original GRU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_gru_candidates import artifact
from scripts.service_model import (
    LocalRuntime,
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
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.residual_scaling import fit_residual_scales, scaled_value_loss
from src.service.modeling.temporal import TemporalPredictor, score_temporal, temporal_inputs
from src.service.modeling.temporal_tuning import prepare_inner


def squared_residuals(model, inputs, ids):
    ids = sorted(ids)
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    indices = [lookup[sample_id] for sample_id in ids]
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            selected = indices[start : start + 256]
            prediction, _ = model(
                torch.as_tensor(inputs["context"][selected], device=device),
                torch.as_tensor(inputs["prior_logits"][selected], device=device),
            )
            chunks.append(prediction.cpu().numpy())
    prediction = np.concatenate(chunks)
    observed = inputs["observed"][indices]
    errors = np.where(observed, (prediction - inputs["values"][indices]) ** 2, 0)
    services = inputs["identity"].set_index("sample_id").loc[ids, "canonical_service_id"].to_numpy()
    return errors, observed, services


def restore(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    settings = checkpoint.get("settings", config["training"])
    if config["schema_version"] != "service-temporal-prior-v1" or config["context_windows"] != 4:
        raise ValueError("Error calibration requires original four-window prior GRU")
    model = TemporalPredictor(hidden_dim=settings["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "prepared",
        "preparation",
        "label-bundle",
        "baseline-run",
        "normal-validation-run",
        "config",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42], required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.config.read_text())
    if config != dict(
        schema_version="service-gru-error-calibration-v1",
        shrinkage_count=32,
        relative_floor=0.1,
        absolute_floor=1e-4,
        scale_source="latest_outer_fit_normal_case_heldout",
        scale_transfer="asinh_normalized_feature_coordinates",
        score="scaled_masked_value_mse_plus_original_modality_bce",
        device="cuda",
        quantiles=[0.95, 0.975, 0.99, 0.995],
    ):
        raise ValueError("Unsupported frozen error calibration protocol")
    config = {**config, "seed": args.seed}
    check_run(args.baseline_run)
    check_run(args.normal_validation_run)
    outer_tables, outer, _ = read_prepared(args.prepared)
    baseline_manifest = json.loads((args.baseline_run / "manifest.json").read_text())
    normal_manifest = json.loads((args.normal_validation_run / "manifest.json").read_text())
    if baseline_manifest["provenance"]["parents"]["prepared"] != _reference(args.prepared):
        raise ValueError("Baseline prepared binding mismatch")
    tables = load_inputs(args.preparation, args.label_bundle, outer_tables.config)
    if tables.hashes != outer_tables.hashes or tables.versions != outer_tables.versions:
        raise ValueError("Normal input binding mismatch")
    provenance = dict(
        stage="gru_error_calibration",
        exploratory=True,
        selection_scope="outer_fit_normal_only",
        versions=outer.versions,
        parents=dict(
            prepared=_reference(args.prepared),
            baseline=_reference(args.baseline_run),
            normal_validation=_reference(args.normal_validation_run),
        ),
    )
    with _record(args.output, config, args, provenance) as recorder:
        inner, inner_audit = prepare_inner(tables, outer, LocalRuntime().tokenizer(tables.config))
        stored_audit = json.loads(
            artifact(
                args.normal_validation_run, normal_manifest, "inner_split_and_preprocessing"
            ).read_text()
        )
        for key in ["actual_fit_ids", "validation_ids", "preprocessor_hash"]:
            if inner_audit[key] != stored_audit[key]:
                raise ValueError(f"Held-out preprocessing differs: {key}")
        ids = inner_audit["validation_ids"]
        labels = inner.labels.set_index("sample_id").loc[ids]
        if not labels.label.eq(0).all() or not labels.label_source.eq("normal_run_reference").all():
            raise ValueError("Scale source must be independent known Normal")
        inner_model, inner_checkpoint = restore(
            artifact(
                args.normal_validation_run, normal_manifest, f"baseline_seed{args.seed}_checkpoint"
            ),
            config["device"],
        )
        if inner_checkpoint["settings"]["seed"] != args.seed or inner_checkpoint[
            "fit_ids"
        ] != sorted(inner.lists.fit):
            raise ValueError("Inner checkpoint seed or fit membership mismatch")
        prior = fit_availability(inner, inner.lists.fit)
        inner_inputs = temporal_inputs(inner, 4, availability_prior=prior)
        errors, observed, services = squared_residuals(inner_model, inner_inputs, ids)
        state = fit_residual_scales(errors, observed, services, sorted(ids), set(ids))
        state["preprocessor_hash"] = inner_audit["preprocessor_hash"]
        model, checkpoint = restore(
            artifact(args.baseline_run, baseline_manifest, "model"), config["device"]
        )
        if checkpoint["config"]["training"]["seed"] != args.seed or checkpoint["fit_ids"] != sorted(
            outer.lists.fit
        ):
            raise ValueError("Outer checkpoint seed or fit membership mismatch")
        inputs = temporal_inputs(outer, 4, availability_prior=checkpoint["availability_prior"])
        scored = {}
        binding = content_hash(dict(config=config, state=state, parents=provenance["parents"]))
        for scope in ["calibration", "test"]:
            selected_ids = sorted(score_ids(outer.lists, scope))
            frame = score_temporal(model, inputs, selected_ids)
            errors, observed, services = squared_residuals(model, inputs, selected_ids)
            frame["original_score"] = frame.score
            frame["original_value_loss"] = frame.value_loss
            frame["value_loss"] = scaled_value_loss(state, errors, observed, services)
            frame["score"] = frame.value_loss + frame.availability_loss
            frame["model_version"] = config["schema_version"]
            frame["config_sha256"] = binding
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
            dict(main_auroc=report["main"]["auroc"], main_ap=report["main"]["ap"]), step=1
        )
        score_path = recorder.output / "run/scores.parquet"
        calibration_path = recorder.output / "run/calibration_scores.parquet"
        scored["test"].to_parquet(score_path, index=False)
        calibration.to_parquet(calibration_path, index=False)
        _publish(
            recorder,
            dict(
                residual_scale_state=state,
                inner_audit=inner_audit,
                evaluation_report=report,
                thresholds=thresholds,
                scores=score_path,
                calibration_scores=calibration_path,
                fit_ids=sorted(outer.lists.fit),
            ),
        )
    print(json.dumps(check_run(args.output)))


if __name__ == "__main__":
    main()
