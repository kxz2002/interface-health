"""Compare frozen GRU candidates using only a whole held-out Normal fit Case."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import LocalRuntime, _publish, _record, _reference, read_prepared
from scripts.service_temporal_model import validate_temporal_config
from src.service.modeling.availability import fit_availability
from src.service.modeling.data import load_inputs
from src.service.modeling.temporal import (
    score_temporal,
    temporal_inputs,
    temporal_model_metadata,
    train_temporal,
)
from src.service.modeling.temporal_tuning import prepare_inner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "preparation", "label-bundle", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--residual-only", action="store_true")
    parser.add_argument("--architecture", choices=["service_bias", "short_long"])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {
        "baseline": "configs/service/tt_temporal_prior_gpu_b_v1.json",
        "context8": "configs/service/tt_temporal_prior_gru_context8_gpu_b_v1.json",
        "hidden64": "configs/service/tt_temporal_prior_gru_hidden64_gpu_b_v1.json",
    }
    if args.residual_only:
        paths = {"residual": "configs/service/tt_temporal_prior_gru_residual_gpu_b_v1.json"}
    if args.architecture:
        if args.residual_only:
            raise ValueError("Choose one independent architecture")
        paths = {
            args.architecture: f"configs/service/tt_temporal_prior_gru_{args.architecture}_gpu_b_v1.json"
        }
    configs = {name: json.loads(Path(path).read_text()) for name, path in paths.items()}
    for config in configs.values():
        validate_temporal_config(config)
    outer_tables, outer, _ = read_prepared(args.prepared)
    tables = load_inputs(args.preparation, args.label_bundle, outer_tables.config)
    if tables.hashes != outer_tables.hashes or tables.versions != outer_tables.versions:
        raise ValueError("Normal validation input binding mismatch")
    provenance = dict(
        stage="gru_whole_normal_validation",
        parents=dict(prepared=_reference(args.prepared)),
        selection_scope="latest_outer_fit_normal_case_only",
        exploratory=True,
    )
    with _record(
        args.output, dict(candidates=configs, seeds=[1, 2, 42]), args, provenance
    ) as recorder:
        inner, audit = prepare_inner(tables, outer, LocalRuntime().tokenizer(tables.config))
        prior = fit_availability(inner, inner.lists.fit)
        results = []
        artifacts = dict(inner_split_and_preprocessing=audit)
        for name, config in configs.items():
            inputs = temporal_inputs(inner, config["context_windows"], availability_prior=prior)
            for seed in [1, 2, 42]:
                settings = copy.deepcopy(config["training"])
                settings["seed"] = seed
                model, history = train_temporal(
                    inner,
                    inputs,
                    settings,
                    progress=lambda record: recorder.log_epoch(
                        {f"{name}_seed{seed}_fit_loss": record["mean_loss"]},
                        step=record["epoch"],
                    ),
                )
                scores = score_temporal(model, inputs, audit["validation_ids"])
                checkpoint = recorder.output / f"run/{name}-seed{seed}.pt"
                torch.save(
                    dict(
                        model_state=model.state_dict(),
                        config=config,
                        settings=settings,
                        fit_ids=sorted(inner.lists.fit),
                        architecture_metadata=temporal_model_metadata(model),
                        availability_prior=prior,
                    ),
                    checkpoint,
                )
                score_path = recorder.output / f"run/{name}-seed{seed}-scores.parquet"
                scores.to_parquet(score_path, index=False)
                artifacts[f"{name}_seed{seed}_checkpoint"] = checkpoint
                artifacts[f"{name}_seed{seed}_scores"] = score_path
                row = dict(
                    candidate=name,
                    seed=seed,
                    validation_windows=len(scores),
                    validation_loss=float(scores.score.mean()),
                    value_loss=float(scores.value_loss.mean()),
                    availability_loss=float(scores.availability_loss.mean()),
                    architecture_metadata=temporal_model_metadata(model),
                    unknown_service_windows=(
                        int((~scores.service_conditioning_known).sum())
                        if "service_conditioning_known" in scores
                        else 0
                    ),
                    training_history=history,
                )
                results.append(row)
                print(
                    json.dumps(
                        {key: value for key, value in row.items() if key != "training_history"}
                    ),
                    flush=True,
                )
        artifacts["normal_validation_results"] = results
        _publish(recorder, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
