"""Explicit exploratory fusion of immutable value and availability score runs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import (
    _json,
    _manifest,
    _publish,
    _record,
    _reference,
    check_run,
    read_prepared,
    score_ids,
)
from src.service.modeling.config import content_hash
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.tail_fusion import fit_tail_fusion, score_tail_fusion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tracking-uri", required=True)
    for channel in ["value", "availability"]:
        for scope in ["fit", "calibration", "test"]:
            parser.add_argument(f"--{channel}-{scope}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    tables, prepared, _ = read_prepared(args.prepared)
    prepared_reference = _reference(args.prepared)
    frames = {scope: {} for scope in ["fit", "calibration", "test"]}
    parents = dict(prepared=prepared_reference)
    for channel in ["value", "availability"]:
        model_reference = None
        for scope in frames:
            path = getattr(args, f"{channel}_{scope}")
            manifest = _manifest(path)
            provenance = manifest["provenance"]
            if (
                manifest["stage"] != "score"
                or provenance["score_scope"] != scope
                or provenance["parents"]["prepared"] != prepared_reference
                or manifest["training_versions"] != prepared.versions
            ):
                raise ValueError("Incompatible fusion score run scope or input binding")
            if model_reference is None:
                model_reference = provenance["parents"]["model"]
            if model_reference != provenance["parents"]["model"]:
                raise ValueError("Fusion channel uses different models across scopes")
            if (channel == "availability") != (provenance["method"] == "availability"):
                raise ValueError("Incorrect fusion channel method")
            frame = pd.read_parquet(path / "run/scores.parquet")
            if set(frame.sample_id) != score_ids(prepared.lists, scope):
                raise ValueError("Fusion score run differs from frozen sample scope")
            frames[scope][channel] = frame
            parents[f"{channel}_{scope}"] = _reference(path)
    state = fit_tail_fusion(prepared, frames["fit"], prepared.lists.fit)
    config = dict(
        schema_version=state["schema_version"],
        formula=state["formula"],
        weights=state["weights"],
        reference_scope="normal_fit_per_service",
        quantiles=tables.config["calibration"]["sensitivity_quantiles"],
        exploratory=True,
    )
    binding = content_hash(dict(config=config, state=state, parents=parents))
    provenance = dict(
        stage="score_fusion",
        method="value_availability_tail_fusion",
        versions=prepared.versions,
        parents=parents,
        lists_hash=prepared.lists.lists_hash,
        fusion_sha256=binding,
    )
    with _record(args.output, config, args, provenance) as recorder:
        scored = {}
        for scope in ["calibration", "test"]:
            frame = score_tail_fusion(
                state, prepared, frames[scope], score_ids(prepared.lists, scope)
            )
            frame["model_version"] = state["schema_version"]
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
            {"main_auroc": report["main"]["auroc"], "main_ap": report["main"]["ap"]}, step=0
        )
        report_path = recorder.output / "evaluation/report.json"
        _json(report_path, report)
        score_path = recorder.output / "run/scores.parquet"
        scored["test"].to_parquet(score_path, index=False)
        calibration_path = recorder.output / "run/calibration_scores.parquet"
        calibration.to_parquet(calibration_path, index=False)
        _publish(
            recorder,
            dict(
                fusion_state=state,
                evaluation_report=report_path,
                thresholds=thresholds,
                scores=score_path,
                calibration_scores=calibration_path,
            ),
        )
    print(json.dumps(check_run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
