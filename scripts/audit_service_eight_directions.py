"""Restore all frozen candidates and verify their scores, lineage and MLflow history."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_gru_architecture import check_contexts
from scripts.service_eight_directions import components, restore_checkpoint, validate_config
from scripts.service_model import (
    LocalRuntime,
    _json,
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
    fused_score,
    fusion_scales,
    score_candidate,
)
from src.service.modeling.evaluation import calibrate, evaluate_run
from src.service.modeling.temporal import temporal_inputs
from src.service.modeling.temporal_tuning import prepare_inner
from src.service.preparation import file_digest
from src.utils.seed import set_seed


def declared(run, manifest, name):
    return run / manifest["artifacts"][name]["path"]


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def canonical_json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def verify_tracking(connection, manifest, histories, config, report):
    run_id = manifest["mlflow_run_id"]
    direction = manifest["run_id"].rsplit("-seed", 1)[0]
    if direction not in {"baseline", *DIRECTIONS}:
        raise ValueError("Unknown training direction")
    expected_schedule = []
    if direction not in {"prototype", "mechanism"}:
        if direction == "masked_pretraining":
            expected_schedule.extend(
                ("pretraining", epoch) for epoch in range(1, config["pretraining_epochs"] + 1)
            )
        expected_schedule.extend(
            ("prediction", epoch) for epoch in range(1, config["training"]["epochs"] + 1)
        )
    if set(histories) != {"inner", "outer"}:
        raise ValueError("Training history scopes differ from the frozen schedule")
    for records in histories.values():
        if (
            any(type(record["epoch"]) is not int for record in records)
            or [(record["phase"], record["epoch"]) for record in records] != expected_schedule
        ):
            raise ValueError("Training history phase/epoch sequence differs from frozen schedule")
    status = connection.execute("SELECT status FROM runs WHERE run_uuid=?", (run_id,)).fetchone()
    if status != ("FINISHED",):
        raise ValueError("MLflow run is not finished")
    count = 0
    for scope, records in histories.items():
        for record in records:
            step = record["epoch"]
            if direction == "masked_pretraining" and record["phase"] == "prediction":
                step += config["pretraining_epochs"]
            if scope == "outer":
                step += 100
            key = f"{scope}_{record['phase']}_fit_loss"
            rows = connection.execute(
                "SELECT value FROM metrics WHERE run_uuid=? AND key=? AND step=?",
                (run_id, key, step),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError("MLflow epoch is missing or duplicated")
            assert_close(rows[0][0], record["mean_loss"])
            count += 1
    tracked_count = connection.execute(
        "SELECT COUNT(*) FROM metrics WHERE run_uuid=? AND key GLOB '*_fit_loss'", (run_id,)
    ).fetchone()[0]
    if tracked_count != count:
        raise ValueError("MLflow training epoch count differs from the frozen schedule")
    for key, value in [("main_auroc", report["main"]["auroc"]), ("main_ap", report["main"]["ap"])]:
        rows = connection.execute(
            "SELECT value FROM metrics WHERE run_uuid=? AND key=? AND step=250", (run_id, key)
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("MLflow main metric is missing or duplicated")
        assert_close(rows[0][0], value)
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["suite", "prepared", "preparation", "label-bundle", "output", "tracking-db"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    suite = json.loads((args.suite / "suite.json").read_text())
    config = suite["protocol"]
    validate_config(config)
    if suite["prepared"] != _reference(args.prepared):
        raise ValueError("Suite prepared reference mismatch")
    for path, digest in suite["source_sha256"].items():
        if file_digest(Path(path)) != digest:
            raise ValueError("Suite source has changed")
    tables, outer, _ = read_prepared(args.prepared)
    raw_tables = load_inputs(args.preparation, args.label_bundle, tables.config)
    if raw_tables.hashes != tables.hashes or raw_tables.versions != tables.versions:
        raise ValueError("Audit raw source binding mismatch")
    inner, audit = prepare_inner(raw_tables, outer, LocalRuntime().tokenizer(tables.config))
    audit = canonical_json_value(audit)
    inputs = {
        scope: temporal_inputs(
            prepared, 4, availability_prior=fit_availability(prepared, prepared.lists.fit)
        )
        for scope, prepared in [("inner", inner), ("outer", outer)]
    }
    connection = sqlite3.connect(f"file:{args.tracking_db.resolve()}?mode=ro", uri=True)
    rows = []
    for direction in ["baseline", *DIRECTIONS]:
        for seed in config["seeds"]:
            set_seed(seed)
            run = args.suite / f"{direction}-seed{seed}"
            checked = check_run(run)
            manifest = json.loads((run / "manifest.json").read_text())
            stored_config = json.loads((run / "run/config.json").read_text())
            if (
                stored_config["protocol"] != config
                or stored_config["direction"] != direction
                or stored_config["seed"] != seed
            ):
                raise ValueError("Run does not match suite protocol")
            provenance = manifest["provenance"]
            if provenance["parents"]["prepared"] != suite["prepared"]:
                raise ValueError("Candidate prepared parent mismatch")
            if (
                provenance["source_sha256"] != suite["source_sha256"]
                or provenance["versions"] != outer.versions
            ):
                raise ValueError("Source or data versions differ")
            check_contexts(run, manifest, outer, 4)
            persisted_audit = json.loads(
                declared(run, manifest, "inner_split_and_preprocessing").read_text()
            )
            if persisted_audit != audit:
                raise ValueError("Rebuilt inner preprocessing differs")
            inner_history = json.loads(
                declared(run, manifest, "inner_context_sample_ids").read_text()
            )
            if inner_history != inputs["inner"]["history_ids"]:
                raise ValueError("Inner contexts differ from the independent reconstruction")
            report = json.loads(declared(run, manifest, "evaluation_report").read_text())
            scores = pd.read_parquet(declared(run, manifest, "scores"))
            calibration = pd.read_parquet(declared(run, manifest, "calibration_scores"))
            validation = pd.read_parquet(declared(run, manifest, "normal_validation_scores"))
            if set(scores.sample_id) != score_ids(outer.lists, "test"):
                raise ValueError("Test membership differs")
            if set(calibration.sample_id) != set(outer.lists.calibration) or set(
                validation.sample_id
            ) != set(audit["validation_ids"]):
                raise ValueError("Normal membership differs")
            thresholds = calibrate(calibration, set(outer.lists.calibration), config["quantiles"])
            persisted_thresholds = json.loads(declared(run, manifest, "thresholds").read_text())
            if thresholds != persisted_thresholds:
                raise ValueError("Recomputed Normal thresholds differ")
            rebuilt = evaluate_run(tables, outer.lists, scores, thresholds, score_scope="test")
            if rebuilt != report:
                raise ValueError("Persisted evaluation differs from recomputation")
            checked_checkpoints = 0
            fusion_reference = json.loads(declared(run, manifest, "fusion_reference").read_text())
            for scope, prepared in [("inner", inner), ("outer", outer)]:
                model, checkpoint = restore_checkpoint(
                    declared(run, manifest, f"{scope}_checkpoint"), config["training"]["device"]
                )
                if (
                    checkpoint["seed"] != seed
                    or checkpoint["direction"] != direction
                    or checkpoint["config"] != config
                ):
                    raise ValueError("Checkpoint seed/direction/configuration mismatch")
                if checkpoint["fit_ids"] != sorted(prepared.lists.fit):
                    raise ValueError("Checkpoint fit membership mismatch")
                expected_direction = (
                    "baseline" if direction in {"prototype", "mechanism"} else direction
                )
                if checkpoint["model_direction"] != expected_direction:
                    raise ValueError("Restored architecture does not match the candidate")
                if direction in {"prototype", "mechanism"}:
                    baseline_path = args.suite / f"baseline-seed{seed}" / f"run/{scope}-model.pt"
                    baseline_checkpoint = torch.load(
                        baseline_path, map_location="cpu", weights_only=True
                    )
                    for name, value in checkpoint["model_state"].items():
                        if not torch.equal(value.cpu(), baseline_checkpoint["model_state"][name]):
                            raise ValueError("Postprocessing changed original GRU parameters")
                    if checkpoint["fusion_scales"] != fusion_reference["scales"]:
                        raise ValueError("Checkpoint fusion reference mismatch")
                if (
                    direction == "prototype"
                    and checkpoint["prototype_memory"]["fit_ids"] != checkpoint["fit_ids"]
                ):
                    raise ValueError("Prototype membership differs from known Normal fitting")
                labels = prepared.labels.set_index("sample_id").loc[checkpoint["fit_ids"]]
                if (
                    not labels.label.eq(0).all()
                    or not labels.label_source.eq("normal_run_reference").all()
                ):
                    raise ValueError("Non-Normal checkpoint fitting")
                if scope == "inner":
                    identity = prepared.identity.set_index("sample_id")
                    if set(identity.loc[checkpoint["fit_ids"], "case_id"]) & set(
                        identity.loc[audit["validation_ids"], "case_id"]
                    ):
                        raise ValueError("Normal validation Case overlaps fitting")
                if checkpoint["availability_prior"] != inputs[scope]["availability_prior"]:
                    raise ValueError("Checkpoint prior differs from fit-only recomputation")
                persisted_frames = [validation] if scope == "inner" else [scores, calibration]
                for persisted in persisted_frames:
                    ids = set(persisted.sample_id)
                    if direction in {"prototype", "mechanism"}:
                        reconstructed, values = components(
                            model, inputs[scope], ids, direction, checkpoint["prototype_memory"]
                        )
                        reconstructed["score"] = fused_score(
                            values, checkpoint["fusion_scales"], config["fusion_auxiliary_weight"]
                        )
                        if scope == "inner":
                            assert_close(
                                fusion_scales(values, config["fusion_scale_floor"]),
                                checkpoint["fusion_scales"],
                            )
                    else:
                        reconstructed = score_candidate(model, inputs[scope], ids)
                    expected = persisted.set_index("sample_id").loc[
                        reconstructed.sample_id, "score"
                    ]
                    assert_close(reconstructed.score.to_numpy(), expected.to_numpy())
                checked_checkpoints += 1
            histories = json.loads(declared(run, manifest, "training_history").read_text())
            epochs = verify_tracking(connection, manifest, histories, config, report)
            normal = json.loads(declared(run, manifest, "normal_validation_summary").read_text())
            fit_summary = json.loads(declared(run, manifest, "fit_summary").read_text())
            if direction == "heteroscedastic":
                normal["original_coordinate_mse"] = float(validation.auxiliary_loss.mean())
            main = report["main"]
            row = dict(
                direction=direction,
                seed=seed,
                run_id=run.name,
                mlflow_run_id=manifest["mlflow_run_id"],
                auroc=main["auroc"],
                ap=main["ap"],
                q99=main["classification"]["0.99"],
                normal_validation=normal,
                groups={
                    name: dict(
                        rows=group["rows"],
                        auroc=group["auroc"],
                        ap=group["ap"],
                        q99=group["classification"]["0.99"],
                    )
                    for name, group in main["by_case"].items()
                },
                artifacts_checked=checked["artifacts_checked"],
                mlflow_epochs_verified=epochs,
                checkpoints_restored=checked_checkpoints,
                parameter_count=fit_summary["parameter_count"],
                training_seconds=sum(
                    records[-1]["elapsed_seconds"] if records else 0
                    for records in histories.values()
                ),
                sample_binding=content_hash(
                    dict(
                        fit=checkpoint["fit_ids"],
                        calibration=sorted(calibration.sample_id),
                        test=sorted(scores.sample_id),
                        main=main["sample_ids"],
                        versions=outer.versions,
                    )
                ),
                report_sha256=file_digest(declared(run, manifest, "evaluation_report")),
                manifest_sha256=file_digest(run / "manifest.json"),
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in ["direction", "seed", "auroc", "ap", "mlflow_epochs_verified"]
                    }
                ),
                flush=True,
            )
    connection.close()
    if len({row["sample_binding"] for row in rows}) != 1:
        raise ValueError("Candidates have different frozen sample identities")
    aggregate = {}
    baseline = {row["seed"]: row for row in rows if row["direction"] == "baseline"}
    for direction in ["baseline", *DIRECTIONS]:
        selected = [row for row in rows if row["direction"] == direction]
        summary = {
            metric: dict(
                mean=float(np.mean([row[metric] for row in selected])),
                sample_std=float(np.std([row[metric] for row in selected], ddof=1)),
                paired_delta=[row[metric] - baseline[row["seed"]][metric] for row in selected],
            )
            for metric in ["auroc", "ap"]
        }
        summary["stable_replacement"] = (
            all(
                row[metric] > baseline[row["seed"]][metric]
                for row in selected
                for metric in ["auroc", "ap"]
            )
            if direction != "baseline"
            else False
        )
        aggregate[direction] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _json(
        args.output,
        dict(
            status="success",
            replay_device=config["training"]["device"],
            runs=len(rows),
            rows=rows,
            aggregate=aggregate,
            source_sha256=suite["source_sha256"],
            prepared=suite["prepared"],
            inner_preprocessor_hash=audit["preprocessor_hash"],
            sample_binding=rows[0]["sample_binding"],
            interpretation="seen_test_exploration_not_blind_validation",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
