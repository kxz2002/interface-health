"""Explicit offline stages for immutable Service model artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import platform
import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.service.modeling.baselines import fit_baselines, score_baselines
from src.service.modeling.config import DEFAULT_CONFIG, content_hash, load_config
from src.service.modeling.data import (
    InputTables,
    PreparedData,
    SampleLists,
    assemble_prepared,
    build_sample_lists,
    load_inputs,
    validate_sample_lists,
)
from src.service.modeling.evaluation import annotate_calibration, calibrate, evaluate_run
from src.service.modeling.logs import (
    _snapshot,
    fit_log_statistics,
    fit_logs,
    prepare_logs,
    transform_log_statistics,
)
from src.service.modeling.model import build_model
from src.service.modeling.numeric import fit_numeric, transform_numeric
from src.service.modeling.runs import RunRecorder, check_reuse
from src.service.modeling.training import _binding, probe_fit, score_model, train_model
from src.service.preparation import checked_path, file_digest
from src.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)
MATRIX_CONFIG = DEFAULT_CONFIG.with_name("tt_model_matrix_v1.json")
NEURAL_METHODS = {
    "transformer_cls": "summary_token",
    "transformer_mean": "mean",
    "concat_svdd": "concat_svdd",
}
BASELINE_METHODS = ("L2", "Max", "numeric_IF", "numeric_OCSVM", "trimodal_IF", "trimodal_OCSVM")
METHODS = (*NEURAL_METHODS, *BASELINE_METHODS)
SCOPES = ("calibration", "main_test", "diagnostic", "extended_test", "test")


class LocalRuntime:
    identity = {"runtime": "local_fixed_revision", "production_bert": True}

    @staticmethod
    def _revision(config, name):
        revision = config["log"][name]
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError(f"Production text requires a fixed 40-hex {name}")
        return revision

    def tokenizer(self, config):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            config["log"]["model"],
            revision=self._revision(config, "tokenizer_revision"),
            local_files_only=True,
        )

    def backbone(self, config):
        from transformers import AutoModel

        return AutoModel.from_pretrained(
            config["log"]["model"],
            revision=self._revision(config, "revision"),
            local_files_only=True,
        )


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _manifest(run):
    check_run(run)
    return _read(Path(run) / "manifest.json")


def _reference(run):
    run = Path(run).resolve()
    return {"path": str(run), "manifest_sha256": file_digest(run / "manifest.json")}


def check_run(run, _visited=None):
    root = Path(run).resolve()
    visited = set() if _visited is None else _visited
    if root in visited:
        raise ValueError("Cyclic run artifact lineage")
    visited.add(root)
    manifest = _read(root / "manifest.json")
    status = _read(root / "status.json")
    if manifest.get("status") != "success" or status.get("status") != "success":
        raise ValueError("Run is not successfully published")
    if manifest.get("run_id") != root.name or status.get("run_id") != root.name:
        raise ValueError("Run directory identity mismatch")
    if manifest.get("mlflow_run_id") != status.get("mlflow_run_id"):
        raise ValueError("MLflow run identity mismatch")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or not records:
        raise ValueError("Missing run artifacts")
    for record in records.values():
        path = checked_path(root, record["path"])
        if not path.is_file() or file_digest(path) != record["sha256"]:
            raise ValueError(f"Run artifact hash mismatch: {record['path']}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError("Run artifact size mismatch")
    for reference in manifest["provenance"].get("parents", {}).values():
        path = Path(reference["path"]) / "manifest.json"
        if not path.is_file() or file_digest(path) != reference["manifest_sha256"]:
            raise ValueError("Parent manifest hash mismatch")
        check_run(reference["path"], visited.copy())
    if _read(root / "run/provenance.json") != manifest["provenance"]:
        raise ValueError("Persisted provenance mismatch")
    config = _read(root / "run/config.json")
    encoded = json.dumps(config, sort_keys=True, indent=2, allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != manifest["config_sha256"]:
        raise ValueError("Run configuration hash mismatch")
    return {
        "status": "success",
        "stage": manifest["stage"],
        "run_id": manifest["run_id"],
        "mlflow_run_id": manifest["mlflow_run_id"],
        "artifacts_checked": len(records),
    }


def score_ids(lists, scope):
    validate_sample_lists(lists)
    if scope == "calibration":
        return set(lists.calibration)
    if scope == "main_test":
        return set(lists.main_test)
    diagnostic = set(lists.diagnostic_pre) | set(lists.diagnostic_positive)
    if scope == "diagnostic":
        return diagnostic
    if scope == "test":
        return set(lists.main_test) | diagnostic
    if scope == "extended_test":
        return (
            set(lists.main_test)
            | diagnostic
            | set(lists.extended_inference)
            | set(lists.extended_main_test)
        )
    raise ValueError("Unknown scoring scope")


def _pack(value, root, artifacts):
    if isinstance(value, pd.DataFrame):
        name = f"prepared_frame_{len(artifacts):03}"
        path = root / "data" / f"{name}.parquet"
        value.to_parquet(path, index=False)
        artifacts[name] = path
        return {"__frame__": str(path.relative_to(root))}
    if isinstance(value, dict):
        return {key: _pack(item, root, artifacts) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item, root, artifacts) for item in value]
    return value


def _unpack(value, root):
    if isinstance(value, dict):
        if set(value) == {"__frame__"}:
            return pd.read_parquet(checked_path(root, value["__frame__"]))
        return {key: _unpack(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, root) for item in value]
    return value


def _lists(value):
    value = copy.deepcopy(value)
    for name in [
        "common_fit",
        "common_calibration",
        "common_main_test",
        "extended_inference",
        "extended_main_test",
        "diagnostic_pre",
        "diagnostic_positive",
        "excluded",
    ]:
        value[name] = tuple(value[name])
    for group in value["diagnostic_groups"].values():
        for name in ["pre", "positive"]:
            group[name] = tuple(group[name])
    lists = SampleLists(**value)
    validate_sample_lists(lists)
    return lists


def _save_prepared(root, tables, prepared, states):
    artifacts = {}
    payload = dict(
        schema_version="service-model-prepared-v1",
        prepared={
            name: getattr(prepared, name)
            for name in [
                "sample_ids",
                "numeric",
                "logs",
                "identity",
                "labels",
                "splits",
                "versions",
                "hashes",
                "provenance",
            ]
        },
        lists=asdict(prepared.lists),
        tables={
            name: getattr(tables, name)
            for name in [
                "grid",
                "labels",
                "splits",
                "evaluation",
                "versions",
                "hashes",
                "config",
                "protocol",
            ]
        },
        preprocessing=states,
    )
    path = root / "data/prepared.json"
    _json(path, _pack(payload, root, artifacts))
    artifacts["prepared"] = path
    artifacts["sample_lists"] = asdict(prepared.lists)
    return artifacts


def read_prepared(run):
    manifest = _manifest(run)
    if manifest["stage"] != "prepare":
        raise ValueError("Expected a prepared run")
    root = Path(run)
    value = _unpack(_read(root / "data/prepared.json"), root)
    if value["schema_version"] != "service-model-prepared-v1":
        raise ValueError("Unsupported prepared artifact schema")
    value["prepared"]["sample_ids"] = tuple(value["prepared"]["sample_ids"])
    prepared = PreparedData(**value["prepared"], lists=_lists(value["lists"]))
    empty = pd.DataFrame()
    tables = InputTables(
        **value["tables"], resource=empty, execution=empty, log=empty, feature_tables={}
    )
    if tables.versions != prepared.versions or tables.hashes != prepared.hashes:
        raise ValueError("Prepared input identity mismatch")
    if manifest["provenance"]["lists_hash"] != prepared.lists.lists_hash:
        raise ValueError("Prepared lists manifest mismatch")
    return tables, prepared, manifest


def read_scores(run):
    manifest = _manifest(run)
    if manifest["stage"] != "score":
        raise ValueError("Expected a score run")
    return pd.read_parquet(Path(run) / "run/scores.parquet")


def _input_config(config):
    return {
        key: config[key]
        for key in [
            "features",
            "preprocessing",
            "log",
            "quality_rule",
            "window_ms",
            "dataset",
            "preparation_schema_version",
            "label_schema_version",
        ]
    }


def _run_config(config, method, seed):
    config = copy.deepcopy(config)
    if method in NEURAL_METHODS:
        config["model"]["fusion"] = NEURAL_METHODS[method]
    config["training"]["prototype_seed"] = seed
    config.pop("config_sha256", None)
    config["config_sha256"] = content_hash(config)
    return config


def _runtime_identity(runtime, backbone=None):
    result = dict(runtime.identity)
    if backbone is not None:
        digest = hashlib.sha256()
        for name, value in backbone.state_dict().items():
            digest.update(name.encode())
            digest.update(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        result["initial_backbone_sha256"] = digest.hexdigest()
        result["backbone_class"] = f"{type(backbone).__module__}.{type(backbone).__qualname__}"
    return result


def _environment():
    packages = {}
    for name in ["numpy", "pandas", "pyarrow", "scikit-learn", "torch", "transformers", "mlflow"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    return dict(
        python=sys.version,
        platform=platform.platform(),
        packages=packages,
        git_commit=commit,
        git_dirty=dirty,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
    )


def _record(output, config, args, provenance):
    return RunRecorder(
        Path(output),
        {
            **config,
            "tracking_uri": args.tracking_uri,
            "experiment_name": "service-model-infrastructure",
        },
        {**provenance, "environment": _environment()},
    )


def _publish(recorder, artifacts, *, local_only=None):
    artifacts["resolved_config"] = recorder.output / "run/config.json"
    artifacts["provenance"] = recorder.output / "run/provenance.json"
    return recorder.publish(artifacts, local_only=local_only)


def _prepare(args, config, runtime):
    tables = load_inputs(Path(args.preparation), Path(args.label_bundle), config)
    provenance = dict(
        stage=args.command,
        versions=tables.versions,
        input_hashes=tables.hashes,
        protocol_hash=content_hash(tables.protocol),
        runtime=_runtime_identity(runtime),
    )
    if args.command == "check-inputs":
        with _record(args.output, config, args, provenance) as recorder:
            _publish(
                recorder,
                {
                    "input_report": dict(
                        rows=len(tables.grid),
                        versions=tables.versions,
                        hashes=tables.hashes,
                        protocol=tables.protocol,
                        feature_order=config["features"],
                    )
                },
            )
        return
    with _record(args.output, config, args, provenance) as recorder:
        if isinstance(runtime, LocalRuntime):
            runtime._revision(config, "revision")
            runtime._revision(config, "tokenizer_revision")
        tokenizer = runtime.tokenizer(config)
        permitted = set(tables.splits.loc[tables.splits.split.eq("fit"), "sample_id"])
        permitted &= set(tables.labels.loc[tables.labels.label.eq(0).fillna(False), "sample_id"])
        numeric_state = fit_numeric(tables, permitted, config)
        numeric = transform_numeric(tables, numeric_state)
        valid = numeric["masks"].set_index("sample_id").any(axis=1)
        fit_ids = permitted & set(valid.index[valid])
        if not fit_ids:
            raise ValueError("No known-normal common fit windows")
        numeric_state = fit_numeric(tables, fit_ids, config)
        numeric = transform_numeric(tables, numeric_state)
        log_state = fit_logs(tables, fit_ids, config)
        logs = prepare_logs(tables, log_state, tokenizer, config)
        lists = build_sample_lists(tables, numeric, pd.Series(logs["masks"]), "common")
        if set(lists.fit) != fit_ids:
            raise ValueError("Numeric fit membership changed after freezing")
        statistics_state = fit_log_statistics(tables, logs, fit_ids, config)
        logs["statistics"] = transform_log_statistics(tables, logs, statistics_state)
        states = dict(
            numeric=asdict(numeric_state),
            log={
                name: getattr(log_state, name)
                for name in [
                    "templates",
                    "config",
                    "fit_ids",
                    "fit_input_hash",
                    "snapshot_hash",
                    "state_hash",
                ]
            },
            drain_snapshot=_snapshot(log_state.miners),
            log_statistics=asdict(statistics_state),
        )
        prepared = assemble_prepared(
            tables,
            numeric,
            logs,
            lists,
            provenance=dict(
                numeric_state_hash=numeric_state.state_hash,
                log_state_hash=log_state.state_hash,
                log_statistics_state_hash=statistics_state.state_hash,
                preprocessing_reference="data/prepared.json",
                tokenizer=logs["cache"]["tokenizer"],
            ),
        )
        recorder.provenance["lists_hash"] = lists.lists_hash
        _json(recorder.output / "run/provenance.json", recorder.provenance)
        artifacts = _save_prepared(recorder.output, tables, prepared, states)
        _publish(recorder, artifacts, local_only={"prepared"})


def _model_lists(manifest, prepared, root):
    if manifest["provenance"]["method"] in NEURAL_METHODS:
        return prepared.lists
    from src.service.modeling.baselines import baseline_sample_lists

    bundle = torch.load(Path(root) / "run/baseline.pt", map_location="cpu", weights_only=False)
    arm = "trimodal" if manifest["provenance"]["method"].startswith("trimodal") else "numeric"
    return baseline_sample_lists(bundle, prepared, prepared.lists)[arm]


def _frame_hash(frame, ids=None):
    selected = frame if ids is None else frame[frame.sample_id.isin(ids)]
    selected = selected.sort_values("sample_id").reset_index(drop=True)
    selected = selected.drop(
        columns=[name for name in selected if name.endswith("_version")], errors="ignore"
    )
    # Parquet-backed evidence frames contain list/ndarray object cells (for
    # example label evidence IDs).  pandas' native object hasher attempts to
    # factorize those values and fails because ndarrays are unhashable.  Use a
    # deterministic JSON representation for object columns before hashing so
    # the reuse check remains content-sensitive without mutating the source.
    for name in selected.columns:
        if selected[name].dtype == "object":
            selected[name] = selected[name].map(
                lambda value: json.dumps(
                    value.tolist() if isinstance(value, np.ndarray) else value,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
    digest = hashlib.sha256(pd.util.hash_pandas_object(selected, index=False).values.tobytes())
    digest.update(json.dumps(list(selected.columns)).encode())
    return digest.hexdigest()


def _observation_hash(prepared, ids):
    ids = sorted(ids)
    return content_hash(
        dict(
            numeric={name: _frame_hash(frame, ids) for name, frame in prepared.numeric.items()},
            logs={
                name: {sample_id: prepared.logs[name][sample_id] for sample_id in ids}
                for name in [
                    "input_ids",
                    "attention_mask",
                    "special_mask",
                    "group_ids",
                    "masks",
                    "stats",
                ]
            },
            statistics={
                name: _frame_hash(prepared.logs["statistics"][name], ids)
                for name in ["values", "masks"]
            },
            identity=_frame_hash(prepared.identity, ids),
        )
    )


def _reuse_evidence(tables, prepared, config, required=(), scored=()):
    fit, cal = set(prepared.lists.fit), set(prepared.lists.calibration)
    return dict(
        fit_ids_hash=content_hash(dict(ids=sorted(fit))),
        calibration_ids_hash=content_hash(dict(ids=sorted(cal))),
        fit_labels_hash=_frame_hash(prepared.labels, fit),
        calibration_labels_hash=_frame_hash(prepared.labels, cal),
        fit_eligibility_hash=content_hash(dict(ids=sorted(fit))),
        calibration_eligibility_hash=content_hash(dict(ids=sorted(cal))),
        fit_input_hash=_observation_hash(prepared, fit),
        calibration_input_hash=_observation_hash(prepared, cal),
        input_hash=_observation_hash(prepared, prepared.sample_ids),
        mask_hash=_frame_hash(prepared.numeric["masks"]),
        calibration_mask_hash=_frame_hash(prepared.numeric["masks"], cal),
        template_hash=prepared.provenance["log_state_hash"],
        preprocessor_hash=content_hash(prepared.provenance),
        feature_hash=content_hash(config["features"]),
        feature_order=list(prepared.numeric["values"].columns),
        model_config_hash=config["config_sha256"],
        model_revision=config["log"]["revision"],
        seed=config["training"]["prototype_seed"],
        split_hash=_frame_hash(prepared.splits),
        split_version=prepared.versions["split_version"],
        quality_rule_hash=content_hash(dict(rule=config["quality_rule"])),
        calibration_config_hash=content_hash(config["calibration"]),
        label_version=prepared.versions["label_version"],
        versions=prepared.versions,
        evaluation_version=prepared.versions["evaluation_version"],
        evaluation_table_hash=_frame_hash(tables.evaluation),
        evaluation_manifest_hash=content_hash(tables.protocol),
        test_labels_hash=_frame_hash(prepared.labels, set(prepared.sample_ids) - fit - cal),
        main_test_hash=content_hash(dict(ids=prepared.lists.main_test)),
        diagnostic_lists_hash=content_hash(
            dict(pre=prepared.lists.diagnostic_pre, positive=prepared.lists.diagnostic_positive)
        ),
        required_score_ids=sorted(required),
        scored_ids=sorted(scored),
    )


def _compatibility(model_manifest, tables, prepared, config, required=(), scored=()):
    original_path = model_manifest["provenance"]["parents"]["prepared"]["path"]
    old_tables, original, _ = read_prepared(original_path)
    before = _reuse_evidence(old_tables, original, config, required, scored)
    after = _reuse_evidence(tables, prepared, config, required, scored)
    report = check_reuse(before, after)
    if not report["compatible"]:
        raise ValueError(f"Artifact reuse requires refit/recalibration: {report['reasons']}")
    report["training_prepared"] = _reference(original_path)
    report["current_prepared_versions"] = prepared.versions
    report["evidence_before"] = before
    report["evidence_after"] = after
    return original, report


def _fit(args, config, runtime):
    tables, prepared, prepared_manifest = read_prepared(args.prepared)
    if _input_config(tables.config) != _input_config(config):
        raise ValueError("Prepared input preprocessing configuration mismatch")
    config = _run_config(
        config,
        args.method,
        config["training"]["prototype_seed"] if args.seed is None else args.seed,
    )
    provenance = dict(
        stage=args.command,
        versions=prepared.versions,
        parents={"prepared": _reference(args.prepared)},
        lists_hash=prepared.lists.lists_hash,
        method=args.method,
        seed=config["training"]["prototype_seed"],
        runtime=_runtime_identity(runtime),
        feature_order=list(prepared.numeric["values"].columns.drop("sample_id")),
        input_hashes=prepared.hashes,
        preprocessor=prepared.provenance,
    )
    with _record(args.output, config, args, provenance) as recorder:
        set_seed(config["training"]["prototype_seed"])
        needs_text = args.method in NEURAL_METHODS or args.method.startswith("trimodal")
        backbone = runtime.backbone(config) if needs_text else None
        recorder.provenance["runtime"] = _runtime_identity(runtime, backbone)
        _json(recorder.output / "run/provenance.json", recorder.provenance)
        if args.command == "probe-fit":
            if args.method not in NEURAL_METHODS:
                raise ValueError("probe-fit requires a neural method")
            model = build_model(config, backbone).to(config["training"]["device"] or "cpu")
            report = probe_fit(model, prepared, prepared.lists, config)
            _json(recorder.output / "report/probe_report.json", report)
            if report["status"] != "completed":
                raise RuntimeError(f"Fit probe failed: {report}")
            _publish(recorder, {"probe_report": recorder.output / "report/probe_report.json"})
        elif args.method in NEURAL_METHODS:
            model = build_model(config, backbone)
            parameter_counts = dict(
                total=sum(parameter.numel() for parameter in model.parameters()),
                trainable=sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
            )
            result = train_model(
                model, prepared, prepared.lists, config, recorder.output / "run/model"
            )
            if result["status"] != "completed":
                raise RuntimeError(f"Model training failed: {result}")
            for row in result["diagnostics"]:
                recorder.log_epoch(
                    {key: row[key] for key in ["total_variance", "mean_loss"] if key in row},
                    row["epoch"],
                )
            _publish(
                recorder,
                {
                    "checkpoint": Path(result["checkpoint"]),
                    "training_result": result,
                    "sample_lists": asdict(prepared.lists),
                    "parameter_counts": parameter_counts,
                },
            )
        else:
            bundle = fit_baselines(
                prepared, prepared.lists, config, text_backbone=backbone, methods=(args.method,)
            )
            arm = "trimodal" if args.method.startswith("trimodal") else "numeric"
            members = bundle["method_lists"][arm]
            path = recorder.output / "run/baseline.pt"
            torch.save(bundle, path)
            lists_path = recorder.output / "data/method_lists.json"
            _json(lists_path, asdict(members))
            _publish(
                recorder,
                {
                    "baseline": path,
                    "method_lists": lists_path,
                    "baseline_parameters": bundle["resolved_parameters"],
                    "sample_lists": asdict(prepared.lists),
                },
            )


def _model_config(args, config, manifest):
    provenance = manifest["provenance"]
    config = _run_config(config, provenance["method"], provenance["seed"])
    stored = _read(Path(args.model_run) / "run/config.json")
    if stored["config_sha256"] != config["config_sha256"]:
        raise ValueError("Model resolved configuration mismatch")
    return config


def _restore(args, config, runtime, manifest, prepared):
    provenance = manifest["provenance"]
    method = provenance["method"]
    root = Path(args.model_run)
    if method in NEURAL_METHODS:
        set_seed(config["training"]["prototype_seed"])
        backbone = runtime.backbone(config)
        if _runtime_identity(runtime, backbone) != provenance["runtime"]:
            raise ValueError("Text backbone runtime identity mismatch")
        model = build_model(config, backbone)
        path = checked_path(root, manifest["artifacts"]["checkpoint"]["path"])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        binding = _binding(prepared, config)
        if any(checkpoint.get(name) != value for name, value in binding.items()):
            raise ValueError("Checkpoint input/configuration binding mismatch")
        if checkpoint["fit_ids_hash"] != content_hash(dict(ids=sorted(prepared.lists.fit))):
            raise ValueError("Checkpoint fit IDs mismatch")
        model.load_state_dict(checkpoint["model_state"])
        model.training_identity = {**binding, "checkpoint_sha256": file_digest(path)}
        return model.to(config["training"]["device"] or "cpu")
    path = checked_path(root, manifest["artifacts"]["baseline"]["path"])
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle["binding"] != _binding(prepared, config):
        raise ValueError("Baseline configuration binding mismatch")
    return bundle


def _score(args, config, runtime):
    tables, prepared, _ = read_prepared(args.prepared)
    manifest = _manifest(args.model_run)
    if manifest["stage"] != "fit":
        raise ValueError("Scoring requires a successful fit run")
    config = _model_config(args, config, manifest)
    lists = _model_lists(manifest, prepared, args.model_run)
    ids = score_ids(lists, args.score_scope)
    original, reuse = _compatibility(manifest, tables, prepared, config, ids)
    existing = None
    if args.reuse_scores:
        source_manifest = _manifest(args.reuse_scores)
        if source_manifest["stage"] != "score" or source_manifest["provenance"]["parents"][
            "model"
        ] != _reference(args.model_run):
            raise ValueError("Reused scores belong to a different model")
        source_tables, source_prepared, _ = read_prepared(
            source_manifest["provenance"]["parents"]["prepared"]["path"]
        )
        _compatibility(manifest, source_tables, source_prepared, config)
        existing = read_scores(args.reuse_scores)
        existing = existing[existing.sample_id.isin(ids)].copy()
    pending = ids - (set(existing.sample_id) if existing is not None else set())
    reuse["rescore_ids"] = sorted(pending)
    provenance = dict(
        stage="score",
        versions=prepared.versions,
        model_manifest=manifest,
        parents={"prepared": _reference(args.prepared), "model": _reference(args.model_run)},
        method=manifest["provenance"]["method"],
        score_scope=args.score_scope,
        lists_hash=lists.lists_hash,
        required_score_ids=sorted(ids),
        purpose_lists=asdict(lists),
        compatibility=reuse,
    )
    if args.reuse_scores:
        provenance["parents"]["reused_scores"] = _reference(args.reuse_scores)
    with _record(args.output, config, args, provenance) as recorder:
        if pending:
            model = _restore(args, config, runtime, manifest, original)
            if provenance["method"] in NEURAL_METHODS:
                scores = score_model(model, original, pending)
            else:
                scores = score_baselines(model, original, pending)[provenance["method"]]
            if existing is not None:
                scores = pd.concat([existing, scores], ignore_index=True)
        else:
            scores = existing
        scores = scores.sort_values("sample_id").reset_index(drop=True)
        if set(scores.sample_id) != ids or scores.sample_id.duplicated().any():
            raise ValueError("Scores differ from the frozen requested scope")
        if not np.isfinite(scores.score.to_numpy(float)).all():
            raise ValueError("Nonfinite model scores")
        if "config_sha256" in scores and not scores.config_sha256.eq(config["config_sha256"]).all():
            raise ValueError("Conflicting score configuration binding")
        scores["config_sha256"] = config["config_sha256"]
        if args.score_scope == "calibration":
            scores = annotate_calibration(scores, original, ids)
        path = recorder.output / "run/scores.parquet"
        scores.to_parquet(path, index=False)
        _publish(recorder, {"scores": path, "score_lists": provenance["purpose_lists"]})


def _evaluate(args, config):
    tables, prepared, _ = read_prepared(args.prepared)
    model_manifest = _manifest(args.model_run)
    if model_manifest["stage"] != "fit":
        raise ValueError("Evaluation requires a successful fit run")
    config = _model_config(args, config, model_manifest)
    lists = _model_lists(model_manifest, prepared, args.model_run)
    _, reuse = _compatibility(model_manifest, tables, prepared, config)
    scores_manifest, calibration_manifest = _manifest(args.scores), _manifest(
        args.calibration_scores
    )
    score_scope = scores_manifest["provenance"]["score_scope"]
    if (
        score_scope not in {"test", "extended_test"}
        or calibration_manifest["provenance"]["score_scope"] != "calibration"
    ):
        raise ValueError("Evaluation requires test and separate calibration score runs")
    for manifest in [scores_manifest, calibration_manifest]:
        if manifest["stage"] != "score" or manifest["provenance"]["parents"]["model"] != _reference(
            args.model_run
        ):
            raise ValueError("Score/model artifact identity mismatch")
        source_tables, source_prepared, _ = read_prepared(
            manifest["provenance"]["parents"]["prepared"]["path"]
        )
        _compatibility(model_manifest, source_tables, source_prepared, config)
    scores, calibration_scores = read_scores(args.scores), read_scores(args.calibration_scores)
    missing = score_ids(lists, score_scope) - set(scores.sample_id)
    if missing:
        raise ValueError(f"Missing required scores; rescore_ids={sorted(missing)}")
    for frame in [scores, calibration_scores]:
        if not frame.config_sha256.eq(config["config_sha256"]).all():
            raise ValueError("Score model configuration mismatch")
    provenance = dict(
        stage="evaluate",
        versions=prepared.versions,
        model_manifest=model_manifest,
        parents={
            "prepared": _reference(args.prepared),
            "model": _reference(args.model_run),
            "scores": _reference(args.scores),
            "calibration": _reference(args.calibration_scores),
        },
        evaluation_id=Path(args.output).name,
        lists_hash=lists.lists_hash,
        score_scope=score_scope,
        evaluation_table_hash=prepared.hashes.get("evaluation_sha256"),
        protocol_hash=content_hash(tables.protocol),
        compatibility=reuse,
        score_source_versions=scores_manifest["training_versions"],
    )
    with _record(args.output, config, args, provenance) as recorder:
        thresholds = calibrate(
            calibration_scores,
            set(lists.calibration),
            config["calibration"]["sensitivity_quantiles"],
        )
        report = evaluate_run(
            tables,
            lists,
            scores,
            thresholds,
            score_scope=score_scope,
            source_binding=dict(
                versions=scores_manifest["training_versions"],
                lists_hash=str(scores.lists_hash.iloc[0]),
            ),
        )
        path = recorder.output / "evaluation/report.json"
        _json(path, report)
        predictions = pd.DataFrame(report["scores"])
        for quantile, threshold in thresholds["thresholds"].items():
            predictions[f"prediction_q{quantile}"] = predictions.score.gt(threshold)
        predictions["main_test"] = predictions.sample_id.isin(lists.main_test)
        predictions["diagnostic_pre"] = predictions.sample_id.isin(lists.diagnostic_pre)
        predictions["diagnostic_positive"] = predictions.sample_id.isin(lists.diagnostic_positive)
        prediction_path = recorder.output / "evaluation/predictions.parquet"
        predictions.to_parquet(prediction_path, index=False)
        _publish(
            recorder,
            {
                "evaluation_report": path,
                "thresholds": thresholds,
                "predictions": prediction_path,
                "sample_lists": asdict(lists),
            },
        )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ["check-inputs", "prepare", "probe-fit", "fit", "score", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument(
            "--tracking-uri", required=True, help="Absolute local SQLite or file URI"
        )
        if name in {"check-inputs", "prepare"}:
            command.add_argument("--preparation", type=Path, required=True)
            command.add_argument("--label-bundle", type=Path, required=True)
        else:
            command.add_argument("--prepared", type=Path, required=True)
        if name in {"probe-fit", "fit"}:
            command.add_argument("--method", choices=METHODS, default="transformer_cls")
            command.add_argument("--seed", type=int)
        if name in {"score", "evaluate"}:
            command.add_argument("--model-run", type=Path, required=True)
        if name == "score":
            command.add_argument("--score-scope", choices=SCOPES, default="test")
            command.add_argument("--reuse-scores", type=Path)
        if name == "evaluate":
            command.add_argument("--scores", type=Path, required=True)
            command.add_argument("--calibration-scores", type=Path, required=True)
    commands.add_parser("check-run").add_argument("--run", type=Path, required=True)
    return result


def main(argv=None, *, runtime_factory=None):
    args = parser().parse_args(argv)
    if args.command == "check-run":
        print(json.dumps(check_run(args.run), sort_keys=True))
        return 0
    if args.output.exists():
        raise FileExistsError(args.output)
    config = load_config(args.config)
    runtime = runtime_factory or LocalRuntime()
    if args.command in {"check-inputs", "prepare"}:
        _prepare(args, config, runtime)
    elif args.command in {"probe-fit", "fit"}:
        _fit(args, config, runtime)
    elif args.command == "score":
        _score(args, config, runtime)
    else:
        _evaluate(args, config)
    print(json.dumps(check_run(args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        LOGGER.exception("Service model stage failed: %s", error)
        raise SystemExit(1) from error
