"""Local, auditable stage recording and conservative artifact reuse decisions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import mlflow

from src.utils.logger import ExperimentLogger


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(_json(value)).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json(value))
    os.replace(temporary, path)


def check_reuse(previous: dict, current: dict) -> dict:
    fit_keys = {
        "fit_ids_hash",
        "fit_labels_hash",
        "fit_eligibility_hash",
        "fit_input_hash",
        "feature_hash",
        "feature_order",
        "mask_hash",
        "template_hash",
        "preprocessor_hash",
        "model_config_hash",
        "model_revision",
        "seed",
        "split_hash",
        "split_version",
        "input_hash",
        "quality_rule_hash",
    }
    cal_keys = {
        "calibration_ids_hash",
        "calibration_labels_hash",
        "calibration_eligibility_hash",
        "calibration_input_hash",
        "calibration_mask_hash",
        "calibration_config_hash",
    }
    changed = sorted(
        key for key in set(previous) | set(current) if previous.get(key) != current.get(key)
    )
    evaluation_keys = {
        "label_version",
        "evaluation_version",
        "evaluation_table_hash",
        "evaluation_manifest_hash",
        "test_labels_hash",
        "main_test_hash",
        "diagnostic_lists_hash",
        "required_score_ids",
        "scored_ids",
        "versions",
    }
    unknown = set(changed) - fit_keys - cal_keys - evaluation_keys
    refit = bool(set(changed) & fit_keys or unknown)
    if "versions" in changed:
        before, after = previous.get("versions", {}), current.get("versions", {})
        refit |= any(
            before.get(key) != after.get(key)
            for key in set(before) | set(after)
            if key not in {"label_version", "evaluation_version"}
        )
    recalibrate = refit or bool(set(changed) & cal_keys)
    return {
        "refit": refit,
        "recalibrate": recalibrate,
        "reevaluate": bool(changed),
        "rescore_ids": sorted(
            set(current.get("required_score_ids", [])) - set(previous.get("scored_ids", []))
        ),
        "reasons": changed,
        "compatible": not refit and not recalibrate,
    }


class RunRecorder:
    def __init__(self, output: Path, config: dict, provenance: dict):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError(self.output)
        if {"raw", "data"} & set(self.output.parts) or self.output.parts[-2:] == (
            "outputs",
            "service",
        ):
            raise ValueError("Output must be a new run directory outside raw data")
        self.config = json.loads(_json(config))
        self.provenance = json.loads(_json(provenance))
        self.run_id = self.output.name
        self.mlflow_run_id = None
        self._active = None
        self._manifest = None
        self._entered = False
        self._published = False
        uri = config.get("tracking_uri") or config.get("tracking", {}).get("uri")
        if not uri:
            raise ValueError("Explicit local tracking_uri is required")
        uri = str(uri).replace("\\", "/")
        parsed = urlparse(uri)
        if parsed.scheme == "sqlite":
            sqlite_path = str(uri)[len("sqlite:///") :] if str(uri).startswith("sqlite:///") else ""
            if (
                not sqlite_path
                or parsed.query
                or parsed.netloc
                or (not sqlite_path.startswith("/") and not re.match(r"^[A-Za-z]:/", sqlite_path))
            ):
                raise ValueError("Tracking requires an absolute local SQLite URI")
        elif parsed.scheme == "file":
            if parsed.netloc or not Path(parsed.path).is_absolute():
                raise ValueError("Tracking requires an absolute local URI")
        elif parsed.scheme or not Path(uri).is_absolute():
            raise ValueError("Tracking requires an absolute local URI")
        self.uri = str(uri)

    def _status(self, status, error=None):
        _write(
            self.output / "status.json",
            {
                "status": status,
                "run_id": self.run_id,
                "mlflow_run_id": self.mlflow_run_id,
                "stage": self.provenance.get("stage"),
                "error": error,
            },
        )

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            for name in ["data", "run", "evaluation", "report"]:
                (self.output / name).mkdir()
            _write(self.output / "run" / "config.json", self.config)
            _write(self.output / "run" / "provenance.json", self.provenance)
            self._status("running")
            experiment_name = self.config.get("experiment_name", "service-model")
            mlflow.set_tracking_uri(self.uri)
            client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
            if client.get_experiment_by_name(experiment_name) is None:
                parsed_uri = urlparse(self.uri)
                if parsed_uri.scheme == "sqlite":
                    tracking_path = Path(self.uri[len("sqlite:///") :])
                    artifact_root = tracking_path.parent / (tracking_path.stem + "-artifacts")
                elif parsed_uri.scheme == "file":
                    artifact_root = Path(parsed_uri.path) / "artifacts"
                else:
                    artifact_root = Path(self.uri) / "artifacts"
                client.create_experiment(
                    experiment_name, artifact_location=artifact_root.resolve().as_uri()
                )
            self.logger = ExperimentLogger(experiment_name, self.uri)
            model = self.provenance.get("model_manifest", {})
            if self.provenance.get("stage") == "evaluate":
                if not model.get("mlflow_run_id") or not self.provenance.get("evaluation_id"):
                    raise ValueError("Evaluation requires a training manifest and evaluation_id")
            self._active = self.logger.start_run(run_name=self.run_id)
            active = self._active.__enter__()
            self.mlflow_run_id = active.info.run_id
            self._status("running")
            params = {
                "config_sha256": _hash(self.config),
                "provenance_sha256": _hash(self.provenance),
            }
            if model:
                params["training_run_id"] = model.get("mlflow_run_id", "")
                params["training_mlflow_run_id"] = model["mlflow_run_id"]
                params["training_local_run_id"] = model["run_id"]
                mlflow.set_tag("training_mlflow_run_id", model["mlflow_run_id"])
                mlflow.set_tag("training_local_run_id", model["run_id"])
            self.logger.log_params(params)
            self._entered = True
            return self
        except BaseException as error:
            self._status("failed", f"{type(error).__name__}: {error}")
            if self._active is not None:
                self._active.__exit__(type(error), error, error.__traceback__)
            raise

    def log_epoch(self, metrics: dict, step: int):
        if not self._entered or self._published:
            raise RuntimeError("Run is not accepting metrics")
        finite = {key: float(value) for key, value in metrics.items() if value is not None}
        if not all(math.isfinite(value) for value in finite.values()):
            raise ValueError("Non-finite tracking metric")
        self.logger.log_metrics(finite, step=step)
        _write(self.output / "run" / f"epoch-{step}.json", {"metrics": metrics, "step": step})

    def publish(self, artifacts: dict, *, local_only: set[str] | None = None) -> dict:
        if not self._entered or self._published:
            raise RuntimeError("Run may publish exactly once within its context")
        self._published = True
        local_only = set(local_only or ())
        if local_only - set(artifacts):
            raise ValueError("Unknown local-only artifact names")
        records = {}
        stage = self.provenance.get("stage", "fit")
        for name, value in artifacts.items():
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid artifact name")
            group = "evaluation" if stage == "evaluate" else "run"
            if "report" in name:
                group = "report"
            elif "list" in name or "preprocess" in name:
                group = "data"
            if isinstance(value, Path):
                source = value.resolve()
                if not source.is_relative_to(self.output):
                    raise ValueError("Artifact files must be generated inside this run directory")
                path = source
            else:
                path = self.output / group / f"{name}.json"
                _write(path, value)
            records[name] = {
                "path": str(path.relative_to(self.output)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            if name in local_only:
                records[name]["local_only"] = True
        model = self.provenance.get("model_manifest", {})
        self._manifest = {
            "schema_version": "service-run-v1",
            "status": "success",
            "stage": stage,
            "run_id": self.run_id,
            "mlflow_run_id": self.mlflow_run_id,
            "tracking_uri": self.uri,
            "config_sha256": _hash(self.config),
            "provenance": self.provenance,
            "artifacts": records,
            "training_versions": model.get(
                "training_versions", self.provenance.get("versions", {})
            ),
            "evaluation_versions": (
                self.provenance.get("versions", {}) if stage == "evaluate" else {}
            ),
            "training_run_id": model.get("run_id", self.run_id),
            "training_manifest_sha256": _hash(model) if model else None,
            "evaluation_id": self.provenance.get("evaluation_id"),
        }
        self._manifest["training_local_run_id"] = model.get("run_id", self.run_id)
        self._manifest["training_mlflow_run_id"] = model.get("mlflow_run_id", self.mlflow_run_id)
        return self._manifest

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self._status("failed", f"{error_type.__name__}: {error}")
            (self.output / "manifest.json").unlink(missing_ok=True)
            if self._active is not None:
                self._active.__exit__(error_type, error, traceback)
            self._entered = False
            return False
        try:
            if self._manifest is None:
                raise RuntimeError("Run exited without publishing artifacts")
            pending = self.output / "run" / "manifest.pending.json"
            _write(pending, self._manifest)
            prefix = (
                f"evaluation/{self.provenance['evaluation_id']}"
                if self.provenance.get("stage") == "evaluate"
                else "training"
            )
            for record in self._manifest["artifacts"].values():
                if not record.get("local_only", False):
                    self.logger.log_artifact(
                        str(self.output / record["path"]), artifact_path=prefix
                    )
            self.logger.log_artifact(str(pending), artifact_path=prefix)
            mlflow.set_tag(f"{prefix.replace('/', '.')}.manifest_sha256", _hash(self._manifest))
            mlflow.set_tag(f"{prefix.replace('/', '.')}.output", str(self.output))
            if self.provenance.get("stage") == "evaluate":
                client = mlflow.tracking.MlflowClient(tracking_uri=self.uri)
                training_id = self._manifest["training_mlflow_run_id"]
                tag_prefix = prefix.replace("/", ".")
                client.set_tag(training_id, f"{tag_prefix}.mlflow_run_id", self.mlflow_run_id)
                client.set_tag(training_id, f"{tag_prefix}.manifest_sha256", _hash(self._manifest))
                client.set_tag(training_id, f"{tag_prefix}.output", str(self.output))
            self._active.__exit__(None, None, None)
            self._active = None
            _write(self.output / "manifest.json", self._manifest)
            self._status("success")
        except BaseException as failure:
            (self.output / "manifest.json").unlink(missing_ok=True)
            self._status("failed", f"{type(failure).__name__}: {failure}")
            if self._active is not None:
                self._active.__exit__(type(failure), failure, failure.__traceback__)
            if error is None:
                raise
        finally:
            self._entered = False
        return False
