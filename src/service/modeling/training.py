"""Window-weighted, fit-only SVDD optimization and deterministic diagnostics."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed

from .config import content_hash, validate_config
from .data import PreparedData, SampleLists
from .model import initialize_center

LOGGER = logging.getLogger(__name__)
RESOURCE = ["cpu_cores", "memory_usage_ratio", "process_count"]
EXECUTION = [
    "observation_count",
    "duration_mean_ms",
    "duration_p95_ms",
    "duration_max_ms",
    "error_rate",
]
MODEL_VERSION = "service-model-v1"


def _validate(prepared, lists=None):
    ids = set(prepared.sample_ids)
    for name in ["identity", "labels", "splits"]:
        frame = getattr(prepared, name, None)
        if (
            not isinstance(frame, pd.DataFrame)
            or "sample_id" not in frame
            or frame.sample_id.duplicated().any()
            or set(frame.sample_id) != ids
        ):
            raise ValueError(f"Missing or inconsistent prepared {name} metadata")
    required = {"case_id", "canonical_service_id", "window_start_ms"}
    if required - set(prepared.identity):
        raise ValueError("Missing stable identity keys")
    if (
        not {"label_version", "split_version", "evaluation_version"} <= set(prepared.versions)
        or not prepared.hashes
    ):
        raise ValueError("Missing input versions/hashes")
    references = {"numeric_state_hash", "log_state_hash", "log_statistics_state_hash"}
    if not references <= set(prepared.provenance) or any(
        not prepared.provenance[key] for key in references
    ):
        raise ValueError("Missing preprocessor references/hashes")
    if prepared.logs.get("cache", {}).get("state_hash") != prepared.provenance["log_state_hash"]:
        raise ValueError("Log preprocessor hash mismatch")
    if lists is not None:
        if lists.lists_hash != prepared.lists.lists_hash or tuple(lists.fit) != tuple(
            prepared.lists.fit
        ):
            raise ValueError("Frozen membership mismatch")
        if not lists.fit or len(set(lists.fit)) != len(lists.fit) or set(lists.fit) - ids:
            raise ValueError("Empty/invalid fit membership")
        labels = prepared.labels.set_index("sample_id").label.reindex(lists.fit)
        splits = prepared.splits.set_index("sample_id").split.reindex(lists.fit)
        if not labels.eq(0).fillna(False).all() or not splits.eq("fit").all():
            raise ValueError("Training requires known-normal fit membership")


class _Windows:
    def __init__(self, prepared, device):
        self.prepared = prepared
        self.device = device
        self.values = prepared.numeric["values"].set_index("sample_id")
        self.masks = prepared.numeric["masks"].set_index("sample_id")
        statistics = prepared.logs.get("statistics")
        if not isinstance(statistics, dict):
            raise ValueError("Missing fit-normalized log statistics")
        self.statistics = statistics["values"].set_index("sample_id")
        self.statistics_masks = statistics["masks"].set_index("sample_id")
        self.statistics_order = statistics["feature_order"]
        if len(self.statistics_order) != 4:
            raise ValueError("Expected four log statistics")
        if statistics["state_hash"] != prepared.provenance["log_statistics_state_hash"]:
            raise ValueError("Log statistics preprocessor hash mismatch")

    def __call__(self, sample_id):
        def tensor(frame, names, dtype):
            return torch.as_tensor(
                frame.loc[[sample_id], names].to_numpy(
                    dtype=bool if dtype == torch.bool else float
                ),
                dtype=dtype,
                device=self.device,
            )

        logs = self.prepared.logs
        messages = None
        if logs["input_ids"][sample_id]:
            messages = {
                name: torch.as_tensor(logs[name][sample_id], device=self.device)
                for name in ["input_ids", "attention_mask", "special_mask"]
            }
            messages["group_ids"] = logs["group_ids"][sample_id]
        return dict(
            sample_ids=[sample_id],
            resource=tensor(self.values, RESOURCE, torch.float32),
            resource_mask=tensor(self.masks, RESOURCE, torch.bool),
            execution=tensor(self.values, EXECUTION, torch.float32),
            execution_mask=tensor(self.masks, EXECUTION, torch.bool),
            logs=[messages],
            stats=tensor(self.statistics, self.statistics_order, torch.float32),
            stats_mask=tensor(self.statistics_masks, self.statistics_order, torch.bool),
        )


def _binding(prepared, config=None):
    result = dict(
        model_version=MODEL_VERSION,
        versions=prepared.versions,
        input_hashes=prepared.hashes,
        preprocessor=prepared.provenance,
        lists_hash=prepared.lists.lists_hash,
    )
    if config is not None:
        result["config_sha256"] = config["config_sha256"]
    return copy.deepcopy(result)


def _finite(value, name):
    if not torch.isfinite(value).all():
        raise ValueError(f"Nonfinite {name}")


def _diagnostic(model, windows, ids, baseline, config):
    was_training = model.training
    model.eval()
    representations, scores = [], []
    try:
        with torch.no_grad():
            for sample_id in ids:
                result = model(windows(sample_id))
                _finite(result["representation"], "diagnostic representation")
                _finite(result["score"], "diagnostic score")
                representations.append(result["representation"].detach().double().cpu())
                scores.append(result["score"].detach().double().cpu())
        representation = torch.cat(representations)
        score = torch.cat(scores)
        variance = representation.var(0, unbiased=False)
        total = float(variance.sum())
        settings = config["diagnostics"]
        degenerate = total <= settings["variance_absolute_floor"] or (
            baseline is not None and total < baseline * settings["variance_relative_floor"]
        )
        if degenerate:
            LOGGER.warning("Fit representation degeneracy warning: total variance=%s", total)
        return dict(
            dimension_variance=variance.tolist(),
            total_variance=total,
            center_norm=float(model.center.double().norm()),
            representation_norm_mean=float(representation.norm(dim=1).mean()),
            score_quantiles=dict(
                zip(
                    ["0", "0.25", "0.5", "0.75", "0.95", "0.99", "1"],
                    torch.quantile(
                        score,
                        torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0], dtype=torch.float64),
                    ).tolist(),
                )
            ),
            degenerate=degenerate,
            mode="eval",
        )
    finally:
        model.train(was_training)


def _optimizer(model, settings):
    if (
        settings["optimizer"] != "AdamW"
        or settings["scheduler"] is not None
        or settings["checkpoint"] != "last"
        or settings["micro_batch_windows"] != 1
        or settings["accumulation_windows"] != 16
    ):
        raise ValueError("Unsupported training protocol")
    if settings["precision"] not in {None, "float32"}:
        raise ValueError("Precision must be explicitly implemented after hardware preflight")
    backbone = list(model.text_encoder.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    other = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("All model parameters must remain trainable")
    return torch.optim.AdamW(
        [
            dict(params=backbone, lr=settings["bert_lr"]),
            dict(params=other, lr=settings["other_lr"]),
        ],
        weight_decay=settings["weight_decay"],
    )


def _backward_group(model, windows, ids, report=None):
    total = 0.0
    for sample_id in ids:
        result = model(windows(sample_id))
        auxiliary = result.get("aux_losses", {})
        if auxiliary and set(auxiliary) != {"reconstruction"}:
            raise ValueError("Unsupported auxiliary objectives")
        loss = auxiliary["reconstruction"] if auxiliary else result["score"].mean()
        _finite(loss, "training loss")
        (loss / len(ids)).backward()
        if report is not None:
            report["backward_calls"] += 1
        total += float(loss.detach())
    for parameter in model.parameters():
        if parameter.grad is not None:
            _finite(parameter.grad, "gradient")
    return total


def _write_progress(output: Path, progress: dict) -> None:
    temporary = output / "progress.json.tmp"
    temporary.write_text(json.dumps(progress, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "progress.json")


def train_model(
    model, prepared: PreparedData, lists: SampleLists, config: dict, output: Path
) -> dict:
    validate_config(config)
    _validate(prepared, lists)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )
    settings = config["training"]
    seed = settings["prototype_seed"]
    result = dict(
        status="running",
        seed=seed,
        checkpoint=None,
        optimizer_steps=0,
        diagnostics=[],
        **_binding(prepared, config),
    )
    started = time.perf_counter()
    last_progress = started
    progress = dict(
        stage="center_initialization",
        epoch=0,
        epochs=settings["epochs"],
        windows_completed_in_epoch=0,
        fit_windows=len(lists.fit),
        optimizer_steps=0,
        elapsed_seconds=0.0,
        windows_per_second=None,
        eta_training_seconds=None,
    )

    def publish(stage: str) -> None:
        nonlocal last_progress
        last_progress = time.perf_counter()
        progress.update(
            stage=stage,
            optimizer_steps=result["optimizer_steps"],
            elapsed_seconds=last_progress - started,
        )
        _write_progress(output, progress)

    try:
        publish("center_initialization")
        LOGGER.info("Center initialization started: fit_windows=%s", len(lists.fit))
        set_seed(seed)
        device = torch.device(settings["device"] or "cpu")
        model.to(device)
        optimizer = _optimizer(model, settings)
        if type(settings["epochs"]) is not int or settings["epochs"] < 1:
            raise ValueError("Epochs must be positive")
        windows = _Windows(prepared, device)
        fit_ids = sorted(lists.fit)
        identities = prepared.identity.set_index("sample_id").loc[fit_ids]
        result["fit_contribution"] = {
            name: {str(key): int(value) for key, value in identities[name].value_counts().items()}
            for name in ["case_id", "canonical_service_id"]
        }
        limit = min(256, config["diagnostics"]["max_fit_windows"], len(fit_ids))
        if limit < 1:
            raise ValueError("Empty diagnostic sample")
        diagnostic_ids = [
            fit_ids[index] for index in np.linspace(0, len(fit_ids) - 1, limit, dtype=int)
        ]
        result["diagnostic_ids"] = diagnostic_ids
        result["diagnostic_ids_hash"] = content_hash(dict(ids=diagnostic_ids))
        initialize_center(model, (windows(sample_id) for sample_id in fit_ids))
        _finite(model.center, "center")
        LOGGER.info("Center initialization completed")
        publish("initial_diagnostic")
        initial = _diagnostic(model, windows, diagnostic_ids, None, config)
        result["diagnostics"].append(dict(epoch=0, **initial))
        for epoch in range(1, settings["epochs"] + 1):
            epoch_started = time.perf_counter()
            progress.update(
                epoch=epoch,
                windows_completed_in_epoch=0,
                windows_per_second=None,
                eta_training_seconds=None,
            )
            publish("training")
            LOGGER.info(
                "Epoch %s/%s started: fit_windows=%s", epoch, settings["epochs"], len(fit_ids)
            )
            model.train()
            ordered = [fit_ids[index] for index in torch.randperm(len(fit_ids)).tolist()]
            loss_sum = 0.0
            for start in range(0, len(ordered), 16):
                optimizer.zero_grad(set_to_none=True)
                loss_sum += _backward_group(model, windows, ordered[start : start + 16])
                optimizer.step()
                for parameter in model.parameters():
                    _finite(parameter, "updated parameter")
                result["optimizer_steps"] += 1
                completed = min(start + 16, len(ordered))
                elapsed = time.perf_counter() - epoch_started
                rate = completed / elapsed if elapsed > 0 else None
                progress.update(
                    windows_completed_in_epoch=completed,
                    windows_per_second=rate,
                    eta_training_seconds=(
                        ((settings["epochs"] - epoch) * len(fit_ids) + len(fit_ids) - completed)
                        / rate
                        if rate
                        else None
                    ),
                )
                if result["optimizer_steps"] % 10 == 0 or time.perf_counter() - last_progress >= 30:
                    publish("training")
            publish("epoch_diagnostic")
            diagnostic = _diagnostic(
                model, windows, diagnostic_ids, initial["total_variance"], config
            )
            result["diagnostics"].append(
                dict(epoch=epoch, mean_loss=loss_sum / len(fit_ids), **diagnostic)
            )
            LOGGER.info(
                "Epoch %s/%s completed: mean_loss=%s optimizer_steps=%s elapsed_seconds=%.3f",
                epoch,
                settings["epochs"],
                loss_sum / len(fit_ids),
                result["optimizer_steps"],
                time.perf_counter() - epoch_started,
            )
        optimizer.zero_grad(set_to_none=True)
        checkpoint = output / "checkpoint_last.pt"
        state = dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            center=model.center.detach().cpu(),
            epoch=settings["epochs"],
            seed=seed,
            selection="last",
            config=config,
            fit_ids=fit_ids,
            fit_ids_hash=content_hash(dict(ids=fit_ids)),
            diagnostic_ids=diagnostic_ids,
            **_binding(prepared, config),
        )
        torch.save(state, checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        model.training_identity = dict(
            **_binding(prepared, config), checkpoint_sha256=checkpoint_hash
        )
        result.update(
            status="completed", checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash
        )
        progress["eta_training_seconds"] = 0.0
        publish("completed")
    except (ValueError, RuntimeError, FloatingPointError) as error:
        LOGGER.exception("Training failed for seed %s", seed)
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        progress.update(
            error_type=type(error).__name__, error=str(error), eta_training_seconds=None
        )
        publish("failed")
    (output / "training_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def score_model(model, prepared, ids: set[str]) -> pd.DataFrame:
    _validate(prepared)
    if set(ids) - set(prepared.sample_ids):
        raise ValueError("Unknown scoring sample IDs")
    if not bool(model.center_initialized):
        raise ValueError("Scoring requires initialized center")
    binding = getattr(model, "training_identity", None)
    if binding is None or any(binding[key] != value for key, value in _binding(prepared).items()):
        raise ValueError("Missing/incompatible training identity; restore checkpoint binding")
    device = next(model.parameters()).device
    windows = _Windows(prepared, device)
    was_training = model.training
    rows = []
    model.eval()
    try:
        with torch.no_grad():
            for sample_id in sorted(ids):
                score = model(windows(sample_id))["score"]
                _finite(score, "inference score")
                rows.append(dict(sample_id=sample_id, score=float(score.item())))
    finally:
        model.train(was_training)
    frame = pd.DataFrame(rows, columns=["sample_id", "score"]).merge(
        prepared.identity, on="sample_id", how="left", validate="one_to_one"
    )
    frame["model_version"] = MODEL_VERSION
    for name, value in prepared.versions.items():
        frame[name] = value
    for name in ["lists_hash", "config_sha256", "checkpoint_sha256"]:
        frame[name] = binding[name]
    frame.attrs.update(binding)
    return frame


def _cpu_snapshot(tensor):
    return tensor.detach().to(device="cpu", copy=True)


def probe_fit(model, prepared, lists: SampleLists, config: dict) -> dict:
    validate_config(config)
    _validate(prepared, lists)
    set_seed(config["training"]["prototype_seed"])
    original_mode = model.training
    device = next(model.parameters()).device
    original = None
    gradients = None
    mutated = False
    start = time.perf_counter()
    ids = sorted(lists.fit)[:16]
    report = dict(
        status="running",
        windows=len(ids),
        sample_ids=ids,
        device=str(device),
        finite_gradients=False,
        backward_calls=0,
        accumulation_windows=config["training"]["accumulation_windows"],
        log_window_count=sum(bool(prepared.logs["input_ids"][sample_id]) for sample_id in ids),
        backbone_gradient_parameter_count=0,
        backbone_gradient_norm=None,
        backbone_finite_gradients=None,
        backbone_gradient_status="unavailable",
        backbone_gradient_reason="probe_not_completed",
        peak_memory_bytes=0,
        restoration_status="not_required",
    )
    try:
        original = {name: _cpu_snapshot(value) for name, value in model.state_dict().items()}
        gradients = [
            None if parameter.grad is None else _cpu_snapshot(parameter.grad)
            for parameter in model.parameters()
        ]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        windows = _Windows(prepared, device)
        mutated = True
        if not model.center_initialized:
            initialize_center(model, (windows(sample_id) for sample_id in ids))
        model.train()
        model.zero_grad(set_to_none=True)
        _backward_group(model, windows, ids, report)
        backbone_gradients = [
            parameter.grad.detach()
            for parameter in model.text_encoder.backbone.parameters()
            if parameter.grad is not None
        ]
        report["backbone_gradient_parameter_count"] = len(backbone_gradients)
        if report["log_window_count"]:
            backbone_norm = (
                float(
                    torch.stack(
                        [gradient.double().norm() for gradient in backbone_gradients]
                    ).norm()
                )
                if backbone_gradients
                else 0.0
            )
            backbone_finite = bool(backbone_gradients) and all(
                bool(torch.isfinite(gradient).all()) for gradient in backbone_gradients
            )
            report.update(
                backbone_gradient_norm=backbone_norm,
                backbone_finite_gradients=backbone_finite,
                backbone_gradient_status="failed",
                backbone_gradient_reason="missing_or_zero_gradient",
            )
            if not backbone_finite or not np.isfinite(backbone_norm) or backbone_norm <= 0:
                raise ValueError("Log windows require finite nonzero backbone gradients")
            report.update(backbone_gradient_status="passed", backbone_gradient_reason=None)
        else:
            report["backbone_gradient_reason"] = "no_log_windows"
        report.update(
            status="completed",
            finite_gradients=True,
            gradient_norm=float(
                torch.stack(
                    [
                        parameter.grad.detach().norm()
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]
                ).norm()
            ),
        )
    except (ValueError, RuntimeError, FloatingPointError, MemoryError) as error:
        LOGGER.exception("Fit probe failed")
        report.update(status="failed", error=str(error), error_type=type(error).__name__)
    finally:
        try:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                report["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        except RuntimeError as error:
            LOGGER.exception("Fit probe memory measurement failed")
            report.update(status="failed", memory_measurement_error=str(error))
        report["elapsed_seconds"] = time.perf_counter() - start
        if mutated:
            try:
                model.zero_grad(set_to_none=True)
                model.load_state_dict(original)
                for parameter, gradient in zip(model.parameters(), gradients):
                    parameter.grad = None if gradient is None else gradient.to(parameter.device)
                report["restoration_status"] = "restored"
            except (ValueError, RuntimeError, MemoryError) as error:
                LOGGER.exception("Fit probe state restoration failed")
                report.update(
                    status="failed", restoration_status="failed", restoration_error=str(error)
                )
            finally:
                model.train(original_mode)
    return report
