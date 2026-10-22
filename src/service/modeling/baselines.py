"""Fixed normal-only baselines with frozen initial text representations."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_downstream import fit_iforest, fit_ocsvm, score_iforest, score_ocsvm

from .config import content_hash, validate_config
from .data import validate_sample_lists
from .training import _binding, _validate


def baseline_sample_lists(bundle, prepared, lists) -> dict:
    validate_sample_lists(lists)
    masks = prepared.numeric["masks"].set_index("sample_id")
    numeric_valid = set(masks.index[masks.any(axis=1)])
    text_valid = {
        sample_id
        for sample_id in prepared.sample_ids
        if bundle["text_fit_valid"] and bool(prepared.logs["input_ids"][sample_id])
    }
    result = {}
    for arm, supported in [("numeric", numeric_valid), ("trimodal", numeric_valid | text_valid)]:
        if not set(lists.fit) <= supported or not set(lists.calibration) <= supported:
            raise ValueError("Method support cannot change common fit/calibration membership")
        fields = {}
        for name in [
            "common_main_test",
            "extended_inference",
            "extended_main_test",
            "diagnostic_pre",
            "diagnostic_positive",
        ]:
            fields[name] = tuple(
                sample_id for sample_id in getattr(lists, name) if sample_id in supported
            )
        fields["diagnostic_groups"] = {
            name: dict(
                pre=tuple(sample_id for sample_id in group["pre"] if sample_id in supported),
                positive=tuple(
                    sample_id for sample_id in group["positive"] if sample_id in supported
                ),
                reference_available=any(sample_id in supported for sample_id in group["pre"]),
            )
            for name, group in lists.diagnostic_groups.items()
        }
        if fields["common_main_test"] != lists.common_main_test:
            raise ValueError("Baseline support cannot shrink the frozen common main test")
        fields["excluded"] = tuple(
            sorted(set(lists.excluded) | (set(prepared.sample_ids) - supported))
        )
        fields["reasons"] = copy.deepcopy(lists.reasons)
        for sample_id in set(prepared.sample_ids) - supported:
            fields["reasons"].setdefault(sample_id, []).append(f"{arm}_baseline_unsupported_input")
        payload = asdict(lists)
        payload.pop("lists_hash")
        payload.update(fields)
        fields["lists_hash"] = content_hash(payload)
        result[arm] = replace(lists, **fields)
    return result


def _numeric(prepared, ids, columns):
    values = prepared.numeric["values"].set_index("sample_id").loc[ids, columns].to_numpy(float)
    masks = prepared.numeric["masks"].set_index("sample_id").loc[ids, columns].to_numpy(bool)
    cleaned = np.where(masks, values, 0.0)
    if not np.isfinite(cleaned).all():
        raise ValueError("Nonfinite observed numeric baseline input")
    return cleaned, masks


def _pool(backbone, logs, sample_id):
    device = next(backbone.parameters()).device
    ids = torch.as_tensor(logs["input_ids"][sample_id], device=device)
    if not ids.numel():
        return None
    attention = torch.as_tensor(logs["attention_mask"][sample_id], device=device)
    special = torch.as_tensor(logs["special_mask"][sample_id], device=device)
    groups = logs["group_ids"][sample_id]
    if (
        ids.ndim != 2
        or ids.shape != attention.shape
        or ids.shape != special.shape
        or len(groups) != len(ids)
    ):
        raise ValueError("Invalid baseline text message shapes")
    body = attention.bool() & ~special.bool()
    if not body.any(-1).all():
        raise ValueError("Baseline message without body tokens")
    hidden = backbone(input_ids=ids, attention_mask=attention).last_hidden_state
    means = hidden.masked_fill(~body[:, :, None], 0).sum(1) / body.sum(1, keepdim=True)
    result = torch.stack(
        [
            means[[index for index, value in enumerate(groups) if value == group]].mean(0)
            for group in sorted(set(groups))
        ]
    ).mean(0)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite baseline text vector")
    return result.detach().cpu().numpy()


def _text(backbone, prepared, ids):
    with torch.no_grad():
        vectors = [_pool(backbone, prepared.logs, sample_id) for sample_id in ids]
    valid = np.array([value is not None for value in vectors], dtype=bool)
    values = (
        np.stack(
            [
                value if value is not None else np.zeros(backbone.config.hidden_size)
                for value in vectors
            ]
        )
        if ids
        else np.empty((0, backbone.config.hidden_size))
    )
    return values, valid


def _features(bundle, prepared, ids):
    values, masks = _numeric(prepared, ids, bundle["numeric_columns"])
    transform = bundle.get("numeric_transform", "identity")
    if transform not in {"identity", "asinh"}:
        raise ValueError("Unsupported baseline numeric transform")
    if transform == "asinh":
        values = np.arcsinh(values)
    numeric = np.concatenate([values, masks.astype(float)], axis=1)
    if bundle["backbone"] is None:
        return numeric, numeric, values, masks, np.zeros(len(ids), dtype=bool)
    text, text_valid = _text(bundle["backbone"], prepared, ids)
    text_valid &= bundle["text_fit_valid"]
    text = np.where(text_valid[:, None], (text - bundle["text_mean"]) / bundle["text_scale"], 0.0)
    statistics = prepared.logs.get("statistics")
    if (
        not isinstance(statistics, dict)
        or statistics.get("state_hash") != prepared.provenance["log_statistics_state_hash"]
    ):
        raise ValueError("Missing/incompatible normalized baseline log statistics")
    columns = bundle["statistics_columns"]
    if statistics["feature_order"] != columns or len(columns) != 4:
        raise ValueError("Baseline requires the same four log statistics")
    stats = statistics["values"].set_index("sample_id").loc[ids, columns].to_numpy(float)
    stat_masks = statistics["masks"].set_index("sample_id").loc[ids, columns].to_numpy(bool)
    stat_masks &= text_valid[:, None]
    stats = np.where(stat_masks, stats, 0.0)
    if transform == "asinh":
        stats = np.arcsinh(stats)
    trimodal = np.concatenate(
        [numeric, text, np.repeat(text_valid[:, None], text.shape[1], axis=1), stats, stat_masks],
        axis=1,
    )
    if not np.isfinite(trimodal).all():
        raise ValueError("Nonfinite baseline features")
    return numeric, trimodal, values, masks, text_valid


def fit_baselines(prepared, lists, config: dict, *, text_backbone=None, methods=None) -> dict:
    validate_config(config)
    _validate(prepared, lists)
    ids = sorted(lists.fit)
    supported = {"L2", "Max", "numeric_IF", "numeric_OCSVM", "trimodal_IF", "trimodal_OCSVM"}
    requested = supported if methods is None else set(methods)
    if not requested or requested - supported:
        raise ValueError("Unknown or empty baseline methods")
    needs_text = any(name.startswith("trimodal") for name in requested)
    backbone = text_backbone
    if needs_text and backbone is None:
        from transformers import AutoModel

        if not config["log"]["revision"]:
            raise ValueError("Baseline requires an explicit initial backbone revision")
        backbone = AutoModel.from_pretrained(
            config["log"]["model"], revision=config["log"]["revision"], local_files_only=True
        )
    digest = hashlib.sha256()
    if needs_text:
        backbone = copy.deepcopy(backbone).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        for name, tensor in backbone.state_dict().items():
            digest.update(name.encode())
            digest.update(
                tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
        text, valid = _text(backbone, prepared, ids)
    else:
        backbone = None
        text, valid = np.empty((len(ids), 0)), np.zeros(len(ids), dtype=bool)
    mean = text[valid].mean(0) if valid.any() else np.zeros(text.shape[1])
    scale = text[valid].std(0) if valid.any() else np.ones(text.shape[1])
    scale = np.where(scale > config["preprocessing"]["scale_epsilon"], scale, 1.0)
    statistics = prepared.logs.get("statistics", {})
    bundle = dict(
        backbone=backbone,
        text_mean=mean,
        text_scale=scale,
        text_fit_valid=bool(valid.any()),
        backbone_sha256=digest.hexdigest(),
        numeric_columns=list(prepared.numeric["values"].columns.drop("sample_id")),
        statistics_columns=statistics.get("feature_order"),
        binding=_binding(prepared, config),
        fit_ids=ids,
    )
    if "reconstruction_asinh" in config["model"]["auxiliary_losses"]:
        bundle["numeric_transform"] = "asinh"
    numeric, trimodal, _, masks, _ = _features(bundle, prepared, ids)
    if not masks.any(1).all():
        raise ValueError("Common fit requires valid numeric input")
    seed = config["training"]["prototype_seed"]
    models = {}
    for arm, values in [("numeric", numeric), ("trimodal", trimodal)]:
        if f"{arm}_IF" in requested:
            models[f"{arm}_IF"] = fit_iforest(values, seed)
        if f"{arm}_OCSVM" in requested:
            models[f"{arm}_OCSVM"] = fit_ocsvm(values, seed)
    bundle["models"] = models
    bundle["method_lists"] = baseline_sample_lists(bundle, prepared, lists)
    bundle["resolved_parameters"] = {name: model.get_params() for name, model in models.items()}
    bundle["baseline_sha256"] = content_hash(
        dict(
            binding=bundle["binding"],
            backbone_sha256=bundle["backbone_sha256"],
            fit_ids=ids,
            parameters=bundle["resolved_parameters"],
            text_mean=mean.tolist(),
            text_scale=scale.tolist(),
            numeric_columns=bundle["numeric_columns"],
            statistics_columns=bundle["statistics_columns"],
            numeric_transform=bundle.get("numeric_transform", "identity"),
            method_lists={arm: asdict(members) for arm, members in bundle["method_lists"].items()},
        )
    )
    return bundle


def score_baselines(bundle: dict, prepared, ids: set[str]) -> dict[str, pd.DataFrame]:
    _validate(prepared)
    if not isinstance(ids, set) or ids - set(prepared.sample_ids):
        raise ValueError("Invalid baseline scoring IDs")
    if any(bundle["binding"].get(key) != value for key, value in _binding(prepared).items()):
        raise ValueError("Baseline input binding mismatch")
    ids = sorted(ids)
    numeric, trimodal, values, masks, text_valid = _features(bundle, prepared, ids)
    numeric_valid = masks.any(1)
    trimodal_valid = numeric_valid | text_valid
    outputs = {}
    for name in ["L2", "Max", *bundle["models"]]:
        arm = "trimodal" if name.startswith("trimodal") else "numeric"
        members = bundle["method_lists"][arm]
        validate_sample_lists(members)
        valid = trimodal_valid if name.startswith("trimodal") else numeric_valid
        selected = np.flatnonzero(valid)
        if name == "L2":
            scores = np.linalg.norm(values[selected], axis=1)
        elif name == "Max":
            scores = np.abs(values[selected]).max(1) if len(selected) else np.array([])
        else:
            features = trimodal if name.startswith("trimodal") else numeric
            scorer = score_iforest if name.endswith("_IF") else score_ocsvm
            scores = (
                scorer(bundle["models"][name], features[selected])
                if len(selected)
                else np.array([])
            )
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite baseline score")
        frame = pd.DataFrame(
            dict(
                sample_id=[ids[index] for index in selected],
                score=scores,
                valid_dimensions=masks[selected].sum(1),
            )
        )
        frame = frame.merge(prepared.identity, on="sample_id", validate="one_to_one")
        frame["model_version"] = f"service-baseline-v1:{name}"
        for key, value in prepared.versions.items():
            frame[key] = value
        frame["lists_hash"] = members.lists_hash
        frame["baseline_sha256"] = bundle["baseline_sha256"]
        frame = frame.merge(
            prepared.labels[["sample_id", "label"]], on="sample_id", validate="one_to_one"
        )
        frame = frame.merge(
            prepared.splits[["sample_id", "split"]], on="sample_id", validate="one_to_one"
        )
        frame.attrs.update(bundle["binding"])
        frame.attrs["lists_hash"] = members.lists_hash
        frame.attrs["model_version"] = f"service-baseline-v1:{name}"
        frame.attrs.update(prepared.versions)
        frame.attrs["excluded_ids"] = [
            sample_id for index, sample_id in enumerate(ids) if not valid[index]
        ]
        frame.attrs["score_direction"] = "higher_is_anomalous"
        outputs[name] = frame
    return outputs
