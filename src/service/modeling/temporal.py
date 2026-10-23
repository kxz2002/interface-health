"""Normal-only causal forecasting with explicit modality availability targets."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.utils.seed import set_seed

from .availability import availability_prior_logits
from .training import EXECUTION, RESOURCE

LOGGER = logging.getLogger(__name__)


def temporal_inputs(
    prepared,
    context_windows: int = 4,
    *,
    availability_prior=None,
    text_embeddings=None,
    availability_prior_scope="per_service",
):
    if type(context_windows) is not int or context_windows < 1:
        raise ValueError("Temporal context must have a positive integer length")
    ids = sorted(prepared.sample_ids)
    if text_embeddings is not None:
        text_embeddings = np.asarray(text_embeddings, dtype=np.float32)
        if (
            text_embeddings.ndim != 2
            or text_embeddings.shape[0] != len(ids)
            or not text_embeddings.shape[1]
        ):
            raise ValueError("Temporal text embeddings must have one vector per sample")
        if not np.isfinite(text_embeddings).all():
            raise ValueError("Temporal text embeddings must be finite")
        text_lookup = {sample_id: text_embeddings[index] for index, sample_id in enumerate(ids)}
        text_dim = text_embeddings.shape[1]
    else:
        text_lookup = {}
        text_dim = 0
    identity = prepared.identity.set_index("sample_id").loc[ids]
    splits = prepared.splits.set_index("sample_id").split.loc[ids]
    if identity.index.duplicated().any() or splits.isna().any():
        raise ValueError("Invalid temporal input identity or partition")
    numeric = prepared.numeric["values"].set_index("sample_id").loc[ids, RESOURCE + EXECUTION]
    masks = prepared.numeric["masks"].set_index("sample_id").loc[ids, RESOURCE + EXECUTION]
    statistics = prepared.logs["statistics"]
    if statistics["state_hash"] != prepared.provenance["log_statistics_state_hash"]:
        raise ValueError("Temporal log statistics binding mismatch")
    columns = statistics["feature_order"]
    if len(columns) != 4:
        raise ValueError("Temporal predictor requires four fixed log statistics")
    stats = statistics["values"].set_index("sample_id").loc[ids, columns]
    stat_mask_frame = statistics["masks"].set_index("sample_id").loc[ids, columns]
    for frame in [masks, stat_mask_frame]:
        if any(
            not pd.api.types.is_bool_dtype(frame[column]) or frame[column].isna().any()
            for column in frame
        ):
            raise ValueError("Temporal field masks must be nonnull booleans")
    stat_masks = stat_mask_frame.to_numpy(bool)
    log_mask = np.asarray([prepared.logs["masks"][sample_id] for sample_id in ids])
    if log_mask.dtype != bool:
        raise ValueError("Temporal log availability must be boolean")
    observed = np.concatenate([masks.to_numpy(bool), stat_masks & log_mask[:, None]], axis=1)
    values = np.concatenate([numeric.to_numpy(float), stats.to_numpy(float)], axis=1)
    values = np.where(observed, values, 0.0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite observed temporal values")
    values = np.arcsinh(values).astype(np.float32)
    availability = np.stack(
        [masks[RESOURCE].any(axis=1), masks[EXECUTION].any(axis=1), log_mask], axis=1
    ).astype(np.float32)
    features = np.concatenate([values, observed.astype(np.float32), availability], axis=1)
    keys = [
        (row.case_id, row.canonical_service_id, partition, int(row.window_start_ms))
        for row, partition in zip(identity.itertuples(), splits)
    ]
    lookup = {key: index for index, key in enumerate(keys)}
    if len(lookup) != len(ids):
        raise ValueError("Duplicate temporal case/service/partition/window key")
    fit_ids = set(prepared.lists.fit)
    context = np.zeros((len(ids), context_windows, features.shape[1] + 1), dtype=np.float32)
    text_context = (
        np.zeros((len(ids), context_windows, text_dim), dtype=np.float32) if text_dim else None
    )
    history_ids = {}
    for index, (sample_id, key) in enumerate(zip(ids, keys)):
        case, service, partition, timestamp = key
        history = []
        for lag in range(1, context_windows + 1):
            previous = lookup.get((case, service, partition, timestamp - lag * 15000))
            if previous is None or (partition == "fit" and ids[previous] not in fit_ids):
                break
            context[index, context_windows - lag, :-1] = features[previous]
            context[index, context_windows - lag, -1] = 1.0
            if text_context is not None:
                text_context[index, context_windows - lag] = text_lookup[ids[previous]]
            history.append(ids[previous])
        history_ids[sample_id] = list(reversed(history))
    return dict(
        ids=ids,
        identity=identity.reset_index(),
        context=context,
        values=values,
        observed=observed,
        availability=availability,
        features=features,
        history_ids=history_ids,
        feature_order=RESOURCE + EXECUTION + columns,
        prior_logits=(
            availability_prior_logits(
                availability_prior, prepared, ids, scope=availability_prior_scope
            )
            if availability_prior is not None
            else np.zeros((len(ids), 3), dtype=np.float32)
        ),
        availability_prior=availability_prior,
        text_context=text_context,
        text_embedding_dim=text_dim,
    )


class TemporalPredictor(nn.Module):
    def __init__(self, input_dim: int = 28, hidden_dim: int = 32, text_dim: int = 0):
        super().__init__()
        if type(text_dim) is not int or text_dim < 0:
            raise ValueError("Temporal text dimension must be a nonnegative integer")
        self.text_dim = text_dim
        self.text_projection = nn.Linear(text_dim, 64, bias=False) if text_dim else None
        self.recurrent = nn.GRU(input_dim + (64 if text_dim else 0), hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 15)

    def forward(self, context, prior_logits=None, text_context=None):
        if self.text_projection is not None:
            if (
                text_context is None
                or text_context.ndim != 3
                or text_context.shape[-1] != self.text_dim
            ):
                raise ValueError("Temporal text context must match configured text dimension")
            if text_context.shape[:2] != context.shape[:2]:
                raise ValueError("Temporal text context must match batch and history length")
            text_valid = (context[..., -1] > 0) & (context[..., -2] > 0)
            projected = self.text_projection(text_context).masked_fill(~text_valid[..., None], 0)
            context = torch.cat([context, projected], dim=-1)
        elif text_context is not None:
            raise ValueError("Temporal model is not configured for text context")
        _, hidden = self.recurrent(context)
        result = self.output(hidden[-1])
        logits = result[:, 12:]
        if prior_logits is not None:
            if prior_logits.shape != logits.shape:
                raise ValueError("Temporal prior logits must match batch and modalities")
            logits = logits + prior_logits
        return result[:, :12], logits


def temporal_loss(prediction, logits, values, observed, availability):
    targets = values.masked_fill(~observed, 0)
    residual = (prediction - targets).square().masked_fill(~observed, 0)
    value_loss = residual.sum(-1) / observed.sum(-1).clamp_min(1)
    availability_loss = nn.functional.binary_cross_entropy_with_logits(
        logits, availability, reduction="none"
    ).mean(-1)
    return value_loss + availability_loss, value_loss, availability_loss


def _fit_indices(prepared, inputs):
    ids = sorted(prepared.lists.fit)
    labels = prepared.labels.set_index("sample_id").label.reindex(ids)
    splits = prepared.splits.set_index("sample_id").split.reindex(ids)
    if (
        not ids
        or len(ids) != len(set(ids))
        or not labels.eq(0).fillna(False).all()
        or not splits.eq("fit").all()
    ):
        raise ValueError("Temporal fitting requires nonempty known-normal frozen fit IDs")
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    return np.asarray([lookup[sample_id] for sample_id in ids], dtype=int)


def train_temporal(prepared, inputs, settings, progress=None):
    indices = _fit_indices(prepared, inputs)
    set_seed(settings["seed"])
    device = torch.device(settings["device"])
    model = TemporalPredictor(
        inputs["context"].shape[-1], settings["hidden_dim"], inputs.get("text_embedding_dim", 0)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    tensors = {
        name: torch.as_tensor(inputs[name][indices], device=device)
        for name in ["context", "values", "observed", "availability", "prior_logits"]
    }
    if inputs.get("text_context") is not None:
        tensors["text_context"] = torch.as_tensor(inputs["text_context"][indices], device=device)
    history = []
    started = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(indices), device=device)
        for start in range(0, len(order), settings["batch_size"]):
            selected = order[start : start + settings["batch_size"]]
            prediction, logits = model(
                tensors["context"][selected],
                tensors["prior_logits"][selected],
                tensors.get("text_context", None)[selected] if "text_context" in tensors else None,
            )
            loss, _, _ = temporal_loss(
                prediction,
                logits,
                tensors["values"][selected],
                tensors["observed"][selected],
                tensors["availability"][selected],
            )
            loss = loss.mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite temporal training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        record = dict(
            epoch=epoch,
            mean_loss=total / len(indices),
            elapsed_seconds=time.perf_counter() - started,
        )
        history.append(record)
        LOGGER.info(
            "Temporal epoch %s/%s loss=%.6f elapsed=%.2fs",
            epoch,
            settings["epochs"],
            record["mean_loss"],
            record["elapsed_seconds"],
        )
        if progress is not None:
            progress(record)
    return model, history


def score_temporal(model, inputs, ids, batch_size=256):
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    ids = sorted(ids)
    indices = [lookup[sample_id] for sample_id in ids]
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            tensors = {
                name: torch.as_tensor(inputs[name][selected], device=device)
                for name in ["context", "values", "observed", "availability", "prior_logits"]
            }
            if inputs.get("text_context") is not None:
                tensors["text_context"] = torch.as_tensor(
                    inputs["text_context"][selected], device=device
                )
            prediction, logits = model(
                tensors["context"], tensors["prior_logits"], tensors.get("text_context")
            )
            components = temporal_loss(
                prediction, logits, tensors["values"], tensors["observed"], tensors["availability"]
            )
            rows.extend(torch.stack(components, dim=1).cpu().tolist())
    scores = np.asarray(rows, dtype=float).reshape(-1, 3)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite temporal scores")
    frame = inputs["identity"].set_index("sample_id").loc[ids].reset_index()
    frame[["score", "value_loss", "availability_loss"]] = scores
    return frame


def lag_features(inputs):
    parts = [inputs["context"].reshape(len(inputs["ids"]), -1), inputs["features"]]
    if inputs.get("text_context") is not None:
        parts.append(inputs["text_context"].reshape(len(inputs["ids"]), -1))
    if inputs.get("availability_prior") is not None:
        parts.append(inputs["prior_logits"])
    return np.concatenate(parts, axis=1)
