"""Frozen Normal-only candidates for the eight-direction exploratory comparison."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional

from src.utils.seed import set_seed

from .temporal import (
    GroupedPredictor,
    TemporalPredictor,
    _fit_indices,
    latest_observed_values,
    multiscale_context,
    temporal_loss,
)

LOGGER = logging.getLogger(__name__)
DIRECTIONS = (
    "dual_head",
    "multiscale",
    "grouped",
    "prototype",
    "mechanism",
    "heteroscedastic",
    "tcn_gru",
    "masked_pretraining",
)


def reference_with_mask(context):
    reference = latest_observed_values(context)
    valid = ((context[..., 12:24] > 0) & (context[..., -1:] > 0)).any(1)
    return reference, valid


class ExplorationPredictor(nn.Module):
    def __init__(self, direction, config):
        super().__init__()
        if direction not in DIRECTIONS and direction != "baseline":
            raise ValueError("Unknown exploration direction")
        self.direction = direction
        self.config = config
        hidden = config["training"]["hidden_dim"]
        if direction == "grouped":
            self.base = GroupedPredictor(28, config["group_hidden_dim"])
        else:
            input_dim = 76 if direction == "multiscale" else 28
            if direction == "tcn_gru":
                channels = config["tcn_channels"]
                self.convolutions = nn.ModuleList(
                    [
                        nn.Conv1d(
                            28 if index == 0 else channels,
                            channels,
                            config["tcn_kernel"],
                            dilation=dilation,
                        )
                        for index, dilation in enumerate(config["tcn_dilations"])
                    ]
                )
                input_dim = channels
            self.base = TemporalPredictor(input_dim, hidden)
        if direction == "dual_head":
            self.delta_head = nn.Linear(hidden, 12)
        if direction == "heteroscedastic":
            self.variance_head = nn.Linear(hidden, 12)
            nn.init.zeros_(self.variance_head.weight)
            nn.init.zeros_(self.variance_head.bias)
        if direction == "masked_pretraining":
            self.reconstruction_head = nn.Linear(hidden, 12)

    def encode(self, context):
        if self.direction == "multiscale":
            context = multiscale_context(context)
        elif self.direction == "tcn_gru":
            signal = context.transpose(1, 2)
            for convolution in self.convolutions:
                padding = (convolution.kernel_size[0] - 1) * convolution.dilation[0]
                signal = functional.gelu(convolution(functional.pad(signal, (padding, 0))))
            context = signal.transpose(1, 2)
        sequence, hidden = self.base.recurrent(context)
        return sequence, hidden[-1]

    def forward(self, context, prior):
        if self.direction == "grouped":
            prediction, logits = self.base(context, prior)
            return dict(prediction=prediction, logits=logits)
        _, hidden = self.encode(context)
        output = self.base.output(hidden)
        result = dict(prediction=output[:, :12], logits=output[:, 12:] + prior, hidden=hidden)
        if self.direction == "dual_head":
            result["delta"] = self.delta_head(hidden)
        if self.direction == "heteroscedastic":
            bound = self.config["log_variance_bound"]
            result["log_variance"] = bound * torch.tanh(self.variance_head(hidden) / bound)
        return result


def candidate_loss(model, batch):
    output = model(batch["context"], batch["prior_logits"])
    original, value_loss, availability_loss = temporal_loss(
        output["prediction"],
        output["logits"],
        batch["values"],
        batch["observed"],
        batch["availability"],
    )
    extra = torch.zeros_like(value_loss)
    if model.direction == "dual_head":
        reference, history_valid = reference_with_mask(batch["context"])
        valid = batch["observed"] & history_valid
        errors = (output["delta"] - (batch["values"] - reference)).square()
        extra = errors.masked_fill(~valid, 0).sum(-1) / valid.sum(-1).clamp_min(1)
        value_loss = torch.where(valid.any(-1), (value_loss + extra) / 2, value_loss)
    elif model.direction == "heteroscedastic":
        variance = output["log_variance"]
        errors = (output["prediction"] - batch["values"]).square()
        terms = 0.5 * (
            errors * torch.exp(-variance) + variance + model.config["log_variance_bound"]
        )
        value_loss = terms.masked_fill(~batch["observed"], 0).sum(-1)
        value_loss = value_loss / batch["observed"].sum(-1).clamp_min(1)
        extra = original - availability_loss
    return value_loss + availability_loss, value_loss, availability_loss, extra


def tensor_batch(inputs, selected, device):
    return {
        name: torch.as_tensor(inputs[name][selected], device=device)
        for name in ["context", "values", "observed", "availability", "prior_logits"]
    }


def masked_history_loss(model, context, probability):
    valid = (context[..., 12:24] > 0) & (context[..., -1:] > 0)
    masked = (torch.rand(valid.shape, device=context.device) < probability) & valid
    altered = context.clone()
    altered[..., :12].masked_fill_(masked, 0)
    altered[..., 12:24].masked_fill_(masked, 0)
    sequence, _ = model.encode(altered)
    reconstructed = model.reconstruction_head(sequence)
    errors = (reconstructed - context[..., :12]).square().masked_fill(~masked, 0)
    return errors.sum() / masked.sum().clamp_min(1)


def train_candidate(prepared, inputs, direction, config, seed, progress=None):
    indices = _fit_indices(prepared, inputs)
    set_seed(seed)
    settings = config["training"]
    device = torch.device(settings["device"])
    model = ExplorationPredictor(direction, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    tensors = tensor_batch(inputs, indices, device)
    phases = (
        [("pretraining", config["pretraining_epochs"])] if direction == "masked_pretraining" else []
    ) + [("prediction", settings["epochs"])]
    history = []
    started = time.perf_counter()
    for phase, epochs in phases:
        if phase == "prediction" and direction == "masked_pretraining":
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
            )
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(indices), device=device)
            total = 0.0
            for start in range(0, len(order), settings["batch_size"]):
                selected = order[start : start + settings["batch_size"]]
                batch = {name: tensor[selected] for name, tensor in tensors.items()}
                loss = (
                    masked_history_loss(model, batch["context"], config["mask_probability"])
                    if phase == "pretraining"
                    else candidate_loss(model, batch)[0].mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite exploration loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(), settings["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(selected)
            record = dict(
                phase=phase,
                epoch=epoch,
                mean_loss=total / len(indices),
                elapsed_seconds=time.perf_counter() - started,
            )
            history.append(record)
            LOGGER.info(
                "%s seed=%s %s %s/%s loss=%.6f",
                direction,
                seed,
                phase,
                epoch,
                epochs,
                record["mean_loss"],
            )
            if progress is not None:
                progress(record)
    model.eval()
    return model, history


def score_candidate(model, inputs, ids):
    ordered = sorted(ids)
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    indices = [lookup[sample_id] for sample_id in ordered]
    device = next(model.parameters()).device
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = tensor_batch(inputs, indices[start : start + 256], device)
            chunks.append(torch.stack(candidate_loss(model, batch), dim=1).cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite exploration score")
    frame = inputs["identity"].set_index("sample_id").loc[ordered].reset_index()
    frame[["score", "value_loss", "availability_loss", "auxiliary_loss"]] = result
    return frame


def hidden_states(model, inputs, ids):
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    selected = [lookup[sample_id] for sample_id in sorted(ids)]
    device = next(model.parameters()).device
    chunks = []
    with torch.no_grad():
        for start in range(0, len(selected), 256):
            context = torch.as_tensor(
                inputs["context"][selected[start : start + 256]], device=device
            )
            chunks.append(model.encode(context)[1].cpu().numpy())
    return np.concatenate(chunks)


def fit_prototypes(model, prepared, inputs, config, seed):
    ids = sorted(prepared.lists.fit)
    _fit_indices(prepared, inputs)
    states = hidden_states(model, inputs, ids)
    services = inputs["identity"].set_index("sample_id").loc[ids].canonical_service_id.astype(str)
    memory = {}
    for service in sorted(set(services)):
        values = states[services.to_numpy() == service]
        mean = values.mean(0)
        scale = np.maximum(values.std(0), config["prototype_scale_floor"])
        normalized = (values - mean) / scale
        count = min(config["prototype_count"], len(np.unique(normalized, axis=0)))
        centers = (
            KMeans(n_clusters=count, n_init=10, random_state=seed).fit(normalized).cluster_centers_
        )
        memory[service] = dict(mean=mean.tolist(), scale=scale.tolist(), centers=centers.tolist())
    return dict(fit_ids=ids, services=memory)


def prototype_distances(model, inputs, ids, memory):
    ordered = sorted(ids)
    states = hidden_states(model, inputs, ordered)
    services = (
        inputs["identity"].set_index("sample_id").loc[ordered].canonical_service_id.astype(str)
    )
    distances = np.zeros(len(ordered))
    for service in sorted(set(services)):
        if service not in memory["services"]:
            raise ValueError("Missing fit-normal service prototype")
        state = memory["services"][service]
        selected = services.to_numpy() == service
        normalized = (states[selected] - state["mean"]) / state["scale"]
        distances[selected] = (
            ((normalized[:, None] - np.asarray(state["centers"])[None]) ** 2).mean(-1).min(-1)
        )
    return distances


def mechanism_components(inputs, ids):
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    selected = [lookup[sample_id] for sample_id in sorted(ids)]
    context = torch.from_numpy(inputs["context"][selected])
    reference, valid = reference_with_mask(context)
    valid &= torch.from_numpy(inputs["observed"][selected])
    changes = (
        (torch.from_numpy(inputs["values"][selected]) - reference).square().masked_fill(~valid, 0)
    )
    level = changes.max(-1).values.numpy()
    present = context[..., -1] > 0
    availability = context[..., 24:27]
    last = torch.zeros((len(selected), 3))
    for position in range(context.shape[1]):
        last = torch.where(present[:, position : position + 1], availability[:, position], last)
    transition = (last - torch.from_numpy(inputs["availability"][selected])).abs().mean(-1)
    transition.masked_fill_(~present.any(-1), 0)
    return np.column_stack([level, transition.numpy()])


def fusion_scales(normal_components, floor):
    components = np.asarray(normal_components, dtype=float)
    if components.ndim != 2 or not len(components) or not np.isfinite(components).all():
        raise ValueError("Invalid Normal fusion reference")
    return np.maximum(np.quantile(components, 0.95, axis=0), floor).tolist()


def fused_score(components, scales, weight):
    normalized = np.log1p(np.asarray(components) / np.asarray(scales))
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite fused score")
    return (1 - weight) * normalized[:, 0] + weight * normalized[:, 1:].mean(-1)
