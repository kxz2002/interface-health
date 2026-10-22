"""Shared Service SVDD network and fit-only fixed-center initialization."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .encoders import NumericEncoder, TextEncoder, masked_values


class ServiceSVDD(nn.Module):
    def __init__(
        self,
        text_backbone: nn.Module,
        fusion: str = "transformer_cls",
        reconstruction_weight: float = 0.0,
        representation_dim: int = 32,
        reconstruction_asinh: bool = False,
    ):
        super().__init__()
        if fusion not in {"transformer_cls", "transformer_mean", "concat_svdd"}:
            raise ValueError(f"Unsupported fusion: {fusion}")
        self.fusion = fusion
        self.reconstruction_weight = reconstruction_weight
        self.representation_dim = representation_dim
        self.reconstruction_asinh = reconstruction_asinh
        if reconstruction_weight:
            self.decoder = nn.Sequential(
                nn.Linear(representation_dim, 64), nn.ReLU(), nn.Linear(64, 12)
            )
        self.resource_encoder = NumericEncoder(3)
        self.execution_encoder = NumericEncoder(5)
        self.text_encoder = TextEncoder(text_backbone)
        self.modality_embedding = nn.Embedding(3, 64)
        if fusion == "concat_svdd":
            self.projection = nn.Sequential(
                nn.Linear(195, 64, bias=False),
                nn.ReLU(),
                nn.Linear(64, representation_dim, bias=False),
            )
        else:
            layer = nn.TransformerEncoderLayer(
                64, 4, 128, 0.1, activation="relu", batch_first=True, bias=False
            )
            layer.norm1.bias = nn.Parameter(torch.zeros(64))
            layer.norm2.bias = nn.Parameter(torch.zeros(64))
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            if fusion == "transformer_cls":
                self.summary_token = nn.Parameter(torch.empty(1, 1, 64))
                nn.init.normal_(self.summary_token, std=0.02)
            self.projection = nn.Linear(64, representation_dim, bias=False)
        self.register_buffer("center", torch.zeros(representation_dim))
        self.register_buffer("center_initialized", torch.tensor(False))

    def encode(self, batch: dict) -> dict:
        if self.reconstruction_asinh:
            batch = dict(batch)
            for name in ["resource", "execution", "stats"]:
                batch[name] = torch.asinh(batch[name])
        observed_text = []
        for messages in batch["logs"]:
            if messages is None:
                observed_text.append(False)
            else:
                attention = torch.as_tensor(messages["attention_mask"]).bool()
                special = torch.as_tensor(messages["special_mask"]).bool()
                observed_text.append(bool((attention & ~special).any()))
        available = batch["resource_mask"].any(-1) | batch["execution_mask"].any(-1)
        available = available | torch.tensor(observed_text, device=available.device)
        if not available.all():
            raise ValueError("All modalities unavailable; filter windows before encoding")
        encoded = [
            self.resource_encoder(batch["resource"], batch["resource_mask"]),
            self.execution_encoder(batch["execution"], batch["execution_mask"]),
            self.text_encoder(batch["logs"], batch["stats"], batch["stats_mask"]),
        ]
        valid = torch.stack([modality.valid for modality in encoded], dim=1)
        if not valid.any(1).all():
            raise ValueError("All modalities unavailable; filter windows before encoding")
        tokens = torch.stack([modality.token for modality in encoded], dim=1)
        if tokens.shape != (len(valid), 3, 64):
            raise ValueError("Invalid modality token shape")
        tokens = (tokens + self.modality_embedding.weight).masked_fill(~valid[:, :, None], 0)
        if self.fusion == "concat_svdd":
            combined = torch.cat([tokens.flatten(1), valid.to(tokens.dtype)], dim=1)
        elif self.fusion == "transformer_cls":
            tokens = torch.cat([self.summary_token.expand(len(valid), -1, -1), tokens], dim=1)
            padding = torch.cat(
                [torch.zeros(len(valid), 1, dtype=torch.bool, device=valid.device), ~valid],
                dim=1,
            )
            combined = self.transformer(tokens, src_key_padding_mask=padding)[:, 0]
        else:
            outputs = self.transformer(tokens, src_key_padding_mask=~valid)
            combined = outputs.masked_fill(~valid[:, :, None], 0).sum(1) / valid.sum(
                1, keepdim=True
            )
        representation = self.projection(combined)
        if (
            representation.shape != (len(valid), self.representation_dim)
            or not torch.isfinite(representation).all()
        ):
            raise ValueError("Invalid SVDD representation")
        auxiliary = {}
        for name, modality in zip(["resource", "execution", "log"], encoded):
            auxiliary.update({f"{name}.{key}": value for key, value in modality.aux_losses.items()})
        return dict(representation=representation, aux_losses=auxiliary)

    def forward(self, batch: dict) -> dict:
        if not self.center_initialized:
            raise RuntimeError("SVDD center has not been initialized")
        output = self.encode(batch)
        output["score"] = (output["representation"] - self.center).square().sum(-1)
        if self.reconstruction_weight:
            targets = torch.cat([batch["resource"], batch["execution"], batch["stats"]], dim=-1)
            text_valid = torch.tensor(
                [
                    messages is not None
                    and bool(
                        (
                            torch.as_tensor(messages["attention_mask"]).bool()
                            & ~torch.as_tensor(messages["special_mask"]).bool()
                        ).any()
                    )
                    for messages in batch["logs"]
                ],
                device=targets.device,
            )
            mask = torch.cat(
                [
                    batch["resource_mask"],
                    batch["execution_mask"],
                    batch["stats_mask"] & text_valid[:, None],
                ],
                dim=-1,
            )
            targets = masked_values(targets, mask)
            if self.reconstruction_asinh:
                targets = torch.asinh(targets)
            residual = (self.decoder(output["representation"]) - targets).masked_fill(~mask, 0)
            reconstruction = residual.square().sum(-1) / mask.sum(-1).clamp_min(1)
            output["aux_losses"]["reconstruction"] = (
                self.reconstruction_weight * reconstruction.mean()
            )
            output["score"] = reconstruction
        if not torch.isfinite(output["score"]).all():
            raise ValueError("Nonfinite SVDD scores")
        return output


def initialize_center(model: ServiceSVDD, fit_loader: Iterable[dict], eps: float = 0.1) -> None:
    if model.center_initialized:
        raise RuntimeError("SVDD center already initialized")
    if eps != 0.1:
        raise ValueError("Approved center epsilon is 0.1")
    was_training = model.training
    model.eval()
    total = torch.zeros_like(model.center, dtype=torch.float64)
    count = 0
    try:
        with torch.no_grad():
            for batch in fit_loader:
                representation = model.encode(batch)["representation"]
                total += representation.double().sum(0)
                count += len(representation)
            if count == 0:
                raise ValueError("Empty fit loader for center initialization")
            center = (total / count).to(model.center.dtype)
            center = torch.where(center.abs() < eps, torch.where(center < 0, -eps, eps), center)
            model.center.copy_(center)
            model.center_initialized.fill_(True)
    finally:
        model.train(was_training)


def build_model(config: dict, text_backbone: nn.Module) -> ServiceSVDD:
    settings = config.get("model", {})
    defaults = dict(
        hidden_dim=64,
        token_dim=64,
        transformer_layers=1,
        attention_heads=4,
        feedforward_dim=128,
        dropout=0.1,
        bias=False,
        activation="relu",
        center_epsilon=0.1,
    )
    for name, expected in defaults.items():
        if settings.get(name, expected) != expected:
            raise ValueError(f"Unsupported network setting: {name}")
    auxiliary = settings.get("auxiliary_losses", {})
    if auxiliary and (
        set(auxiliary) not in ({"reconstruction"}, {"reconstruction_asinh"})
        or type(next(iter(auxiliary.values()))) not in {int, float}
        or not 0 < next(iter(auxiliary.values())) < float("inf")
    ):
        raise ValueError("Unsupported reconstruction objective")
    representation_dim = settings.get("representation_dim", 32)
    if type(representation_dim) is not int or representation_dim < 1 or representation_dim > 32:
        raise ValueError("Representation dimension must be an integer between 1 and 32")
    if not auxiliary and representation_dim != 32:
        raise ValueError("SVDD representation dimension remains fixed at 32")
    fusion = settings.get("fusion", "transformer_cls")
    fusion = {"summary_token": "transformer_cls", "mean": "transformer_mean"}.get(fusion, fusion)
    return ServiceSVDD(
        text_backbone,
        fusion,
        next(iter(auxiliary.values()), 0.0),
        representation_dim,
        "reconstruction_asinh" in auxiliary,
    )
