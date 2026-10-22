"""Differentiable modality encoders with explicit observation masks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass
class EncodedModality:
    token: Tensor
    valid: Tensor
    aux_losses: dict[str, Tensor] = field(default_factory=dict)


def masked_values(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("Invalid field mask shape or dtype")
    cleaned = values.masked_fill(~mask, 0)
    if not torch.isfinite(cleaned).all():
        raise ValueError("Nonfinite observed numeric values")
    return cleaned


class NumericEncoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )

    def forward(self, values: Tensor, field_mask: Tensor) -> EncodedModality:
        cleaned = masked_values(values, field_mask)
        valid = field_mask.any(dim=-1)
        token = self.network(torch.cat([cleaned, field_mask.to(values.dtype)], dim=-1))
        return EncodedModality(token.masked_fill(~valid[:, None], 0), valid)


class TextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        if any(not parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("Text backbone must remain jointly trainable")
        self.projection = nn.Linear(backbone.config.hidden_size + 4, 64, bias=False)

    def pool_messages(self, messages: dict) -> Tensor | None:
        device = self.projection.weight.device
        ids = torch.as_tensor(messages["input_ids"], device=device)
        attention = torch.as_tensor(messages["attention_mask"], device=device)
        special = torch.as_tensor(messages["special_mask"], device=device)
        if ids.numel() == 0:
            return None
        if ids.ndim != 2 or ids.shape != attention.shape or ids.shape != special.shape:
            raise ValueError("Invalid message token shapes")
        if len(messages["group_ids"]) != len(ids):
            raise ValueError("Invalid message group count")
        body = attention.bool() & ~special.bool()
        valid = body.any(-1)
        if not valid.any():
            return None
        hidden = self.backbone(
            input_ids=ids[valid], attention_mask=attention[valid]
        ).last_hidden_state
        body = body[valid]
        pooled = hidden.masked_fill(~body[:, :, None], 0).sum(1) / body.sum(1, keepdim=True)
        groups = [group for group, keep in zip(messages["group_ids"], valid.tolist()) if keep]
        group_means = [
            pooled[[index for index, value in enumerate(groups) if value == group]].mean(0)
            for group in sorted(set(groups))
        ]
        result = torch.stack(group_means).mean(0)
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite text representation")
        return result

    def forward(
        self, log_batch: list[dict | None], stats: Tensor, stats_mask: Tensor
    ) -> EncodedModality:
        if stats.shape != (len(log_batch), 4):
            raise ValueError("Invalid log statistics shape")
        vectors = [
            self.pool_messages(messages) if messages is not None else None for messages in log_batch
        ]
        valid = torch.tensor([value is not None for value in vectors], device=stats.device)
        cleaned = masked_values(stats, stats_mask & valid[:, None])
        empty = stats.new_zeros(self.backbone.config.hidden_size)
        pooled = torch.stack([value if value is not None else empty for value in vectors])
        token = self.projection(torch.cat([pooled, cleaned], -1))
        return EncodedModality(token.masked_fill(~valid[:, None], 0), valid)
