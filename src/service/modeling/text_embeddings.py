"""Offline frozen BERT representations for causal service windows."""

from __future__ import annotations

import logging

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def pool_log_messages(backbone, messages: dict, device: torch.device) -> torch.Tensor | None:
    ids = torch.as_tensor(messages["input_ids"], device=device)
    attention = torch.as_tensor(messages["attention_mask"], device=device)
    special = torch.as_tensor(messages["special_mask"], device=device)
    groups = messages["group_ids"]
    if ids.numel() == 0:
        return None
    if ids.ndim != 2 or ids.shape != attention.shape or ids.shape != special.shape:
        raise ValueError("Invalid log token shapes for frozen text embedding")
    if len(groups) != ids.shape[0]:
        raise ValueError("Invalid log group count for frozen text embedding")
    body = attention.bool() & ~special.bool()
    valid = body.any(dim=1)
    if not valid.any():
        return None
    hidden = backbone(input_ids=ids[valid], attention_mask=attention[valid]).last_hidden_state
    body = body[valid]
    pooled = hidden.masked_fill(~body[:, :, None], 0).sum(1) / body.sum(1, keepdim=True)
    kept_groups = [group for group, keep in zip(groups, valid.tolist()) if keep]
    group_means = [
        pooled[[index for index, group in enumerate(kept_groups) if group == name]].mean(0)
        for name in sorted(set(kept_groups))
    ]
    result = torch.stack(group_means).mean(0)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite frozen text embedding")
    return result


def build_frozen_text_embeddings(
    prepared,
    ids: list[str],
    *,
    model_name: str,
    revision: str,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Text embedding batch size must be positive")
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(prepared.sample_ids):
        raise ValueError("Text embedding sample IDs must be nonempty, unique and prepared")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required for frozen text embeddings") from exc
    backbone = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=True).to(
        torch.device(device)
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    target_device = next(backbone.parameters()).device
    LOGGER.info("Encoding %s log windows using frozen BERT on %s", len(ids), target_device)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            for sample_id in ids[start : start + batch_size]:
                messages = {
                    name: prepared.logs[name][sample_id]
                    for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]
                }
                pooled = pool_log_messages(backbone, messages, target_device)
                if pooled is None:
                    pooled = torch.zeros(backbone.config.hidden_size, device=target_device)
                vectors.append(pooled.cpu().numpy())
            if start % (batch_size * 32) == 0 or len(vectors) == len(ids):
                LOGGER.info("Frozen BERT windows %s/%s", len(vectors), len(ids))
    result = np.asarray(vectors, dtype=np.float32)
    if result.shape != (len(ids), backbone.config.hidden_size) or not np.isfinite(result).all():
        raise ValueError("Invalid frozen text embedding cache")
    LOGGER.info(
        "Built frozen text embeddings for %s windows with dim=%s", len(ids), result.shape[1]
    )
    return result
