"""Masked multivariate reconstruction on frozen Normal service windows."""

import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed

from .temporal import _fit_indices


def reconstruction_errors(prediction, values, observed, present):
    if prediction.shape != values.shape or values.shape[-1] != 27:
        raise ValueError("Reconstruction requires matching 27-channel sequences")
    if observed.shape != values.shape[:2] + (12,) or present.shape != values.shape[:2]:
        raise ValueError("Invalid reconstruction masks")
    mask = torch.cat(
        [observed, present.unsqueeze(-1).expand(-1, -1, 15)], dim=-1
    ) & present.unsqueeze(-1)
    count = mask.sum(dim=-1)
    errors = (prediction - values).square().masked_fill(~mask, 0).sum(dim=-1)
    return errors / count.clamp_min(1)


def train_reconstruction(prepared, inputs, settings, model_factory):
    fit_indices = _fit_indices(prepared, inputs)
    eligible = inputs["present"][fit_indices].all(axis=1)
    indices = fit_indices[eligible]
    if not len(indices):
        raise ValueError("Reconstruction requires full Normal fit sequences")
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["lr"] <= 0:
        raise ValueError("Invalid reconstruction training settings")
    set_seed(settings["seed"])
    device = torch.device(settings["device"])
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    values = torch.as_tensor(inputs["sequences"][indices, :, :27], device=device)
    observed = torch.as_tensor(inputs["observed"][indices], device=device)
    present = torch.as_tensor(inputs["present"][indices], device=device)
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(indices), device=device)
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            selected = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values[selected], None, None, None)
            loss = reconstruction_errors(
                prediction, values[selected], observed[selected], present[selected]
            ).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reconstruction training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        history.append({"epoch": epoch, "loss": total / len(indices)})
    return model, {
        "history": history,
        "fit_ids": [inputs["ids"][index] for index in indices],
        "excluded_incomplete_fit_ids": [inputs["ids"][index] for index in fit_indices[~eligible]],
    }


def score_reconstruction(model, inputs, ids, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Reconstruction batch size must be positive")
    ids = sorted(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Reconstruction scoring IDs must be unique")
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    if not set(ids) <= lookup.keys():
        raise ValueError("Reconstruction scoring IDs are missing from inputs")
    indices = [lookup[sample_id] for sample_id in ids]
    device = next(model.parameters()).device
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            values = torch.as_tensor(inputs["sequences"][selected, :, :27], device=device)
            observed = torch.as_tensor(inputs["observed"][selected], device=device)
            present = torch.as_tensor(inputs["present"][selected], device=device)
            errors = reconstruction_errors(
                model(values, None, None, None), values, observed, present
            )
            final = errors[:, -1]
            if not torch.isfinite(final).all():
                raise ValueError("Nonfinite reconstruction scores")
            scores.extend(final.cpu().numpy())
    return pd.DataFrame({"sample_id": ids, "score": np.asarray(scores, dtype=float)})
