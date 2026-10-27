"""PatchAD contrastive objective and causal last-window scoring adapter."""

import numpy as np
import pandas as pd
import torch

from src.utils.seed import set_seed

from .temporal import _fit_indices


def _symmetric_divergences(first, second, sequence_length):
    if len(first) != len(second) or not first:
        raise ValueError("PatchAD requires matching nonempty patch distributions")
    first_loss = None
    second_loss = None
    for first_distribution, second_distribution in zip(first, second):
        first_length = first_distribution.shape[1]
        second_length = second_distribution.shape[1]
        if sequence_length % first_length or sequence_length % second_length:
            raise ValueError("Patch distributions must divide the sequence length")
        first_distribution = first_distribution.repeat(1, sequence_length // first_length, 1)
        second_distribution = second_distribution.repeat_interleave(
            sequence_length // second_length, dim=1
        )
        first_distribution = first_distribution / first_distribution.sum(dim=-1, keepdim=True)
        second_distribution = second_distribution / second_distribution.sum(dim=-1, keepdim=True)

        def divergence(left, right):
            return (left * (torch.log(left + 1e-7) - torch.log(right + 1e-7))).sum(dim=-1)

        first_term = divergence(first_distribution, second_distribution.detach()) + divergence(
            second_distribution.detach(), first_distribution
        )
        second_term = divergence(first_distribution.detach(), second_distribution) + divergence(
            second_distribution, first_distribution.detach()
        )
        first_loss = first_term if first_loss is None else first_loss + first_term
        second_loss = second_term if second_loss is None else second_loss + second_term
    return first_loss, second_loss


def patchad_contrastive_loss(output, sequence_length, patch_weight):
    if not 0 <= patch_weight <= 1:
        raise ValueError("Patch weight must be between zero and one")
    patch_number, patch_size, projected_number, projected_size, _ = output
    first, second = _symmetric_divergences(patch_number, projected_size, sequence_length)
    third, fourth = _symmetric_divergences(projected_number, patch_size, sequence_length)
    fifth, sixth = _symmetric_divergences(patch_number, patch_size, sequence_length)
    return -patch_weight * (first.mean() - second.mean() + third.mean() - fourth.mean()) - (
        1 - patch_weight
    ) * (fifth.mean() - sixth.mean()) / len(patch_number)


def patchad_last_window_score(output, sequence_length):
    patch_number, patch_size, _, _, _ = output
    divergence, _ = _symmetric_divergences(patch_number, patch_size, sequence_length)
    return torch.softmax(-divergence / len(patch_number), dim=-1)[:, -1]


def train_patchad(prepared, inputs, settings, model_factory):
    fit_indices = _fit_indices(prepared, inputs)
    if inputs["sequences"].ndim != 3 or inputs["sequences"].shape[2] != 28:
        raise ValueError("PatchAD requires the 28-field closed-window sequence")
    eligible = inputs["present"][fit_indices].all(axis=1)
    indices = fit_indices[eligible]
    if not len(indices):
        raise ValueError("PatchAD requires at least one full Normal fit sequence")
    if settings["epochs"] < 1 or settings["batch_size"] < 1 or settings["lr"] <= 0:
        raise ValueError("Invalid PatchAD training settings")
    set_seed(settings["seed"])
    device = torch.device(settings["device"])
    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["lr"])
    sequences = torch.as_tensor(inputs["sequences"][indices, :, :27], device=device)
    sequence_length = sequences.shape[1]
    history = []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(indices), device=device)
        total = 0.0
        for start in range(0, len(order), settings["batch_size"]):
            selected = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = patchad_contrastive_loss(
                model(sequences[selected]), sequence_length, settings["patch_weight"]
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite PatchAD training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selected)
        history.append({"epoch": epoch, "loss": total / len(indices)})
    return model, {
        "history": history,
        "fit_ids": [inputs["ids"][index] for index in indices],
        "excluded_incomplete_fit_ids": [inputs["ids"][index] for index in fit_indices[~eligible]],
    }


def score_patchad(model, inputs, ids, batch_size):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("PatchAD batch size must be positive")
    ids = sorted(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("PatchAD scoring IDs must be unique")
    lookup = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    if not set(ids) <= lookup.keys():
        raise ValueError("PatchAD scoring IDs are missing from inputs")
    indices = [lookup[sample_id] for sample_id in ids]
    device = next(model.parameters()).device
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            sequence = torch.as_tensor(inputs["sequences"][selected, :, :27], device=device)
            values = patchad_last_window_score(model(sequence), sequence.shape[1])
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite PatchAD scores")
            scores.extend(values.cpu().numpy())
    return pd.DataFrame({"sample_id": ids, "score": np.asarray(scores, dtype=float)})
