import copy

import numpy as np
import pytest
import torch

from src.service.modeling.neural_baseline_data import closed_window_sequences
from src.service.modeling.patchad_adapter import (
    patchad_contrastive_loss,
    patchad_last_window_score,
    score_patchad,
    train_patchad,
)
from tests.test_service_model_temporal import fixture


def distributions():
    generator = torch.Generator().manual_seed(42)
    return [
        torch.softmax(torch.randn(2, 4, 6, generator=generator), dim=-1).requires_grad_(),
        torch.softmax(torch.randn(2, 3, 6, generator=generator), dim=-1).requires_grad_(),
    ]


def test_patchad_loss_and_last_point_score_are_finite_and_differentiable():
    output = (distributions(), distributions(), distributions(), distributions(), None)
    loss = patchad_contrastive_loss(output, sequence_length=12, patch_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None for group in output[:4] for value in group)
    scores = patchad_last_window_score(output, sequence_length=12)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_patchad_rejects_nondivisible_patch_lengths():
    output = (distributions(), distributions(), distributions(), distributions(), None)
    with pytest.raises(ValueError, match="divide"):
        patchad_last_window_score(output, sequence_length=10)


def test_patchad_score_uses_upstream_patch_expansion_order():
    first = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
    second = torch.tensor([[[0.4, 0.6], [0.7, 0.3]]])
    expanded_first = first[:, [0, 1, 0, 1]]
    expanded_second = second[:, [0, 0, 1, 1]]
    forward = expanded_first * (
        torch.log(expanded_first + 1e-7) - torch.log(expanded_second + 1e-7)
    )
    reverse = expanded_second * (
        torch.log(expanded_second + 1e-7) - torch.log(expanded_first + 1e-7)
    )
    expected = torch.softmax(-(forward + reverse).sum(-1), dim=-1)[:, -1]
    output = ([first], [second], [], [], None)
    torch.testing.assert_close(patchad_last_window_score(output, 4), expected)


class TinyPatchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(27, 3)
        self.second = torch.nn.Linear(27, 3)

    def forward(self, sequence):
        first = torch.softmax(self.first(sequence[:, :2]), dim=-1)
        second = torch.softmax(self.second(sequence[:, 2:]), dim=-1)
        return [first], [second], [first], [second], sequence


def test_training_only_uses_full_known_normal_fit_sequences():
    prepared, _ = fixture()
    settings = dict(epochs=2, batch_size=2, lr=0.001, seed=42, device="cpu", patch_weight=0.1)
    inputs = closed_window_sequences(prepared, 4)
    model, evidence = train_patchad(prepared, inputs, settings, TinyPatchModel)
    assert evidence["fit_ids"] == ["fit-003", "fit-004", "fit-005"]
    assert evidence["excluded_incomplete_fit_ids"] == ["fit-000", "fit-001", "fit-002"]

    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6:, "cpu_cores"] = 99999
    repeat, _ = train_patchad(
        changed, closed_window_sequences(changed, 4), settings, TinyPatchModel
    )
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, repeat.state_dict()[name], rtol=0, atol=0)

    scores = score_patchad(model, inputs, ["test", "calibration"], batch_size=1)
    assert scores.sample_id.tolist() == ["calibration", "test"]
    assert np.isfinite(scores.score).all()
    assert len(evidence["history"]) == 2


def test_training_rejects_non_normal_fit():
    prepared, _ = fixture()
    prepared.labels.loc[0, "label"] = 1
    settings = dict(epochs=1, batch_size=2, lr=0.001, seed=42, device="cpu", patch_weight=0.1)
    with pytest.raises(ValueError, match="known-normal"):
        train_patchad(prepared, closed_window_sequences(prepared, 4), settings, TinyPatchModel)
