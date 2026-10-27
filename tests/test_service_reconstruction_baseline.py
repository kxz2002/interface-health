import copy

import numpy as np
import pytest
import torch

from src.service.modeling.neural_baseline_data import closed_window_sequences
from src.service.modeling.reconstruction_baseline import (
    reconstruction_errors,
    score_reconstruction,
    train_reconstruction,
)
from tests.test_service_model_temporal import fixture


class TinyReconstruction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(27, 27)

    def forward(self, values, *args):
        return self.layer(values)


def test_missing_values_and_padding_do_not_contribute_to_loss():
    target = torch.zeros(1, 2, 27)
    prediction = torch.zeros_like(target)
    prediction[0, 0] = 100
    prediction[0, 1, 0] = 100
    observed = torch.ones(1, 2, 12, dtype=torch.bool)
    observed[0, 1, 0] = False
    present = torch.tensor([[False, True]])
    assert torch.equal(
        reconstruction_errors(prediction, target, observed, present), torch.zeros(1, 2)
    )


def test_normal_fit_training_and_scoring_are_reproducible():
    prepared, _ = fixture()
    inputs = closed_window_sequences(prepared, 4)
    settings = dict(epochs=2, batch_size=2, lr=0.001, seed=42, device="cpu")
    model, evidence = train_reconstruction(prepared, inputs, settings, TinyReconstruction)
    assert evidence["fit_ids"] == ["fit-003", "fit-004", "fit-005"]
    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[6:, "cpu_cores"] = 99999
    repeat, _ = train_reconstruction(
        changed, closed_window_sequences(changed, 4), settings, TinyReconstruction
    )
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, repeat.state_dict()[name], rtol=0, atol=0)
    scores = score_reconstruction(model, inputs, ["test", "calibration"], 1)
    assert scores.sample_id.tolist() == ["calibration", "test"]
    assert np.isfinite(scores.score).all()


def test_fit_rejects_fault_labels():
    prepared, _ = fixture()
    prepared.labels.loc[0, "label"] = 1
    settings = dict(epochs=1, batch_size=2, lr=0.001, seed=42, device="cpu")
    with pytest.raises(ValueError, match="known-normal"):
        train_reconstruction(
            prepared, closed_window_sequences(prepared, 4), settings, TinyReconstruction
        )
