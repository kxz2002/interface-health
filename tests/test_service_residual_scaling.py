import numpy as np
import pytest

from src.service.modeling.residual_scaling import fit_residual_scales, scaled_value_loss


def test_field_residuals_reproduce_original_masked_value_loss():
    from scripts.service_gru_error_calibration import squared_residuals
    from src.service.modeling.temporal import TemporalPredictor, score_temporal, temporal_inputs
    from tests.test_service_model_temporal import fixture

    prepared, _ = fixture()
    inputs = temporal_inputs(prepared)
    model = TemporalPredictor()
    ids = {"test", "calibration", "fit-003"}
    errors, observed, _ = squared_residuals(model, inputs, ids)
    original = score_temporal(model, inputs, ids)
    np.testing.assert_allclose(
        errors.sum(1) / np.maximum(observed.sum(1), 1),
        original.value_loss,
        rtol=1e-6,
    )


def test_service_scale_shrinkage_and_unseen_service_global_fallback():
    errors = np.array([[1.0] * 12, [9.0] * 12])
    observed = np.ones((2, 12), dtype=bool)
    state = fit_residual_scales(errors, observed, ["a", "b"], ["n1", "n2"], {"n1", "n2"})
    assert state["service_scales"]["a"][0] == pytest.approx(161 / 33)
    assert state["global_scale"][0] == 5
    np.testing.assert_allclose(
        scaled_value_loss(state, errors, observed, ["new", "new"]), [0.2, 1.8]
    )


def test_masks_floor_and_exact_normal_membership():
    observed = np.ones((2, 12), dtype=bool)
    state = fit_residual_scales(np.zeros((2, 12)), observed, ["a", "a"], ["n1", "n2"], {"n1", "n2"})
    assert min(state["global_scale"]) == 1e-4
    assert (
        scaled_value_loss(state, np.full((1, 12), np.nan), np.zeros((1, 12), dtype=bool), ["a"])[0]
        == 0
    )
    with pytest.raises(ValueError, match="held-out Normal IDs"):
        fit_residual_scales(np.zeros((2, 12)), observed, ["a", "a"], ["n1", "test"], {"n1", "n2"})
    with pytest.raises(ValueError, match="nonnegative"):
        scaled_value_loss(state, -np.ones((1, 12)), observed[:1], ["a"])
