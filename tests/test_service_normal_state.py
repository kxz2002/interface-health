import numpy as np
import pytest

from src.service.modeling.normal_state import (
    fit_fusion_reference,
    fit_normal_state,
    fuse_state_temporal,
    score_normal_state,
)


def test_normal_reference_ignores_missing_and_falls_back_for_sparse_service():
    values = np.zeros((40, 12))
    observed = np.ones_like(values, dtype=bool)
    services = ["a"] * 39 + ["sparse"]
    values[-1] = 100
    values[0, 0] = np.nan
    observed[0, 0] = False
    ids = [f"n{index}" for index in range(40)]
    state = fit_normal_state(values, observed, services, ids, ids)
    assert state["services"]["sparse"]["center"] == state["global_center"]
    assert min(state["global_scale"]) == 0.05
    np.testing.assert_array_equal(
        score_normal_state(state, np.full((1, 12), np.nan), np.zeros((1, 12), bool), ["new"]), [0]
    )
    assert score_normal_state(state, np.ones((1, 12)), np.ones((1, 12), bool), ["a"])[
        0
    ] == pytest.approx(400)
    with pytest.raises(ValueError, match="permitted"):
        fit_normal_state(values, observed, services, ids, ids[:-1])


def test_fixed_fusion_uses_only_declared_reference_and_retains_negative_scores():
    scores = np.array([[0.0, 2.0], [1.0, 3.0], [2.0, 4.0]])
    state = fit_fusion_reference(scores, ["n1", "n2", "n3"], {"n1", "n2", "n3"})
    assert fuse_state_temporal(state, [[0.0, 2.0]])[0] < 0
    assert fuse_state_temporal(state, [[1.0, 3.0]])[0] == 0
    expected = ((scores - state["center"]) / state["scale"]).max(1)
    np.testing.assert_allclose(fuse_state_temporal(state, scores), expected)
    with pytest.raises(ValueError, match="held-out"):
        fit_fusion_reference(scores, ["n1", "n2", "test"], {"n1", "n2", "n3"})
