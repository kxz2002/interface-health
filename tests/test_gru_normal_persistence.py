import numpy as np
import pytest

from scripts.diagnose_gru_normal_persistence import persistence_diagnostic


def test_persistence_uses_latest_observed_past_and_zero_without_history():
    context = np.zeros((1, 4, 28), dtype=np.float32)
    context[0, 0, 0] = 2
    context[0, 0, 12] = 1
    context[0, 0, -1] = 1
    context[0, 2, 0] = 3
    context[0, 2, 12] = 1
    context[0, 2, -1] = 1
    context[0, 3, 0] = 999
    context[0, 3, 1] = 999
    context[0, 3, 13] = 1
    values = np.zeros((1, 12), dtype=np.float32)
    values[0, :2] = [4, 2]
    observed = np.zeros((1, 12), dtype=bool)
    observed[0, :2] = True
    inputs = dict(
        ids=["validation"],
        context=context,
        values=values,
        observed=observed,
        feature_order=[f"feature{index}" for index in range(12)],
    )
    result = persistence_diagnostic(inputs, ["validation"])
    assert result["value_loss"] == pytest.approx(2.5)
    assert result["observed_without_history"] == 1
    assert result["per_feature"]["feature0"]["mse"] == 1
    assert result["per_feature"]["feature1"]["mse"] == 4
