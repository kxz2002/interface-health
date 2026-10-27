import copy

import numpy as np
import pytest

from src.service.modeling.neural_baseline_data import closed_window_sequences
from tests.test_service_model_temporal import fixture


def test_sequences_include_current_window_and_only_contiguous_past():
    prepared, _ = fixture()
    result = closed_window_sequences(prepared, 4)
    index = result["ids"].index("fit-002")
    assert result["history_ids"]["fit-002"] == ["fit-000", "fit-001"]
    np.testing.assert_array_equal(result["present"][index], [False, True, True, True])
    assert result["sequences"][index, -1, 0] == pytest.approx(np.arcsinh(3.0))
    assert not result["observed"][index, 0].any()

    changed = copy.deepcopy(prepared)
    changed.numeric["values"].loc[3, "cpu_cores"] = 9999
    altered = closed_window_sequences(changed, 4)
    np.testing.assert_array_equal(result["sequences"][index], altered["sequences"][index])


@pytest.mark.parametrize("boundary", ["case", "service", "split", "gap", "excluded_fit"])
def test_sequences_do_not_cross_boundaries(boundary):
    prepared, _ = fixture()
    if boundary == "case":
        prepared.identity.loc[1, "case_id"] = "other"
    elif boundary == "service":
        prepared.identity.loc[1, "canonical_service_id"] = "other"
    elif boundary == "split":
        prepared.splits.loc[1, "split"] = "test"
    elif boundary == "gap":
        prepared.identity.loc[1, "window_start_ms"] += 5000
    else:
        prepared.lists.fit = tuple(sample for sample in prepared.lists.fit if sample != "fit-001")
    result = closed_window_sequences(prepared, 4)
    index = result["ids"].index("fit-002")
    assert result["history_ids"]["fit-002"] == []
    np.testing.assert_array_equal(result["present"][index], [False, False, False, True])


def test_missing_observation_mask_is_preserved():
    prepared, _ = fixture()
    prepared.numeric["masks"].loc[2, "cpu_cores"] = False
    prepared.numeric["values"].loc[2, "cpu_cores"] = np.nan
    result = closed_window_sequences(prepared, 3)
    index = result["ids"].index("fit-002")
    assert not result["observed"][index, -1, 0]
    assert result["sequences"][index, -1, 0] == 0


@pytest.mark.parametrize("length", [0, 1, 1.5, True])
def test_invalid_sequence_length_is_rejected(length):
    prepared, _ = fixture()
    with pytest.raises(ValueError, match="Sequence length"):
        closed_window_sequences(prepared, length)
