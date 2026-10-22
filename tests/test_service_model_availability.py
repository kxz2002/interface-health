import copy

import numpy as np
import pytest

from src.service.modeling.availability import fit_availability, score_availability
from src.service.modeling.baselines import fit_baselines, score_baselines
from tests.test_service_model_baselines import baseline_fixture


def fixture():
    prepared, lists, config = baseline_fixture()
    prepared.logs["masks"] = {sample_id: True for sample_id in prepared.sample_ids}
    return prepared, lists, config


def test_availability_uses_fit_counts_and_scores_rare_patterns():
    prepared, lists, _ = fixture()
    state = fit_availability(prepared, lists.fit)
    assert state["counts"] == {"service": [0, 0, 0, 0, 0, 0, 0, 3]}
    normal = score_availability(state, prepared, ["test"])[0]
    prepared.logs["masks"]["test"] = False
    columns = prepared.numeric["masks"].columns[4:]
    prepared.numeric["masks"].loc[3, columns] = False
    rare = score_availability(state, prepared, ["test"])[0]
    assert normal == pytest.approx(-np.log(4 / 11))
    assert rare == pytest.approx(-np.log(1 / 11))
    assert rare > normal
    assert fit_availability(prepared, lists.fit) == state
    restored = copy.deepcopy(state)
    np.testing.assert_array_equal(
        score_availability(restored, prepared, ["test"]),
        score_availability(state, prepared, ["test"]),
    )


def test_availability_rejects_nonfit_or_abnormal_fit():
    prepared, lists, _ = fixture()
    with pytest.raises(ValueError, match="fit IDs"):
        fit_availability(prepared, ["calibration"])
    with pytest.raises(ValueError, match="fit IDs"):
        fit_availability(prepared, [lists.fit[0], lists.fit[0]])
    prepared.labels.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="known-normal"):
        fit_availability(prepared, lists.fit)


def test_availability_baseline_has_no_text_backbone_and_keeps_ids():
    prepared, lists, config = fixture()
    bundle = fit_baselines(prepared, lists, config, methods=("availability",))
    assert bundle["backbone"] is None
    assert set(bundle["models"]) == {"availability"}
    result = score_baselines(bundle, prepared, {"test"})["availability"]
    assert result.sample_id.tolist() == ["test"]
    assert result.score.iloc[0] == pytest.approx(-np.log(4 / 11))


def test_availability_rejects_unknown_service_and_nonboolean_log_mask():
    prepared, lists, _ = fixture()
    state = fit_availability(prepared, lists.fit)
    prepared.identity.loc[3, "canonical_service_id"] = "unseen"
    with pytest.raises(ValueError, match="service"):
        score_availability(state, prepared, ["test"])
    prepared.logs["masks"]["test"] = "False"
    with pytest.raises(ValueError, match="boolean"):
        score_availability(state, prepared, ["test"])
