import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.service.modeling.tail_fusion import fit_tail_fusion, score_tail_fusion
from tests.test_service_model_baselines import baseline_fixture


def fixture():
    prepared, lists, _ = baseline_fixture()
    frames = {
        "value": pd.DataFrame(dict(sample_id=lists.fit, score=[1.0, 2.0, 3.0])),
        "availability": pd.DataFrame(dict(sample_id=lists.fit, score=[0.0, 0.0, 0.0])),
    }
    return prepared, lists, frames


def test_tail_fusion_fits_normal_only_and_handles_ties_and_tails():
    prepared, lists, frames = fixture()
    state = fit_tail_fusion(prepared, frames, lists.fit)
    restored = json.loads(json.dumps(state))
    test_frames = {
        "value": pd.DataFrame(dict(sample_id=["test"], score=[4.0])),
        "availability": pd.DataFrame(dict(sample_id=["test"], score=[0.0])),
    }
    result = score_tail_fusion(restored, prepared, test_frames, ["test"])
    assert result.score.iloc[0] == pytest.approx(np.log(4))
    assert result.value_tail.iloc[0] == 0.25
    assert result.availability_tail.iloc[0] == 1.0
    test_frames["availability"].score = 1.0
    result = score_tail_fusion(state, prepared, test_frames, ["test"])
    assert result.score.iloc[0] == pytest.approx(2 * np.log(4))
    assert fit_tail_fusion(prepared, frames, lists.fit) == state


def test_tail_fusion_rejects_nonfit_abnormal_and_incomplete_channels():
    prepared, lists, frames = fixture()
    with pytest.raises(ValueError, match="complete frozen fit"):
        fit_tail_fusion(prepared, frames, ["test"])
    frames["value"] = frames["value"].iloc[:2]
    with pytest.raises(ValueError, match="exactly match"):
        fit_tail_fusion(prepared, frames, lists.fit)
    prepared, lists, frames = fixture()
    prepared.labels.loc[0, "label"] = 1
    with pytest.raises(ValueError, match="known-normal"):
        fit_tail_fusion(prepared, frames, lists.fit)


def test_tail_fusion_is_service_specific_and_rejects_unseen_service():
    prepared, lists, frames = fixture()
    prepared.identity.loc[0, "canonical_service_id"] = "other"
    state = fit_tail_fusion(prepared, frames, lists.fit)
    assert state["references"]["other"]["value"] == [1.0]
    state = copy.deepcopy(state)
    del state["references"]["service"]
    with pytest.raises(ValueError, match="Missing service"):
        score_tail_fusion(state, prepared, frames, lists.fit)


def test_tail_fusion_rejects_nonfinite_conflicting_identity_and_changed_weights():
    prepared, lists, frames = fixture()
    frames["value"]["canonical_service_id"] = "wrong"
    with pytest.raises(ValueError, match="Conflicting"):
        fit_tail_fusion(prepared, frames, lists.fit)
    frames["value"] = frames["value"].drop(columns="canonical_service_id")
    state = fit_tail_fusion(prepared, frames, lists.fit)
    frames["value"].loc[0, "score"] = np.inf
    with pytest.raises(ValueError, match="Nonfinite"):
        score_tail_fusion(state, prepared, frames, lists.fit)
    state["weights"]["value"] = 2.0
    with pytest.raises(ValueError, match="Unsupported"):
        score_tail_fusion(state, prepared, frames, lists.fit)
