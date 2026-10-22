from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.service.modeling.baselines import fit_baselines, score_baselines
from src.service.modeling.config import content_hash
from src.service.modeling.data import SampleLists
from tests.test_service_model_training import fixture


def hashed(lists):
    payload = asdict(lists)
    payload.pop("lists_hash")
    return replace(lists, lists_hash=content_hash(payload))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, attention_mask):
        values = input_ids.float() * self.weight
        return SimpleNamespace(last_hidden_state=torch.stack([values, values * 2], -1))


def baseline_fixture():
    prepared, lists, config = fixture(3)
    lists = hashed(
        SampleLists(
            lists.fit, ("calibration",), (), ("test",), (), (), (), {}, (), {}, "", "common"
        )
    )
    prepared.lists = lists
    for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]:
        prepared.logs[name] = {
            sample_id: (
                [[1, index + 2, 3]]
                if name == "input_ids"
                else (
                    [[1, 1, 1]]
                    if name == "attention_mask"
                    else [[1, 0, 1]] if name == "special_mask" else ["template:a"]
                )
            )
            for index, sample_id in enumerate(prepared.sample_ids)
        }
    return prepared, lists, config


def test_rules_masks_and_frozen_text_baseline():
    prepared, lists, config = baseline_fixture()
    column = prepared.numeric["values"].columns[1]
    prepared.numeric["masks"].loc[3, column] = False
    prepared.numeric["values"].loc[3, column] = 999999
    backbone = TinyBackbone()
    bundle = fit_baselines(prepared, lists, config, text_backbone=backbone)
    scores = score_baselines(bundle, prepared, {"test"})
    assert scores["L2"].score.iloc[0] == pytest.approx(np.sqrt(7 * 16))
    assert scores["Max"].score.iloc[0] == 4
    assert scores["L2"].valid_dimensions.iloc[0] == 7
    assert all(not parameter.requires_grad for parameter in bundle["backbone"].parameters())
    assert backbone.weight.requires_grad
    np.testing.assert_allclose(bundle["text_mean"], [3, 6])
    before = bundle["backbone"].weight.detach().clone()
    score_baselines(bundle, prepared, {"calibration"})
    assert torch.equal(before, bundle["backbone"].weight)
    prepared.numeric["masks"].loc[3, prepared.numeric["masks"].columns[1:]] = False
    missing = score_baselines(bundle, prepared, {"test"})
    assert missing["L2"].empty and missing["Max"].empty
    assert missing["L2"].attrs["excluded_ids"] == ["test"]


def test_params_and_score_orientation():
    prepared, lists, config = baseline_fixture()
    bundle = fit_baselines(prepared, lists, config, text_backbone=TinyBackbone())
    model = bundle["models"]["numeric_IF"]
    assert model.n_estimators == 100 and model.n_jobs == 1 and model.random_state == 42
    svm = bundle["models"]["numeric_OCSVM"]
    assert svm.kernel == "rbf" and svm.gamma == "scale" and svm.nu == 0.1
    result = score_baselines(bundle, prepared, {"test"})
    values = prepared.numeric["values"].set_index("sample_id").loc[["test"]].to_numpy()
    features = np.concatenate([values, np.ones_like(values)], axis=1)
    assert result["numeric_IF"].score.iloc[0] == pytest.approx(-model.score_samples(features)[0])
    assert result["numeric_OCSVM"].score.iloc[0] == pytest.approx(
        -svm.decision_function(features)[0]
    )


def test_fit_text_statistics_exclude_missing_and_never_use_test():
    prepared, lists, config = baseline_fixture()
    for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]:
        prepared.logs[name]["fit-001"] = []
    prepared.logs["input_ids"]["test"] = [[1, 900000, 3]]
    bundle = fit_baselines(prepared, lists, config, text_backbone=TinyBackbone())
    np.testing.assert_allclose(bundle["text_mean"], [3, 6])
    np.testing.assert_allclose(bundle["text_scale"], [1, 2])
    assert bundle["text_fit_valid"]
    for sample_id in lists.fit:
        for name in ["input_ids", "attention_mask", "special_mask", "group_ids"]:
            prepared.logs[name][sample_id] = []
    no_text = fit_baselines(prepared, lists, config, text_backbone=TinyBackbone())
    assert not no_text["text_fit_valid"]
    np.testing.assert_array_equal(no_text["text_mean"], [0, 0])
    np.testing.assert_array_equal(no_text["text_scale"], [1, 1])


def test_template_balanced_pooling_uses_body_tokens_and_same_four_statistics():
    from src.service.modeling.baselines import _features, _pool

    prepared, lists, config = baseline_fixture()
    prepared.logs["input_ids"]["fit-000"] = [[100, 2, 200], [100, 4, 200], [100, 9, 200]]
    prepared.logs["attention_mask"]["fit-000"] = [[1, 1, 1]] * 3
    prepared.logs["special_mask"]["fit-000"] = [[1, 0, 1]] * 3
    prepared.logs["group_ids"]["fit-000"] = ["template:a", "template:a", "unmatched:b"]
    with torch.no_grad():
        vector = _pool(TinyBackbone(), prepared.logs, "fit-000")
    np.testing.assert_allclose(vector, [6, 12])
    bundle = fit_baselines(prepared, lists, config, text_backbone=TinyBackbone())
    prepared.logs["statistics"]["values"].loc[3, "s0"] = 7
    _, combined, _, _, _ = _features(bundle, prepared, ["test"])
    assert combined[0, -8] == 7
    prepared.logs["statistics"]["masks"].loc[3, "s0"] = False
    _, combined, _, _, _ = _features(bundle, prepared, ["test"])
    assert combined[0, -8] == 0 and combined[0, -4] == 0


def test_method_support_lists_are_frozen_before_scoring():
    from src.service.modeling.baselines import baseline_sample_lists
    from src.service.modeling.data import SampleLists

    prepared, original, config = baseline_fixture()
    lists = SampleLists(
        common_fit=original.fit,
        common_calibration=("calibration",),
        common_main_test=(),
        extended_inference=("test",),
        extended_main_test=(),
        diagnostic_pre=(),
        diagnostic_positive=(),
        diagnostic_groups={},
        excluded=(),
        reasons={},
        lists_hash="frozen",
        mode="extended",
    )
    lists = hashed(lists)
    prepared.lists = lists
    prepared.numeric["masks"].loc[3, prepared.numeric["masks"].columns[1:]] = False
    bundle = fit_baselines(prepared, lists, config, text_backbone=TinyBackbone())
    supported = baseline_sample_lists(bundle, prepared, lists)
    assert supported["numeric"].extended_inference == ()
    assert supported["trimodal"].extended_inference == ("test",)
    assert supported["numeric"].fit == lists.fit
    assert supported["numeric"].calibration == lists.calibration
    assert lists.extended_inference == ("test",)
    result = score_baselines(bundle, prepared, {"test"})
    assert result["numeric_IF"].empty
    assert set(result["trimodal_IF"].sample_id) == set(supported["trimodal"].extended_inference)
    assert result["trimodal_IF"].lists_hash.eq(bundle["method_lists"]["trimodal"].lists_hash).all()
    assert (
        result["trimodal_IF"].attrs["model_version"] == result["trimodal_IF"].model_version.iloc[0]
    )
