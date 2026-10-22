from types import SimpleNamespace

import pandas as pd
import pytest

from src.service.modeling.data import build_sample_lists


def membership_fixture():
    ids = list("abcdefgh")
    grid = pd.DataFrame(
        {
            "sample_id": ids,
            "case_id": ["normal"] * 3 + ["fault"] * 5,
            "canonical_service_id": ["A"] * 7 + ["B"],
        }
    )
    labels = grid.assign(
        label=pd.array([0, 0, 0, 0, 1, None, 1, 1], dtype="Int64"),
        label_source=["normal_run_reference"] * 3
        + ["pre_intervention_reference", "positive", "unknown", "positive", "positive"],
    )
    splits = grid.assign(split=["fit", "calibration"] + ["test"] * 6)
    evaluation = grid.assign(
        evaluation_eligible=[False, False, True, False, True, False, True, True]
    )
    masks = pd.DataFrame(
        {"sample_id": ids, "cpu_cores": [True] * 6 + [False, True], "error_rate": False}
    )
    return (
        SimpleNamespace(grid=grid, labels=labels, splits=splits, evaluation=evaluation),
        {"masks": masks},
        pd.Series([False] * 6 + [True, False], index=ids),
    )


def test_common_and_extended_membership():
    tables, numeric, logs = membership_fixture()
    result = build_sample_lists(tables, numeric, logs, "common")
    assert result.common_fit == ("a",)
    assert result.common_calibration == ("b",)
    assert result.common_main_test == ("c", "e", "h")
    assert "g" in result.extended_inference and "g" not in result.common_main_test
    assert "f" not in result.common_fit + result.common_calibration + result.common_main_test
    assert result.extended_main_test == ("c", "e", "g", "h")
    assert result.fit == result.common_fit
    assert result.calibration == result.common_calibration
    assert result.main_test == result.common_main_test
    assert result.full_test == result.extended_main_test
    assert result.fit_diagnostic == result.common_fit
    assert result.exclusions == result.excluded
    assert build_sample_lists(tables, numeric, logs, "common").lists_hash == result.lists_hash


def test_fault_pre_excluded_from_main_but_kept_for_diagnostics():
    tables, numeric, logs = membership_fixture()
    result = build_sample_lists(tables, numeric, logs, "common")
    assert result.diagnostic_pre == ("d",)
    assert result.diagnostic_positive == ("e", "h")
    assert result.diagnostic_groups["fault|A"]["reference_available"]
    assert not result.diagnostic_groups["fault|B"]["reference_available"]
    assert "d" not in result.common_main_test
    numeric["masks"].loc[7, "cpu_cores"] = False
    logs.loc["h"] = False
    assert "h" in build_sample_lists(tables, numeric, logs, "common").excluded


def test_invalid_mode_rejected():
    tables, numeric, logs = membership_fixture()
    with pytest.raises(ValueError):
        build_sample_lists(tables, numeric, logs, "score_selected")


def test_fault_pre_never_enters_main_even_if_marked_eligible():
    tables, numeric, logs = membership_fixture()
    tables.evaluation.loc[tables.evaluation.sample_id.eq("d"), "evaluation_eligible"] = True
    result = build_sample_lists(tables, numeric, logs, "common")
    assert "d" not in result.common_main_test
    assert "d" not in result.extended_main_test
    assert "d" in result.diagnostic_pre
