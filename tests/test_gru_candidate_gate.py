import copy

import pytest

from scripts.audit_gru_candidates import residual_normal_gate


def records():
    return [
        dict(candidate=candidate, seed=seed, validation_loss=loss, value_loss=loss - 0.1)
        for candidate, loss in [("baseline", 1.0), ("residual", 0.9)]
        for seed in [1, 2, 42]
    ]


def test_normal_gate_requires_each_seed_to_improve_both_components():
    rows = records()
    assert residual_normal_gate(rows)["passed"]
    changed = copy.deepcopy(rows)
    changed[-1]["validation_loss"] = 1.01
    assert not residual_normal_gate(changed)["passed"]
    changed = copy.deepcopy(rows)
    changed[-1]["value_loss"] = 0.91
    assert not residual_normal_gate(changed)["passed"]


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan"])
def test_normal_gate_rejects_incomplete_or_nonfinite_evidence(problem):
    rows = records()
    if problem == "missing":
        rows.pop()
    elif problem == "duplicate":
        rows.append(rows[-1])
    else:
        rows[-1]["validation_loss"] = float("nan")
    with pytest.raises(ValueError):
        residual_normal_gate(rows)
