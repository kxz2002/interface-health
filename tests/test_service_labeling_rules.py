"""Behavioral boundaries for pure injection-association labels."""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.service.labeling.evidence import EvidenceIndex
from src.service.labeling.rules import build_splits, classify_windows


@pytest.fixture
def protocol():
    return json.loads(Path("configs/service/tt_labels_a_v1.json").read_text())


def inputs(protocol, normal=False):
    cid = next(iter(protocol["normal_splits"])) if normal else "fault"
    grid = pd.DataFrame(
        [
            dict(
                dataset_id="tt",
                run_id=cid,
                case_id=cid,
                canonical_service_id=s,
                window_start_ms=t,
                window_end_ms=t + 15000,
                sample_id=f"{cid}-{s}-{t}",
                resource_valid=False,
            )
            for s in ["target", "other"]
            for t in range(0, 75000, 15000)
        ]
    )
    case = dict(
        is_fault=not normal,
        admission={"admitted": True},
        boundary={"first_intervention_requested_ms": 15000, "reasons": []},
        fault={
            "effective_ms": 15000,
            "removed_ms": 45000,
            "document": {
                "metadata": {"labels": {"target_service": "target"}},
                "spec": {"action": "stress"},
            },
        },
        candidate_services=["target", "other"],
        transport_exceptions=[],
        evidence_ids=["control"],
    )
    return grid, EvidenceIndex(grid, {cid: case}, {}, protocol, "manifest"), case


def at(labels, service, time):
    return labels.loc[
        (labels.canonical_service_id == service) & (labels.window_start_ms == time)
    ].iloc[0]


def test_full_containment(protocol):
    grid, evidence, case = inputs(protocol)
    labels = classify_windows(grid, evidence, protocol).labels
    assert at(labels, "target", 15000).label == 1
    assert at(labels, "target", 30000).label == 1
    assert pd.isna(at(labels, "other", 15000).label)
    case["fault"]["effective_ms"] = 16000
    labels = classify_windows(grid, evidence, protocol).labels
    assert pd.isna(at(labels, "target", 15000).label)
    assert at(labels, "target", 15000).label_reason == "partial_boundary"


def test_pre_intervention_negative(protocol):
    grid, evidence, case = inputs(protocol)
    case["fault"]["effective_ms"] = 30000
    labels = classify_windows(grid, evidence, protocol).labels
    assert at(labels, "other", 0).label == 0
    assert pd.isna(at(labels, "other", 15000).label)
    assert pd.isna(at(labels, "target", 45000).label)
    case["boundary"] = {
        "first_intervention_requested_ms": None,
        "reasons": ["unknown_pause_policy"],
    }
    assert classify_windows(grid, evidence, protocol).labels.label.eq(0).sum() == 0


@pytest.mark.parametrize("kind", ["kill", "database", "zero"])
def test_instant_and_database(protocol, kind):
    grid, evidence, case = inputs(protocol)
    if kind == "kill":
        case["fault"]["document"]["spec"]["action"] = "pod-kill"
    elif kind == "database":
        case["fault"]["document"]["metadata"]["labels"]["target_service"] = "database"
    else:
        case["fault"]["removed_ms"] = 15000
    tables = classify_windows(grid, evidence, protocol)
    assert len(tables.labels) == len(grid)
    assert tables.labels.label.eq(1).sum() == 0
    assert tables.labels.label.eq(0).sum() == 2


def test_timeout_all_services(protocol):
    grid, evidence, case = inputs(protocol, normal=True)
    case["transport_exceptions"] = [dict(start_ms=20000, end_ms=31000, evidence_id="timeout")]
    labels = classify_windows(grid, evidence, protocol).labels
    assert labels.loc[labels.window_start_ms.isin([15000, 30000]), "label"].isna().all()
    assert labels.loc[labels.window_start_ms.isin([0, 45000]), "label"].eq(0).all()
    assert "timeout" in at(labels, "other", 15000).evidence_ids
    case["transport_exceptions"][0].update(start_ms=15000, end_ms=30000)
    assert at(classify_windows(grid, evidence, protocol).labels, "other", 30000).label == 0


def test_cross_intervention_timeout(protocol):
    grid, evidence, case = inputs(protocol)
    case["transport_exceptions"] = [dict(start_ms=10000, end_ms=20000, evidence_id="cross")]
    assert at(classify_windows(grid, evidence, protocol).labels, "other", 0).label == 0


def test_fixed_split_and_mask(protocol):
    grid, evidence, case = inputs(protocol, normal=True)
    tables = classify_windows(grid, evidence, protocol)
    assert str(tables.labels.label.dtype) == "Int8"
    assert tables.labels.label.eq(0).all()  # invalid resource is irrelevant
    assert tables.splits.split.eq("fit").all()
    assert tables.labels.sample_id.tolist() == grid.sample_id.tolist()
    changed = deepcopy(protocol)
    cid = grid.case_id.iloc[0]
    changed["normal_splits"][cid] = "test"
    assert build_splits(grid, changed).split_version.iloc[0] != tables.splits.split_version.iloc[0]
    case["admission"]["admitted"] = False
    tables = classify_windows(grid, evidence, protocol)
    assert tables.labels.label.isna().all()
    assert tables.splits.split.eq("fit").all()


def test_duplicate_and_conflict_rejected(protocol):
    grid, evidence, case = inputs(protocol)
    with pytest.raises(ValueError, match="duplicate"):
        classify_windows(pd.concat([grid, grid]), evidence, protocol)
    case["boundary"]["first_intervention_requested_ms"] = 45000
    with pytest.raises(ValueError, match="conflict"):
        classify_windows(grid, evidence, protocol)


def test_reason_priority_and_deterministic_splits(protocol):
    grid, evidence, case = inputs(protocol)
    case["admission"]["admitted"] = False
    case["fault"]["document"]["spec"]["action"] = "pod-kill"
    case["fault"]["document"]["metadata"]["labels"]["target_service"] = "database"
    tables = classify_windows(grid, evidence, protocol)
    row = at(tables.labels, "target", 15000)
    assert row.label_reason == "run_admission_failed"
    assert row.label_reasons == [
        "run_admission_failed",
        "instantaneous_operation",
        "target_outside_candidates",
    ]
    assert row.evidence_ids == ["control"]
    reordered = build_splits(grid.iloc[::-1], protocol).sort_values("sample_id")
    pd.testing.assert_frame_equal(tables.splits.sort_values("sample_id"), reordered)


def test_completed_pre_exception_and_fault_test_role(protocol):
    grid, evidence, case = inputs(protocol)
    case["transport_exceptions"] = [dict(start_ms=0, end_ms=15000, evidence_id="pre")]
    tables = classify_windows(grid, evidence, protocol)
    assert at(tables.labels, "target", 0).label_reason == "negative_exception"
    assert tables.splits.split.eq("test").all()
    assert len(tables.labels) == len(tables.splits) == len(grid)


def test_grid_cannot_drop_validated_windows(protocol):
    grid, evidence, _ = inputs(protocol)
    with pytest.raises(ValueError, match="validated grid"):
        classify_windows(grid.iloc[:-1], evidence, protocol)
