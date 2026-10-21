"""Versioned controlled exposure and fixed evaluation boundaries."""

from copy import deepcopy

import pandas as pd
import pytest

from src.service.labeling.acceptance import join_evaluation_scores
from src.service.labeling.evaluation import evaluation_table
from src.service.labeling.evidence import EvidenceIndex
from src.service.labeling.expansion import lifecycle_intervals, network_intervals
from src.service.labeling.rules import classify_windows


def network_fixture():
    spec = {"action": "delay", "direction": "to"}
    fault = dict(
        effective_ms=2000,
        removed_ms=32000,
        document={"spec": spec},
        objects=[
            dict(
                kind="NetworkChaos",
                spec=spec,
                status={
                    "experiment": {
                        "containerRecords": [
                            dict(
                                selectorKey=".Target",
                                phase="Injected",
                                id="default/pod",
                                events=[
                                    dict(
                                        operation="Apply",
                                        type="Succeeded",
                                        timestamp="1970-01-01T00:00:01Z",
                                    )
                                ],
                            )
                        ]
                    }
                },
            )
        ],
    )
    runtime = {
        "identity": {
            "records": [
                dict(
                    namespace="default",
                    pod_name="pod",
                    pod_uid="uid",
                    container_role="regular",
                    services=["svc"],
                )
            ]
        }
    }
    return fault, runtime


def test_network_requires_success_unique_identity_and_start_before_effective():
    fault, runtime = network_fixture()
    assert network_intervals(fault, runtime, ["svc"])[0]["start_ms"] == 2000
    assert network_intervals(fault, runtime, ["other"]) == []
    conflicting = deepcopy(runtime)
    conflicting["identity"]["records"].append(
        dict(
            namespace="default",
            pod_name="pod",
            pod_uid="new",
            container_role="regular",
            services=["svc"],
        )
    )
    assert network_intervals(fault, conflicting, ["svc"]) == []
    fault["effective_ms"] = 500
    assert network_intervals(fault, runtime, ["svc"]) == []


def snapshot(timestamp, pod=True, route=True):
    return dict(
        event_type="identity_snapshot",
        received_ms=timestamp,
        records=[
            dict(
                pod_uid="uid",
                services=["svc"],
                container_role="regular",
                pod_ready=pod,
                container_ready=pod,
                phase="Running",
            )
        ],
        endpoint_slices=[dict(service="svc", endpoints=[dict(ready=route)])],
    )


def test_lifecycle_only_joint_absence_and_observed_core():
    events = [
        snapshot(0),
        snapshot(1000, True, False),
        snapshot(2000, False, False),
        snapshot(12000, False, False),
        snapshot(22000, False, False),
        snapshot(25000),
    ]
    latest = {
        "observation_gaps": [],
        "relationship_observation": {"status": "current", "errors": []},
    }
    intervals = lifecycle_intervals(events, latest, "svc", 1000, 15000)
    assert [(row["start_ms"], row["end_ms"]) for row in intervals] == [(2000, 22000)]
    assert lifecycle_intervals(events, {"observation_gaps": [{}]}, "svc", 1000, 15000) == []
    assert lifecycle_intervals(events, latest, "svc", 1000, 5000) == []
    assert lifecycle_intervals(events[1:], latest, "svc", 1000, 15000) == []
    assert lifecycle_intervals(events[:-1], latest, "svc", 1000, 15000) == []
    later = events + [snapshot(30000, False, False), snapshot(40000, False, False), snapshot(45000)]
    assert len(lifecycle_intervals(later, latest, "svc", 1000, 15000)) == 1


def test_lifecycle_rejects_unordered_or_missing_observation():
    latest = {
        "observation_gaps": [],
        "relationship_observation": {"status": "current", "errors": []},
    }
    with pytest.raises(ValueError, match="Unordered"):
        lifecycle_intervals([snapshot(1), snapshot(0)], latest, "svc", 0, 15000)
    event = snapshot(1)
    del event["endpoint_slices"]
    with pytest.raises(KeyError):
        lifecycle_intervals([event], latest, "svc", 0, 15000)


@pytest.mark.parametrize(
    "events",
    [
        [
            snapshot(0),
            snapshot(31000, False, False),
            snapshot(41000, False, False),
            snapshot(42000),
        ],
        [
            snapshot(0),
            snapshot(10000),
            snapshot(20000),
            snapshot(31000, False, False),
            snapshot(41000, False, False),
            snapshot(42000),
        ],
        [snapshot(0), snapshot(2000, False, False), snapshot(12000, False, False), snapshot(43000)],
    ],
)
def test_lifecycle_rejects_stale_entry_or_exit_observations(events):
    latest = {
        "observation_gaps": [],
        "relationship_observation": {"status": "current", "errors": []},
    }
    assert lifecycle_intervals(events, latest, "svc", 1000, 30000) == []


def test_evaluation_excludes_pre_unknown_fit_and_calibration():
    rows = []
    sources = [
        "normal_run_reference",
        "pre_intervention_reference",
        "unknown",
        "network_participant_association",
        "normal_run_reference",
        "normal_run_reference",
    ]
    for index, source in enumerate(sources):
        rows.append(
            dict(
                dataset_id="tt",
                run_id=str(index),
                case_id=str(index),
                canonical_service_id="svc",
                window_start_ms=0,
                window_end_ms=15000,
                sample_id=str(index),
                label=[0, 0, pd.NA, 1, 0, 0][index],
                label_source=source,
            )
        )
    labels = pd.DataFrame(rows)
    splits = labels.copy()
    splits["split"] = ["test", "test", "test", "test", "fit", "calibration"]
    result = evaluation_table(labels, splits, {"evaluation_version": "eval-v2"})
    assert set(result.loc[result.evaluation_eligible, "sample_id"]) == {"0", "3"}
    assert result.evaluation_version.eq("eval-v2").all()
    labels.attrs.update(label_version="label-v2", split_version="split-v1")
    splits = splits.drop(columns=["label", "label_source"])
    splits.attrs.update(labels.attrs)
    data = dict(
        labels=labels, splits=splits, protocol={"evaluation_version": "eval-v2"}, evaluation=result
    )
    scores = labels[["sample_id"]].copy()
    scores["score"] = 0.5
    scores.attrs.update(labels.attrs, evaluation_version="eval-v2")
    joined = join_evaluation_scores(data, scores, set(labels.sample_id))
    assert set(joined.sample_id) == {"0", "3"}
    with pytest.raises(ValueError, match="Missing required scores"):
        join_evaluation_scores(data, scores[scores.sample_id.ne("3")], set(labels.sample_id))


def test_expansion_requires_containment_and_admission_without_cache_propagation():
    rows = [
        dict(
            dataset_id="tt",
            run_id="fault",
            case_id="fault",
            canonical_service_id=service,
            window_start_ms=timestamp,
            window_end_ms=timestamp + 15000,
            sample_id=f"{service}-{timestamp}",
        )
        for service in ["svc", "other"]
        for timestamp in [0, 15000, 30000, 45000]
    ]
    grid = pd.DataFrame(rows)
    protocol = dict(
        window_ms=15000,
        fault_split="test",
        normal_splits={},
        split_version="s",
        expansion_rule="network_and_joint_lifecycle_v1",
    )
    case = dict(
        is_fault=True,
        admission={"admitted": True},
        boundary={"first_intervention_requested_ms": 10000, "reasons": []},
        candidate_services=["svc", "other"],
        evidence_ids=[],
        transport_exceptions=[],
        fault={
            "effective_ms": 10000,
            "removed_ms": 50000,
            "document": {
                "metadata": {"labels": {"target_service": "mysql"}},
                "spec": {"action": "stress"},
            },
        },
        expansion_intervals=[
            dict(
                service="svc",
                start_ms=10000,
                end_ms=50000,
                source="network_participant_association",
            )
        ],
    )

    def classify():
        index = EvidenceIndex(grid, {"fault": case}, {}, protocol, "m")
        return classify_windows(grid, index, protocol).labels

    labels = classify()
    assert set(labels.loc[labels.label.eq(1), "sample_id"]) == {"svc-15000", "svc-30000"}
    case["admission"]["admitted"] = False
    assert classify().label.isna().all()
    case["admission"]["admitted"] = True
    case["expansion_intervals"] = []
    assert not classify().label.eq(1).any()
