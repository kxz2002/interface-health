"""Evidence rules: identity, admission and clock anchors precede label generation."""

import json
from pathlib import Path

import pytest

from src.service.labeling import evidence


def test_selected_attempt_binding(tmp_path):
    path = tmp_path / "metadata.json"
    meta = {
        "case_id": "selected",
        "attempt_id": "chosen",
        "run_id": "batch",
        "plan_hash": "plan",
        "plan_item_id": "item",
    }
    path.write_text(json.dumps(meta))
    expected = dict(meta)
    digest = evidence.file_digest(path)
    evidence.validate_selected_case(path, expected, digest)
    with pytest.raises(ValueError, match="identity"):
        evidence.validate_selected_case(path, {**expected, "attempt_id": "failed"}, digest)
    path.write_text(json.dumps({**meta, "attempt_id": "failed"}))
    with pytest.raises(ValueError, match="hash"):
        evidence.validate_selected_case(path, expected, digest)


def test_admission_is_not_health_gold():
    quality = dict(
        collection_status="complete",
        coverage_status="passed",
        processing_status="passed",
        errors=[],
        cleanup_status="passed",
        recovery={"terminal_status": "passed"},
    )
    load = dict(
        errors=[],
        status="complete",
        attempt_health={"healthy": True},
        failures=2,
        http_status=200,
        log_level="ERROR",
    )
    result = evidence.run_admission(quality, load, is_fault=False)
    assert result["admitted"] is True
    assert "load_failures_unlocalized" in result["warnings"]
    quality["errors"] = ["collector failed"]
    assert evidence.run_admission(quality, load, is_fault=False)["admitted"] is False
    quality["errors"] = []
    quality["cleanup_status"] = "failed"
    assert not evidence.run_admission(quality, load, is_fault=False)["admitted"]
    assert evidence.run_admission(quality, load, is_fault=True)["admitted"]


def test_pause_anchor_precedes_confirmation():
    signal = dict(
        start_ms=100000,
        start_mono_ns=1000000000,
        probe_pause_at_mono_ns=31000000000,
        probe_write_policy="baseline-recover-v1",
    )
    events = [
        {
            "event_type": "probe_write_pause_verified",
            "utc_ms": 150000,
            "payload": {"pause_at_mono_ns": 31000000000},
        }
    ]
    result = evidence.intervention_boundary({"apply_requested_ms": 160000}, signal, events)
    assert result["first_intervention_requested_ms"] == 130000
    events[0]["payload"]["pause_at_mono_ns"] += 1
    assert (
        evidence.intervention_boundary({"apply_requested_ms": 160000}, signal, events)[
            "first_intervention_requested_ms"
        ]
        is None
    )


def test_optional_pause_vs_missing_required_anchor():
    fault = {"apply_requested_ms": 160000}
    no_pause = {"probe_write_policy": "disabled", "probe_pause_at_mono_ns": None}
    assert (
        evidence.intervention_boundary(fault, no_pause, [])["first_intervention_requested_ms"]
        == 160000
    )
    missing = {"probe_write_policy": "baseline-recover-v1"}
    result = evidence.intervention_boundary(fault, missing, [])
    assert result["first_intervention_requested_ms"] is None
    assert "missing_pause_anchor" in result["reasons"]
    assert evidence.intervention_boundary(fault, {}, [])["first_intervention_requested_ms"] is None


def test_only_issued_load_exception_has_interval(tmp_path):
    base = dict(
        case_id="case",
        source="load",
        event="end",
        transport_started=True,
        outcome="exception",
        start_ms=100000,
        end_ms=130000,
    )
    events = [
        base,
        {**base, "outcome": "suppressed"},
        {**base, "transport_started": False},
        {**base, "source": "probe"},
        {**base, "event": "begin"},
        {**base, "end_ms": None},
    ]
    path = tmp_path / "api_attempts.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in events))
    intervals, warnings = evidence.load_transport_exceptions(path, "case", "source")
    assert len(intervals) == 1
    assert intervals[0]["start_ms"] == 100000
    assert intervals[0]["source_locator"] == "line:1"
    assert warnings
    path.write_text(json.dumps({**base, "case_id": "failed"}) + "\n")
    with pytest.raises(ValueError, match="case"):
        evidence.load_transport_exceptions(path, "case", "source")


def test_conflicting_measurement_anchor_is_unknown():
    signal = dict(
        start_ms=100000,
        start_mono_ns=1000000000,
        probe_pause_at_mono_ns=31000000000,
        probe_write_policy="baseline-recover-v1",
    )
    events = [{"event_type": "measurement_started", "payload": {**signal, "start_ms": 100001}}]
    result = evidence.intervention_boundary({"apply_requested_ms": 160000}, signal, events)
    assert result["first_intervention_requested_ms"] is None
    assert "pause_clock_conflict" in result["reasons"]


def test_invalid_delete_request_does_not_silently_disappear():
    result = evidence.intervention_boundary(
        {"apply_requested_ms": 160000},
        {"probe_write_policy": "disabled"},
        [{"event_type": "single_delete_requested", "utc_ms": None}],
    )
    assert result["first_intervention_requested_ms"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"errors": ["collection failure"]},
        {"status": "unknown"},
        {"attempt_health": {"healthy": False}},
        {"attempt_health": {}},
    ],
)
def test_missing_or_failed_load_collection_cannot_admit(change):
    quality = dict(
        collection_status="complete",
        coverage_status="passed",
        processing_status="passed",
        errors=[],
    )
    load = dict(errors=[], status="complete", attempt_health={"healthy": True}, failures=0)
    assert not evidence.run_admission(quality, {**load, **change}, is_fault=True)["admitted"]


def test_transport_boolean_time_is_not_integer_interval(tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(
        json.dumps(
            dict(
                case_id="case",
                source="load",
                event="end",
                transport_started=True,
                outcome="exception",
                start_ms=True,
                end_ms=100000,
            )
        )
        + "\n"
    )
    intervals, warnings = evidence.load_transport_exceptions(path, "case", "source")
    assert intervals == []
    assert warnings == ["invalid_exception_interval:line:1"]


@pytest.mark.parametrize("start,end", [(True, 150), (150, 140), (90, 160), (150, 210)])
def test_execution_range_integrity(start, end):
    with pytest.raises(ValueError, match="execution"):
        evidence.validate_execution_interval(start, end, 100, 200)


def test_missing_and_instantaneous_execution_is_not_invented():
    assert evidence.validate_execution_interval(None, None, 100, 200) is None
    assert evidence.validate_execution_interval(150, 150, 100, 200) == (150, 150)


@pytest.mark.parametrize("source", ["apply", "delete"])
@pytest.mark.parametrize("request_ms", [99000, 300000])
def test_intervention_request_outside_case_raises(source, request_ms):
    signal = {"start_ms": 100000, "end_ms": 200000, "probe_write_policy": "disabled"}
    fault = {"effective_ms": 150000, "apply_requested_ms": 140000}
    events = []
    if source == "apply":
        fault["apply_requested_ms"] = request_ms
    else:
        events = [{"event_type": "single_delete_requested", "utc_ms": request_ms}]
    with pytest.raises(ValueError, match="request.*case range"):
        evidence.intervention_boundary(fault, signal, events)


@pytest.mark.parametrize("source", ["apply", "delete"])
def test_intervention_request_after_execution_start_raises(source):
    signal = {"start_ms": 100000, "end_ms": 200000, "probe_write_policy": "disabled"}
    fault = {"effective_ms": 150000, "apply_requested_ms": 140000}
    events = []
    if source == "apply":
        fault["apply_requested_ms"] = 160000
    else:
        events = [{"event_type": "single_delete_requested", "utc_ms": 160000}]
    with pytest.raises(ValueError, match="request.*execution start"):
        evidence.intervention_boundary(fault, signal, events)


def test_valid_delete_request_can_precede_apply_and_equal_execution_start():
    signal = {"start_ms": 100000, "end_ms": 200000, "probe_write_policy": "disabled"}
    fault = {"effective_ms": 150000, "apply_requested_ms": 150000}
    events = [{"event_type": "single_delete_requested", "utc_ms": 130000}]
    assert (
        evidence.intervention_boundary(fault, signal, events)["first_intervention_requested_ms"]
        == 130000
    )
