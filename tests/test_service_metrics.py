"""Resource semantics: fixtures catch cross-series differences and fabricated health."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.service.protocol import load_protocol, validate_features

SVC = "ts-order-service"
CASE = "Normal_20260728T161044Z"
CPU = "container_cpu_usage_seconds_total"


@pytest.fixture
def protocol():
    protocol = load_protocol(
        Path("configs/service/minimal_v1.yaml"), Path("configs/data/new_merge.yaml")
    )
    # Translate only fixture observation bounds; retain frozen entity/schema rules.
    protocol.config["cases"][0]["window_start_ms"] = 0
    protocol.config["cases"][0]["window_end_ms"] = 120000
    return protocol


def row(t, value, metric=CPU, **kw):
    return dict(
        metric_name=metric,
        timestamp=t,
        value=value,
        namespace="default",
        pod=SVC,
        container=SVC,
        id="old",
        instance="exporter",
        job="kubelet-cadvisor",
        cpu="total" if metric == CPU else "",
        **kw,
    )


def build(tmp_path, protocol, rows, start=0, end=120000):
    path = tmp_path / CASE / "metric_data" / "fixture.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    mod = importlib.import_module("src.service.metrics")
    frame, quality = mod.build_metric_windows(
        [path],
        dict(case_id=CASE, window_start_ms=start, window_end_ms=end, is_anomaly_case=False),
        protocol,
    )
    validate_features(frame, protocol)
    return frame[frame.canonical_service_id.eq(SVC)].set_index("window_start_ms"), quality


def test_counter_uses_id_and_real_dt(tmp_path, protocol):
    rows = [row(t, v) for t, v in [(0, 10), (15, 13), (40, 18), (55, 18), (70, 2), (101, 5)]]
    f, q = build(tmp_path, protocol, rows)
    assert f.loc[0, "cpu_cores__status"] == "UNOBSERVED"
    assert f.loc[15000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[30000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[45000, "cpu_cores"] == 0
    assert f.loc[60000, "cpu_cores__status"] == "COUNTER_RESET"
    assert f.loc[90000, "cpu_cores__status"] == "GAP"
    assert q.loc[q.query_time_ms.eq(40000), "counter_dt_s"].iloc[0] == 25
    assert not any("ready" in c.lower() for c in f)


def test_container_isolation_and_ratio(tmp_path, protocol):
    rows = [row(0, 10), row(15, 13)]
    for t in [15, 30, 45]:
        rows += [
            row(t, 240, "container_memory_usage_bytes"),
            row(t, 200 if t != 30 else 0, "container_spec_memory_limit_bytes"),
        ]
    rows += [row(15, 0, "container_processes")]
    parent = row(15, 999, "container_memory_usage_bytes")
    parent["container"] = ""
    rows += [parent]
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[15000, "memory_usage_ratio"] == 1.2
    assert f.loc[15000, "process_count"] == 0
    assert f.loc[30000, "memory_usage_ratio__status"] == "NO_DENOMINATOR"
    assert np.isnan(f.loc[45000, "process_count"])


def test_duplicate_conflict(tmp_path, protocol):
    rows = [
        row(0, 10),
        row(15, 13),
        row(15, 13),
        row(30, 16),
        row(30, 99),
        row(35, 17),
        row(45, 20),
    ]
    f, q = build(tmp_path, protocol, rows)
    assert f.loc[15000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[30000, "cpu_cores__status"] == "CONFLICTING_SAMPLE"
    assert q.dedup_count.max() == 2
    assert q.loc[q.query_time_ms.eq(45000), "counter_dt_s"].iloc[0] == 10


def test_old_new_identity_is_unknown(tmp_path, protocol):
    rows = [row(0, 10), row(15, 13), row(30, 16)]
    for t, v in [(30, 1), (45, 4), (60, 7), (75, 10)]:
        r = row(t, v)
        r["id"] = "new"
        rows.append(r)
        r = row(t, 1, "container_processes")
        r["id"] = "new"
        rows.append(r)
    f, q = build(tmp_path, protocol, rows)
    assert f.loc[30000, "cpu_cores__status"] == "IDENTITY_AMBIGUOUS"
    assert f.loc[45000, "cpu_cores__status"] == "IDENTITY_AMBIGUOUS"
    assert f.loc[60000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[30000, "observed_instance_count"] == 2
    assert (
        q.loc[
            (q.container_id.eq("new")) & q.query_time_ms.eq(30000) & q.metric_name.eq(CPU),
            "series_status",
        ].iloc[0]
        == "UNOBSERVED"
    )


def test_transition_without_overlap(tmp_path, protocol):
    rows = [row(0, 10), row(15, 13)]
    for t, v in [(60, 1), (75, 4), (90, 7)]:
        r = row(t, v)
        r["id"] = "new"
        rows.append(r)
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[60000, "cpu_cores__status"] == "TRANSITION"
    assert f.loc[75000, "cpu_cores__status"] == "TRANSITION"
    assert f.loc[90000, "cpu_cores"] == pytest.approx(0.2)


def test_missing_window_retained(tmp_path, protocol):
    f, q = build(tmp_path, protocol, [row(0, 10)], start=1000, end=61000)
    assert f.index.tolist() == [15000, 30000, 45000]
    assert f.resource_valid.eq(False).all()
    assert f.observed_instance_count.eq(0).all()
    assert f.attrs["partial_windows"] == {"leading_partial_ms": 14000, "trailing_partial_ms": 1000}


def test_cpu_total_preference_and_full_series(tmp_path, protocol):
    rows = [row(0, 10), row(15, 13)]
    for t, v in [(0, 100), (15, 200)]:
        r = row(t, v)
        r["cpu"] = ""
        rows.append(r)
    r = row(30, 50)
    r["instance"] = "other"
    rows.append(r)
    f, q = build(tmp_path, protocol, rows)
    assert f.loc[15000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[30000, "cpu_cores__status"] == "UNOBSERVED"


def test_missing_denominator_stays_unobserved(tmp_path, protocol):
    rows = [row(15, 100, "container_memory_usage_bytes")]
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[15000, "memory_usage_ratio__status"] == "UNOBSERVED"


def test_empty_identity_cpu_unsupported(tmp_path, protocol):
    rows = []
    for t, v in [(0, 10), (15, 13)]:
        r = row(t, v)
        r["id"] = ""
        rows.append(r)
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[15000, "cpu_cores__status"] == "UNSUPPORTED"
    assert np.isnan(f.loc[15000, "cpu_cores"])


def test_predecessor_budget_and_no_future_cpu_label(tmp_path, protocol):
    rows = []
    for t, v in [(0, 10), (15, 13), (30, 16), (45, 19)]:
        r = row(t, v)
        r["cpu"] = ""
        rows.append(r)
    rows += [row(60, 30), row(75, 33)]
    f, _ = build(tmp_path, protocol, rows, start=15000, end=90000)
    assert f.loc[15000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[60000, "cpu_cores__status"] == "UNOBSERVED"
    assert f.loc[75000, "cpu_cores"] == pytest.approx(0.2)


def test_cross_file_conflict_preserves_each_source_value(tmp_path, protocol):
    from src.service.metrics import build_metric_windows

    directory = tmp_path / CASE / "metric_data"
    directory.mkdir(parents=True)
    paths = [directory / "a.csv", directory / "b.csv"]
    pd.DataFrame([row(0, 10), row(15, 13)]).to_csv(paths[0], index=False)
    pd.DataFrame([row(15, 99)]).to_csv(paths[1], index=False)
    meta = dict(case_id=CASE, window_start_ms=0, window_end_ms=30000, is_anomaly_case=False)
    f, q = build_metric_windows(paths, meta, protocol)
    conflict = q[q.query_time_ms.eq(15000)]
    assert set(conflict.value) == {13.0, 99.0}
    assert set(conflict.source_path) == set(map(str, paths))
    assert set(conflict.source_row_number) == {1, 2}
    assert conflict.series_status.eq("CONFLICTING_SAMPLE").all()
    assert (
        f.loc[
            f.canonical_service_id.eq(SVC) & f.window_start_ms.eq(15000), "cpu_cores__status"
        ].iloc[0]
        == "CONFLICTING_SAMPLE"
    )


def test_stream_filters_metrics_entities_and_case_interval(tmp_path, protocol):
    from src.service.metrics import iter_metric_batches

    rows = [
        row(15, 13),
        row(130, 40),
        row(-31, 1),
        row(-30, 2),
        row(15, 100, "node_cpu_seconds_total"),
    ]
    for container in ["", "POD", "sidecar"]:
        r = row(15, 99)
        r["container"] = container
        rows.append(r)
    r = row(15, 99)
    r["namespace"] = "other"
    rows.append(r)
    path = tmp_path / CASE / "metrics.csv"
    path.parent.mkdir()
    pd.DataFrame(rows).to_csv(path, index=False)
    f = pd.concat(list(iter_metric_batches([path], protocol)))
    assert f.query_time_ms.tolist() == [15000, -30000]
    assert f.source_row_number.tolist() == [1, 4]


def test_diagnostic_conflict_excludes_otherwise_observed_r3(tmp_path, protocol):
    rows = [
        row(0, 10),
        row(15, 13),
        row(15, 100, "container_memory_usage_bytes"),
        row(15, 200, "container_spec_memory_limit_bytes"),
        row(15, 1, "container_processes"),
        row(15, 40, "container_memory_working_set_bytes"),
        row(15, 50, "container_memory_working_set_bytes"),
    ]
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[15000, "cpu_cores"] == pytest.approx(0.2)
    assert f.loc[15000, "memory_usage_ratio"] == 0.5
    assert f.loc[15000, "identity_ambiguity"]
    assert not f.loc[15000, "resource_valid"]
    assert "CONFLICTING_SAMPLE" in f.loc[15000, "quality_reasons"]


def test_empty_input_keeps_all_candidates(tmp_path, protocol):
    from src.service.metrics import build_metric_windows

    f, q = build_metric_windows(
        [],
        dict(case_id=CASE, window_start_ms=0, window_end_ms=30000, is_anomaly_case=False),
        protocol,
    )
    assert len(f) == 50
    assert len(set(f.canonical_service_id)) == 25
    assert f.resource_valid.eq(False).all()
    assert f.metric_receipt_time_ms.isna().all()
    assert q.empty


def test_ratio_overflow_never_produces_infinity(tmp_path, protocol):
    rows = [
        row(15, 1e308, "container_memory_usage_bytes"),
        row(15, 1e-308, "container_spec_memory_limit_bytes"),
    ]
    f, _ = build(tmp_path, protocol, rows)
    assert np.isnan(f.loc[15000, "memory_usage_ratio"])
    assert f.loc[15000, "memory_usage_ratio__status"] == "UNSUPPORTED"


def test_round_timestamp_before_interval_filter(tmp_path, protocol):
    from src.service.metrics import iter_metric_batches

    path = tmp_path / CASE / "metrics.csv"
    path.parent.mkdir()
    pd.DataFrame([row(119.9996, 13), row(-30.0004, 10)]).to_csv(path, index=False)
    f = pd.concat(list(iter_metric_batches([path], protocol)))
    assert f.query_time_ms.tolist() == [-30000]


def test_returning_identity_restarts_transition(tmp_path, protocol):
    # Removing episode restart would score 135s despite the switch at 120s.
    protocol.config["cases"][0]["window_end_ms"] = 180000
    rows = []
    for identity, points in [("old", [0, 15]), ("new", [60, 75]), ("old", [120, 135, 150])]:
        for t in points:
            for metric, value in [
                (CPU, t),
                ("container_memory_usage_bytes", 100),
                ("container_spec_memory_limit_bytes", 200),
                ("container_processes", 1),
            ]:
                r = row(t, value, metric)
                r["id"] = identity
                rows.append(r)
    f, _ = build(tmp_path, protocol, rows, end=180000)
    assert f.loc[120000, "cpu_cores__status"] == "TRANSITION"
    assert f.loc[135000, "cpu_cores__status"] == "TRANSITION"
    assert not f.loc[135000, "resource_valid"]
    assert f.loc[150000, "cpu_cores"] == 1.0
    assert f.loc[150000, "resource_valid"]
    assert f.loc[15000, "cpu_cores"] == 1.0  # Future episodes do not alter past output.


def test_counter_rate_overflow_is_unsupported(tmp_path, protocol):
    # Finite counter inputs with dt=.001s must not abort the full grid with inf.
    f, q = build(tmp_path, protocol, [row(0, 0), row(0.001, 1e308)])
    assert np.isnan(f.loc[0, "cpu_cores"])
    assert f.loc[0, "cpu_cores__status"] == "UNSUPPORTED"
    point = q[q.query_time_ms.eq(1)].iloc[0]
    assert point.series_status == "UNSUPPORTED"
    assert point.counter_dt_s == 0.001
    assert point.counter_delta == 1e308
    assert point.value == 1e308


def test_interleaved_old_snapshots_do_not_restart_new_cooldown(tmp_path, protocol):
    # Moving new's first observation from30 to60 wrongly excludes75s window.
    rows = []
    for identity, points in [("old", [0, 15, 45]), ("new", [30, 60, 75, 90])]:
        for t in points:
            for metric, value in [
                (CPU, t),
                ("container_memory_usage_bytes", 100),
                ("container_spec_memory_limit_bytes", 200),
                ("container_processes", 1),
            ]:
                r = row(t, value, metric)
                r["id"] = identity
                rows.append(r)
    f, _ = build(tmp_path, protocol, rows)
    assert f.loc[60000, "cpu_cores__status"] == "IDENTITY_AMBIGUOUS"
    assert not f.loc[75000, "identity_ambiguity"]
    assert f.loc[75000, "cpu_cores"] == 1.0
    assert f.loc[75000, "resource_valid"]
