import io
import json

import pandas as pd
import pytest

from src.service import modality_preparation as m


def grid():
    return m.full_grid("tt", "run", ["a", "b"], 0, 30000)


def test_complete_grid_and_blocked_sn():
    g = m.full_grid("sn", "r", ["b", "a"], 1, 45001)
    assert len(g) == 4
    assert g.window_start_ms.tolist() == [15000, 30000, 15000, 30000]
    tables = m.blocked_tables(g)
    for table in tables.values():
        assert table[m.KEYS].equals(g[m.KEYS])
        assert table.quality_status.str.startswith("BLOCKED_").all()
    assert tables["execution_windows"].observation_count.isna().all()


def test_entry_ends_in_window_and_unknown_error_is_not_false():
    spans = [
        dict(
            node_id="1",
            type="Entry",
            service_code="a",
            start_timestamp_ms=-1,
            end_timestamp_ms=15001,
            duration_ms=15002,
            is_error=None,
        ),
        dict(
            node_id="2",
            type="Local",
            service_code="a",
            start_timestamp_ms=1,
            end_timestamp_ms=2,
            duration_ms=1,
            is_error=True,
        ),
    ]
    rows = m.entry_observations([{"spans": spans}, {"spans": spans}], "run")
    table, diagnostics = m.execution_windows(grid(), rows)
    a = table[table.canonical_service_id.eq("a")]
    assert pd.isna(a.iloc[0].observation_count)
    assert a.iloc[1].observation_count == 1
    assert a.iloc[1].duration_mean_ms == 15002
    assert a.iloc[1].error_known_count == 0
    assert pd.isna(a.iloc[1].error_count)
    assert diagnostics["duplicates"] == 1


def test_span_identity_collision_fails():
    span = dict(
        node_id="x",
        type="Entry",
        service_code="a",
        start_timestamp_ms=0,
        end_timestamp_ms=1,
        duration_ms=1,
    )
    with pytest.raises(ValueError, match="conflicting"):
        m.execution_windows(
            grid(),
            m.entry_observations(
                [{"spans": [span]}, {"spans": [dict(span, duration_ms=2)]}], "run"
            ),
        )


def test_stream_top_array_validates_tail():
    assert list(m.iter_top_array(io.StringIO('{"meta":1,"traces":[{"spans":[]}]}'), "traces")) == [
        {"spans": []}
    ]
    with pytest.raises(ValueError):
        list(m.iter_top_array(io.StringIO('{"traces":[{}]} bad'), "traces"))


def test_log_repeated_occurrences_multiline_replay_and_instance_boundary():
    base = dict(
        case_id="run",
        run_id="batch",
        attempt_id="attempt",
        plan_item_id="item",
        source_timestamp="1970-01-01T00:00:01+00:00",
        partial=False,
        segment=0,
        replay=False,
        source_occurrence=1,
    )
    records = [
        dict(base, raw_line=base["source_timestamp"] + " ERROR failure"),
        dict(base, raw_line=base["source_timestamp"] + " \tat stack.Frame"),
        dict(base, raw_line=base["source_timestamp"] + " ERROR failure", source_occurrence=2),
        dict(base, raw_line=base["source_timestamp"] + " ERROR failure", replay=True),
    ]
    events, diag = m.log_events(records, "a", "pod/container", "file.jsonl", "run")
    assert len(events) == 2
    assert "\n\tat stack.Frame" in events[0]["message"]
    assert diag["replays"] == 1
    table = m.log_windows(grid(), events, {"a": "COLLECTION_PRESENT"})
    assert table.iloc[0].event_count == 2
    assert len(json.loads(table.iloc[0].messages_json)[0]["references"]) == 2
    assert table.iloc[2].quality_status == "COLLECTION_MISSING"
    assert pd.isna(table.iloc[2].event_count)


def test_resource_preserves_values_and_rejects_incomplete_grid():
    g = grid()
    source = g.drop(columns=["dataset_id", "run_id"]).assign(case_id="run", cpu_cores=3.5)
    result = m.resource_windows(g, source)
    assert result.cpu_cores.tolist() == [3.5] * 4
    with pytest.raises(ValueError, match="grid"):
        m.resource_windows(g, source.iloc[:-1])
    source["resource_valid"] = [True, False, True, False]
    source["quality_reasons"] = ["", "GAP", "", "MISSING"]
    quality = m.resource_windows(g, source)
    assert quality.quality_status.tolist() == ["RESOURCE_VALID", "RESOURCE_INVALID"] * 2
    assert quality.quality_reasons.tolist() == ["", "GAP", "", "MISSING"]


def test_path_mapping_cannot_leave_dataset(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        m.mapped_path(tmp_path, "/old/x", {"exact_files": {"/old/x": "/etc/passwd"}})


def test_bound_file_checks_copy_manifest_and_rejects_tamper(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text("original")
    records = {"trace.json": {"sha256": m.file_digest(p), "destination_sha256": m.file_digest(p)}}
    assert m.bound_file(tmp_path, "trace.json", records)["sha256"] == m.file_digest(p)
    p.write_text("tampered")
    with pytest.raises(ValueError, match="Hash"):
        m.bound_file(tmp_path, "trace.json", records)
    with pytest.raises(ValueError, match="manifest"):
        m.bound_file(tmp_path, "unlisted.json", records)


def test_same_log_message_keeps_real_occurrences_and_unknown_time():
    base = dict(
        case_id="run",
        source_timestamp="1970-01-01T00:00:01Z",
        raw_line="1970-01-01T00:00:01Z hello",
        replay=False,
        partial=False,
    )
    events, diag = m.log_events(
        [base, dict(base, source_occurrence=2), dict(base, source_timestamp="1970-01-01T00:00:02")],
        "a",
        "instance",
        "f",
        "run",
    )
    result = m.log_windows(grid(), events, {"a": "COLLECTION_PRESENT"})
    assert json.loads(result.iloc[0].messages_json)[0]["count"] == 2
    assert diag["invalid_source_time_or_raw"] == 1


def test_writer_sn_emits_blocked_complete_three_tables_without_training_fields(tmp_path):
    from src.service.window_preparation import write_window_tables

    inventory = {
        "tt": {"runs": [], "dataset_id": "tt"},
        "sn": {
            "runs": {
                "snrun": {
                    "metadata": {"start": 0.001, "end": 45.001},
                    "metrics": {"a": {}},
                    "logs": {},
                    "traces": {"services": {"b": 1}},
                }
            }
        },
    }
    summary = write_window_tables(inventory, tmp_path, tmp_path)
    assert summary["training_ready"] is False
    for name in ["resource_windows", "execution_windows", "log_windows"]:
        table = pd.read_parquet(tmp_path / "sn" / f"{name}.parquet")
        assert len(table) == 4
        assert not {"label", "split"} & set(table)
        schema = json.loads((tmp_path / "sn" / f"{name}.schema.json").read_text())
        assert schema["key"] == m.KEYS


def test_case_binding_rejects_quality_lineage_mismatch(tmp_path):
    from src.service import preparation

    case = dict(
        case_id="run",
        source_metadata="meta.json",
        service_quality="quality.json",
        plan_item_id="item",
        service_features_sha256="featurehash",
    )
    identity = dict(
        case_id="run", run_id="batch", plan_hash="plan", plan_item_id="item", attempt_id="attempt"
    )
    (tmp_path / "meta.json").write_text(json.dumps(identity))
    (tmp_path / "quality.json").write_text(json.dumps(dict(identity, attempt_id="WRONG")))
    p = tmp_path / "batch_control/research_service_inputs/run"
    p.mkdir(parents=True)
    (p / "source_manifest.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="lineage"):
        preparation.bind_tt_case(tmp_path, case, dict(run_id="batch", plan_hash="plan"), {})


def test_cached_sn_inventory_requires_binding(tmp_path):
    from src.service.preparation import cached_sn_inventory

    archive = tmp_path / "archive"
    archive.write_text("test")
    p = tmp_path / "inventory.json"
    p.write_text(json.dumps({"sn": {"archive_md5": m.file_digest(archive, "md5"), "runs": {}}}))
    with pytest.raises(ValueError, match="inventory hash"):
        cached_sn_inventory(p, "wrong", archive)
    with pytest.raises(ValueError, match="archive"):
        cached_sn_inventory(p, m.file_digest(p), archive)


def test_tt_writer_reads_one_bound_trace_and_only_jsonl(tmp_path):
    from src.service.window_preparation import write_window_tables

    root, output = tmp_path / "raw", tmp_path / "output"
    root.mkdir()
    output.mkdir()
    (root / "_handoff").mkdir()
    resource = grid().drop(columns=["dataset_id", "run_id"]).assign(case_id="run", cpu_cores=2.0)
    resource.to_parquet(root / "features.parquet")
    folder = root / "run/log_data"
    folder.mkdir(parents=True)
    (folder / "stream.log").write_text("MUST NOT COUNT")
    stamp = "1970-01-01T00:00:01Z"
    identity = dict(case_id="run", run_id="batch", attempt_id="attempt", plan_item_id="item")
    record = dict(
        identity,
        source_timestamp=stamp,
        raw_line=stamp + " hello",
        partial=False,
        segment=0,
        replay=False,
        source_occurrence=1,
    )
    (folder / "stream.jsonl").write_text(json.dumps(record) + "\n")
    stream = dict(
        identity,
        container_name="a",
        pod_uid="pod",
        container_id="container",
        segment=0,
        log_file="/old/stream.log",
        opened_ms=0,
        closed_ms=30000,
    )
    (folder / "stream_registry.json").write_text(json.dumps(dict(identity, streams=[stream])))
    trace_folder = root / "run/trace_data"
    trace_folder.mkdir()
    (trace_folder / "run_skywalking_traces_1.json").write_text(
        json.dumps(
            {
                "traces": [
                    {
                        "spans": [
                            dict(
                                node_id="id",
                                service_code="a",
                                type="Entry",
                                start_timestamp_ms=-2,
                                end_timestamp_ms=15000,
                                duration_ms=15002,
                                is_error=True,
                            )
                        ]
                    }
                ]
            }
        )
    )
    mapping = {"exact_files": {"/old/stream.log": str(folder / "stream.log")}}
    (root / "_handoff/path_map.json").write_text(json.dumps(mapping))
    files = list(root.rglob("*"))
    records = [
        dict(
            relative_path=str(p.relative_to(root)),
            sha256=m.file_digest(p),
            destination_sha256=m.file_digest(p),
        )
        for p in files
        if p.is_file()
    ]
    (root / "_handoff/dataset_copy_manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    inventory = {
        "tt": {
            "dataset_id": "tt",
            "runs": [
                dict(
                    run_id="run",
                    collection_identity=identity,
                    sources={
                        "service_features": {
                            "path": "features.parquet",
                            "sha256": m.file_digest(root / "features.parquet"),
                        }
                    },
                )
            ],
        },
        "sn": {"runs": {}},
    }
    write_window_tables(inventory, root, output)
    e = pd.read_parquet(output / "tt/execution_windows.parquet")
    assert e.iloc[1].observation_count == 1
    l = pd.read_parquet(output / "tt/log_windows.parquet")
    assert l.event_count.sum() == 1
    assert "MUST NOT COUNT" not in "".join(l.messages_json)
    with pytest.raises(FileExistsError):
        write_window_tables(inventory, root, output)


def test_cli_output_gate_rejects_outside_service_output(tmp_path):
    from scripts.prepare_service_data import validate_output

    with pytest.raises(ValueError, match="outputs/service"):
        validate_output(tmp_path / "other", tmp_path / "raw")


def test_collection_coverage_and_partial_lines_never_cross_segment():
    streams = [dict(container_name="a", opened_ms=1000, closed_ms=16000)]
    statuses = m.collection_window_status(grid(), streams)
    assert statuses == [
        "COLLECTION_PARTIAL_WINDOW",
        "COLLECTION_PARTIAL_WINDOW",
        "COLLECTION_MISSING",
        "COLLECTION_MISSING",
    ]
    stamp = "1970-01-01T00:00:01Z"
    record = dict(
        case_id="run",
        source_timestamp=stamp,
        raw_line=stamp + " first",
        partial=True,
        replay=False,
        segment=0,
    )
    events, _ = m.log_events(
        [record, dict(record, segment=1, raw_line=stamp + " second")],
        "a",
        "instance",
        "file",
        "run",
    )
    assert len(events) == 2


@pytest.mark.parametrize(
    "kind", ["missing_first", "missing_last", "expected_count", "metadata_end"]
)
def test_tt_bound_grid_rejects_tidy_truncation(kind):
    from src.service.window_preparation import validate_tt_bound_grid

    candidate = grid()
    run = dict(
        run_id="run",
        candidate_services=["a", "b"],
        expected_service_windows=4,
        start_ms=0,
        end_ms=30000,
    )
    if kind == "missing_first":
        candidate = candidate[candidate.window_start_ms.ne(0)]
    elif kind == "missing_last":
        candidate = candidate[candidate.window_start_ms.ne(15000)]
    elif kind == "expected_count":
        run["expected_service_windows"] = 5
    else:
        run["end_ms"] = 45000
    with pytest.raises(ValueError, match="bound grid"):
        validate_tt_bound_grid(candidate, run)


def test_tt_bound_grid_accepts_complete_source_grid():
    from src.service.window_preparation import validate_tt_bound_grid

    validate_tt_bound_grid(
        grid(),
        dict(
            run_id="run",
            candidate_services=["a", "b"],
            expected_service_windows=4,
            start_ms=0,
            end_ms=30000,
        ),
    )
