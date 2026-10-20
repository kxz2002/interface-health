import importlib
import json

import pandas as pd
import pytest

from src.service.modality_preparation import full_grid


def test_unknown_log_service_rejected_but_missing_stream_keeps_grid():
    sn = api()
    grid = full_grid("eadro_sn_v1", "run", ["svc", "missing"], 0, 30000)
    with pytest.raises(ValueError, match="Unknown SN log service"):
        sn.sn_log_windows(grid, {"typo": []}, "logs.json")
    table, _ = sn.sn_log_windows(grid, {"svc": []}, "logs.json")
    assert len(table) == 4
    assert (
        table[table.canonical_service_id.eq("missing")]
        .quality_status.eq("COLLECTION_MISSING")
        .all()
    )


@pytest.mark.parametrize(
    "processes,message",
    [
        ({"p": {"serviceName": "typo"}}, "Unknown SN span service"),
        ({}, "Missing SN span process"),
        ({"p": {}}, "Missing SN span service"),
    ],
)
def test_span_service_identity_rejected_separately_from_time(processes, message):
    sn = api()
    grid = full_grid("eadro_sn_v1", "run", ["svc"], 0, 30000)
    trace = {
        "traceID": "t",
        "processes": processes,
        "spans": [{"spanID": "s", "processID": "p", "startTime": 0, "duration": 1}],
    }
    with pytest.raises(ValueError, match=message):
        sn.span_windows(grid, [trace], "spans.json")


@pytest.mark.parametrize(
    "field,value",
    [
        ("resource_unit", "bytes"),
        ("resource_measurement_semantics", "counter"),
        ("resource_aggregation", "rate"),
        ("execution_object", "request"),
    ],
)
def test_profile_rejects_metadata_contradicting_exploratory_operations(field, value):
    sn = api()
    profile = json.loads(
        __import__("pathlib").Path("configs/service/sn_exploratory_v1.json").read_text()
    )
    with pytest.raises(ValueError, match="profile"):
        sn.validate_profile(dict(profile, **{field: value}))


def api():
    module = importlib.util.find_spec("src.service.sn_exploratory")
    assert module is not None, "Independent SN exploratory adapter is missing"
    return importlib.import_module("src.service.sn_exploratory")


def test_span_end_window_dedup_unknown_error_and_raw_reference():
    sn = api()
    grid = full_grid("eadro_sn_v1", "run", ["svc"], 0, 30000)
    span = {
        "spanID": "s",
        "processID": "p",
        "startTime": 28814900000,
        "duration": 100000,
        "tags": [],
    }
    trace = {"traceID": "t", "processes": {"p": {"serviceName": "svc"}}, "spans": [span, span]}
    table, diag = sn.span_windows(grid, [trace], "member")
    assert pd.isna(table.iloc[0].observation_count)
    assert table.iloc[1].observation_count == 1
    assert pd.isna(table.iloc[1].error_count)
    assert table.iloc[1].statistical_object == "Span"
    assert diag["duplicates"] == 1
    assert json.loads(table.iloc[1].observations_json)[0]["raw_startTime_us"] == 28814900000
    trace["spans"] = [span, dict(span, duration=200000)]
    with pytest.raises(ValueError, match="conflict"):
        sn.span_windows(grid, [trace], "member")


def test_log_all_records_timezone_continuations_frequency_and_diagnostics():
    sn = api()
    grid = full_grid("eadro_sn_v1", "run", ["svc"], 0, 30000)
    head = "[1970-Jan-01 08:00:00.000000] <error>: full message"
    messages = [
        head + "\n  stack",
        head,
        "  Caused by: reason",
        "[1970-Jan-01 08:00:15.000000] next",
        "[1970-Jan-01 08:00:01.000000] backward",
        "[invalid] bad",
        "[1970-Jan-01 08:00:30.000000] outside",
    ]
    table, diag = sn.sn_log_windows(grid, {"svc": messages}, "logs.json")
    assert table.iloc[0].event_count == 3
    assert table.iloc[1].event_count == 1
    assert diag["svc"]["backward_timestamps"] == 1
    assert diag["svc"]["invalid_timestamp"] == 1
    assert diag["svc"]["outside_grid"] == 1
    entries = json.loads(table.iloc[0].messages_json)
    assert entries[0]["message"] == "<error>: full message\n  stack"
    assert entries[1]["message"].endswith("Caused by: reason")
    repeated, _ = sn.sn_log_windows(grid, {"svc": [head, head]}, "logs.json")
    item = json.loads(repeated.iloc[0].messages_json)[0]
    assert item["count"] == 2 and len(item["references"]) == 2


def test_resource_raw_sequence_mathematical_mean_and_coverage():
    sn = api()
    grid = full_grid("eadro_sn_v1", "run", ["svc", "missing"], 0, 30000)
    rows = [
        {"timestamp": str(t), **{f: str(v) for f in sn.RESOURCE_FIELDS}}
        for t, v in [(0, 10), (1, 30), (15, 2)]
    ]
    table, diag = sn.sn_resource_windows(grid, "svc", rows, "metric.csv")
    row = table[table.canonical_service_id.eq("svc")].iloc[0]
    assert row.cpu_usage_total_mean == 20
    assert row.resource_sample_count == 2
    assert row.resource_coverage == pytest.approx(2 / 15)
    assert row.measurement_semantics == "unknown"
    assert json.loads(row.raw_samples_json)[1]["values"]["cpu_usage_total"] == 30
    assert diag["accepted"] == 3
    assert row.cpu_usage_total_valid_count == 2
    rows[1]["cpu_usage_total"] = "nan"
    partial, _ = sn.sn_resource_windows(grid, "svc", rows, "metric.csv")
    first = partial[partial.canonical_service_id.eq("svc")].iloc[0]
    assert first.cpu_usage_total_mean == 10 and first.cpu_usage_total_valid_count == 1
    assert first.quality_status == "OBSERVED_RAW_PARTIAL_NONFINITE_UNKNOWN_SEMANTICS"


def test_profile_requires_explicit_fixed_assumption():
    sn = api()
    profile = sn.load_profile(__import__("pathlib").Path("configs/service/sn_exploratory_v1.json"))
    assert profile["offset_seconds"] == -28800
    assert len(profile["runs"]) == 7
    assert len(profile["services"]) == 12
    with pytest.raises(ValueError):
        sn.validate_profile(dict(profile, offset_seconds=0))


def test_writer_exploratory_requires_archive_and_explicit_profile(tmp_path):
    from src.service.window_preparation import write_window_tables

    with pytest.raises(ValueError, match="explicit"):
        write_window_tables(
            {"tt": {"runs": []}, "sn": {"runs": {}}}, tmp_path, tmp_path, sn_profile="exploratory"
        )


def test_cached_inventory_defers_only_archive_digest_for_exploratory(tmp_path):
    from src.service.preparation import cached_sn_inventory, file_digest
    from src.service.sn_preparation import SN_MD5

    path = tmp_path / "inventory.json"
    path.write_text(json.dumps({"sn": {"archive_md5": SN_MD5, "runs": {}}}))
    result = cached_sn_inventory(
        path, file_digest(path), tmp_path / "not-opened.zip", defer_archive_digest=True
    )
    assert result["archive_md5"] == SN_MD5


def test_exact_containers_stream_all_runs_write_complete_tables(tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile
    import zipfile
    from pathlib import Path

    from src.service.window_preparation import write_window_tables

    sn = api()
    profile = json.loads(Path("configs/service/sn_exploratory_v1.json").read_text())
    archive = tmp_path / "fixture.zip"
    inventory = {"runs": {}}
    with zipfile.ZipFile(archive, "w") as outer:
        for container in sorted({v["container"] for v in profile["runs"].values()}):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w:gz") as inner:
                for rid, spec in profile["runs"].items():
                    if spec["container"] != container:
                        continue
                    inventory["runs"][rid] = dict(
                        container=container,
                        metadata={"start": 0, "end": 30},
                        metrics={s: {} for s in profile["services"]},
                    )
                    sources = {
                        "logs.json": json.dumps({s: [] for s in profile["services"]}),
                        "spans.json": "[]",
                    }
                    for service in profile["services"]:
                        sources["metrics/" + service + ".csv"] = (
                            ",".join(["timestamp"] + sn.RESOURCE_FIELDS)
                            + "\n"
                            + ",".join(["0"] + ["7"] * 7)
                            + "\n"
                        )
                    for name, value in sources.items():
                        payload = value.encode()
                        member = tarfile.TarInfo("data/" + rid + "/" + name)
                        member.size = len(payload)
                        inner.addfile(member, io.BytesIO(payload))
            outer.writestr(container, data.getvalue())
        outer.writestr("SN Dataset/data/do-not-read.txt", "duplicate outer packaging excluded")
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(sn, "SN_MD5", digest)
    inventory["archive_md5"] = profile["archive_md5"] = digest
    output = tmp_path / "out"
    output.mkdir()
    summary = write_window_tables(
        {"tt": {"runs": []}, "sn": inventory},
        tmp_path,
        output,
        sn_profile="exploratory",
        sn_archive=archive,
        sn_profile_config=profile,
    )
    assert summary["window_rows"] == 7 * 12 * 2
    for name in ["resource_windows", "execution_windows", "log_windows"]:
        table = pd.read_parquet(output / "sn" / f"{name}.parquet")
        assert len(table) == 7 * 12 * 2
    provenance = json.loads((output / "modality_sources.json").read_text())["sn"]
    assert all(len(r["sources"]) == 14 for r in provenance["runs"].values())
    schema = json.loads((output / "sn/resource_windows.schema.json").read_text())
    assert schema["measurement_semantics"] == "unknown"
    assert len(schema["feature_fields"]) == 7
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        list(sn.exploratory_runs(archive, inventory, profile))
