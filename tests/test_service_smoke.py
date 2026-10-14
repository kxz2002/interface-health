"""Prepare preserves frozen candidates and writes isolated reviewable data."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.service.protocol import load_protocol


def fixture_inputs(tmp_path):
    config = Path("configs/service/minimal_v1.yaml")
    raw = yaml.safe_load(config.read_text())
    root = tmp_path / "raw"
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(dict(name="fixture", roots=[str(root)], normal_source=str(root)))
    )
    for case in raw["cases"]:
        folder = root / case["case_id"]
        (folder / "trace_data").mkdir(parents=True)
        meta = {k: v for k, v in case.items() if k != "role"}
        (folder / "trace_data" / "case_metadata.json").write_text(json.dumps(meta))
        (folder / "metric_data").mkdir()
        rows = []
        for service in ["ts-order-service", "ts-preserve-service"]:
            for i in range(4):
                for metric, value in [
                    ("container_cpu_usage_seconds_total", i * 3),
                    ("container_memory_usage_bytes", 50),
                    ("container_spec_memory_limit_bytes", 100),
                    ("container_processes", 3),
                ]:
                    rows.append(
                        dict(
                            metric_name=metric,
                            timestamp=case["window_start_ms"] / 1000 + i * 15,
                            value=value,
                            namespace="default",
                            pod=service,
                            container=service,
                            id="abc",
                            instance="host",
                            job="kubelet-cadvisor",
                            cpu="total" if "cpu" in metric else "",
                        )
                    )
        pd.DataFrame(rows).to_csv(
            folder / "metric_data" / f"{case['case_id']}_metrics.csv", index=False
        )
    sentinel = root / "sentinel.txt"
    sentinel.write_text("unchanged")
    return config, dataset, root, sentinel


def test_prepare_small_fixture(tmp_path, monkeypatch):
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    def no_fit(*args, **kwargs):
        raise AssertionError("prepare must not fit any detector")

    monkeypatch.setattr(OneClassSVM, "fit", no_fit)
    monkeypatch.setattr(IsolationForest, "fit", no_fit)
    from scripts.service_experiment import prepare

    config, dataset, root, sentinel = fixture_inputs(tmp_path)
    legacy = tmp_path / "legacy_endpoint_contract.txt"
    legacy.write_text("old labels")
    out = tmp_path / "result" / "data"
    prepare(config, dataset, out, Path("configs/service/evidence_minimal_v1.json"))
    features = pd.read_parquet(out / "features.parquet")
    index = pd.read_parquet(out / "index.parquet")
    assert not features.sample_id.duplicated().any()
    assert features.groupby(["case_id", "window_start_ms"]).size().eq(25).all()
    assert index.groupby(["case_id", "window_start_ms"]).partition.nunique().eq(1).all()
    assert index.paired_resource_eligible.sum() == features.resource_valid.sum() > 0
    assert (
        features.loc[
            ~features.canonical_service_id.isin(["ts-order-service", "ts-preserve-service"]),
            "resource_valid",
        ]
        .eq(False)
        .all()
    )
    assert sentinel.read_text() == "unchanged"
    assert legacy.read_text() == "old labels"
    assert (out / "_SUCCESS").is_file()
    assert (out / ".gitignore").read_text() == "/*.parquet\n/*.joblib\n"
    evidence = pd.read_parquet(out / "evidence.parquet")
    assert str(evidence.independent_label.dtype) == "Int8"
    assert str(evidence.proxy_label.dtype) == "Int8"
    assert not (out / "models.joblib").exists()
    assert index.loc[index.paired_resource_eligible, "exclusion_reasons"].map(len).eq(0).all()
    assert json.loads((out / "quality_report.json").read_text())["cases"][0]["partial_windows"]


def test_prepare_refuses_existing_output_and_bad_case(tmp_path):
    from scripts.service_experiment import prepare

    config, dataset, root, _ = fixture_inputs(tmp_path)
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="exists"):
        prepare(config, dataset, out)
    with pytest.raises(ValueError, match="protected|受保护"):
        prepare(config, dataset, root / "new")
    meta_path = next(root.glob("*/trace_data/case_metadata.json"))
    meta = json.loads(meta_path.read_text())
    meta["case_id"] = "unconfigured"
    meta_path.write_text(json.dumps(meta))
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="case_id"):
        prepare(config, dataset, failed)
    assert not (failed / "_SUCCESS").exists()
    assert (failed / "_FAILURE").is_file()
    meta["case_id"] = meta_path.parents[1].name
    meta_path.write_text(json.dumps(meta))
    next(root.glob("*/metric_data/*.csv")).unlink()
    with pytest.raises(FileNotFoundError):
        prepare(config, dataset, tmp_path / "missing")


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_service", "ts-order-service"),
        ("chaos_kind", "PodChaos"),
        ("inject_start_ms", 1),
        ("is_anomaly_case", False),
    ],
)
def test_prepare_rejects_metadata_drift(tmp_path, field, value):
    from scripts.service_experiment import prepare

    config, dataset, root, _ = fixture_inputs(tmp_path)
    path = root / "Lv_P_CPU_preserve_20260728T164711Z_em" / "trace_data/case_metadata.json"
    meta = json.loads(path.read_text())
    meta[field] = value
    path.write_text(json.dumps(meta))
    out = tmp_path / "drift"
    with pytest.raises(ValueError, match=field):
        prepare(config, dataset, out)
    assert not (out / "features.parquet").exists()
    assert (out / "_FAILURE").exists()


def test_prepare_manifest_output_hashes_and_small_sources(tmp_path):
    import hashlib

    from scripts.service_experiment import prepare
    from src.service.protocol import write_manifest

    config, dataset, _, _ = fixture_inputs(tmp_path)
    out = tmp_path / "hashed"
    prepare(config, dataset, out)
    manifest = json.loads((out / "manifest.json").read_text())
    expected = {
        "features.parquet",
        "instance_quality.parquet",
        "index.parquet",
        "events.parquet",
        "evidence.parquet",
        "resolved_config.yaml",
        "schema.json",
        "quality_report.json",
        "source_manifest.json",
        "coverage.csv",
        ".gitignore",
    }
    assert set(manifest["outputs"]) == expected
    assert manifest["excluded_outputs"] == {
        "manifest.json": "self-reference excluded",
        "_SUCCESS": "completion marker written after manifest",
        "_FAILURE": "failure marker is not an immutable data artifact",
    }
    for name, identity in manifest["outputs"].items():
        assert identity["sha256"] == hashlib.sha256((out / name).read_bytes()).hexdigest()
        assert identity["size_bytes"] == (out / name).stat().st_size
    first = (out / "manifest.json").read_bytes()
    protocol = load_protocol(config, dataset)
    write_manifest(out, [Path(p) for p in manifest["inputs"]], protocol)
    assert (out / "manifest.json").read_bytes() == first
    parquet = out / "features.parquet"
    original = parquet.read_bytes()
    parquet.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert parquet.stat().st_size == manifest["outputs"]["features.parquet"]["size_bytes"]
    assert (
        hashlib.sha256(parquet.read_bytes()).hexdigest()
        != manifest["outputs"]["features.parquet"]["sha256"]
    )
    sources = json.loads((out / "source_manifest.json").read_text())["sources"]
    small_evidence = [s for s in sources if s["path"].endswith((".yaml", ".json"))]
    assert small_evidence
    assert all("sha256" in s for s in small_evidence)
    assert all("sha256" not in s for s in sources if s["path"].endswith(".csv"))


def test_prepare_detects_same_stat_small_source_mutation(tmp_path, monkeypatch):
    import os

    import scripts.service_experiment as experiment

    config, dataset, root, _ = fixture_inputs(tmp_path)
    meta_path = root / "Normal_20260728T161044Z" / "trace_data/case_metadata.json"
    original_build = experiment.build_metric_windows
    modified = False

    def mutate_after_snapshot(*args):
        nonlocal modified
        if not modified:
            snapshot = meta_path.stat()
            text = meta_path.read_text()
            assert '"normal"' in text
            meta_path.write_text(text.replace('"normal"', '"NORMAL"'))
            os.utime(meta_path, ns=(snapshot.st_atime_ns, snapshot.st_mtime_ns))
            modified = True
        return original_build(*args)

    # Add unrelated same-length metadata text; core validation still matches config.
    meta = json.loads(meta_path.read_text())
    meta["note"] = "normal"
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(experiment, "build_metric_windows", mutate_after_snapshot)
    out = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="input changed"):
        experiment.prepare(config, dataset, out)
    assert (out / "_FAILURE").exists()
    assert not (out / "_SUCCESS").exists()


def test_report_is_score_readonly(tmp_path, monkeypatch):
    import hashlib

    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM

    import scripts.service_experiment as experiment
    from src.service.evaluation import render_report

    config, dataset, _, _ = fixture_inputs(tmp_path)
    data, run, evaluation = [tmp_path / name for name in ["data", "run", "evaluation"]]
    # Extend only the synthetic Normal fixture to exercise actual toy model fitting.
    raw_config = yaml.safe_load(config.read_text())
    normal = next(c for c in raw_config["cases"] if c["role"] == "normal")
    root = Path(yaml.safe_load(dataset.read_text())["roots"][0])
    metric_path = root / normal["case_id"] / "metric_data" / f"{normal['case_id']}_metrics.csv"
    seed_rows = pd.read_csv(metric_path)
    first = seed_rows[seed_rows.timestamp == seed_rows.timestamp.min()]
    expanded = []
    count = int((normal["window_end_ms"] - normal["window_start_ms"]) // 15000)
    for i in range(count):
        chunk = first.copy()
        chunk["timestamp"] = normal["window_start_ms"] / 1000 + 15 * i
        chunk.loc[chunk.metric_name == "container_cpu_usage_seconds_total", "value"] = i * 3
        expanded.append(chunk)
    pd.concat(expanded, ignore_index=True).to_csv(metric_path, index=False)
    experiment.prepare(config, dataset, data)
    experiment.score(config, dataset, data, run)
    experiment.evaluate(config, dataset, data, run, evaluation)
    import joblib

    assert joblib.load(run / "models.joblib")
    saved_scores = pd.read_parquet(run / "scores.parquet")
    assert saved_scores.score_status.eq("Unknown").any()
    assert saved_scores[["feature_group", "method"]].drop_duplicates().shape[0] == 8
    snapshots = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in [data, run, evaluation]
        for p in folder.iterdir()
        if p.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("report must not fit, prepare, score or parse raw")

    for name in ["prepare", "score", "build_metric_windows"]:
        monkeypatch.setattr(experiment, name, forbidden)
    monkeypatch.setattr(IsolationForest, "fit", forbidden)
    monkeypatch.setattr(OneClassSVM, "fit", forbidden)
    out = tmp_path / "report"
    render_report(data, run, evaluation, out)
    assert (out / "_SUCCESS").is_file()
    assert len(pd.read_csv(out / "event_comparison.csv")) == 24
    assert pd.read_csv(out / "event_comparison.csv").n_candidates.eq(25).all()
    assert pd.read_csv(out / "service_coverage.csv").canonical_service_id.nunique() == 25
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in snapshots.items()
    )
    with pytest.raises(ValueError, match="exists"):
        render_report(data, run, evaluation, out)
    target = data / "features.parquet"
    target.write_bytes(target.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        render_report(data, run, evaluation, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
