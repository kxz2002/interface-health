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
