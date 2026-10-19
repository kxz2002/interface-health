"""Synthetic inputs for the original four-case Service protocol; no research assets."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


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


def build_frozen_service_chain(root: Path) -> Path:
    from scripts.service_experiment import evaluate, prepare, score

    config, dataset, raw_root, _ = fixture_inputs(root)
    normal = next(c for c in yaml.safe_load(config.read_text())["cases"] if c["role"] == "normal")
    metric_path = raw_root / normal["case_id"] / "metric_data" / f"{normal['case_id']}_metrics.csv"
    seed_rows = pd.read_csv(metric_path)
    first = seed_rows[seed_rows.timestamp == seed_rows.timestamp.min()]
    expanded = []
    count = (normal["window_end_ms"] - normal["window_start_ms"]) // 15000
    for i in range(count):
        chunk = first.copy()
        chunk["timestamp"] = normal["window_start_ms"] / 1000 + 15 * i
        chunk.loc[chunk.metric_name == "container_cpu_usage_seconds_total", "value"] = i * 3
        expanded.append(chunk)
    pd.concat(expanded, ignore_index=True).to_csv(metric_path, index=False)
    prepare(config, dataset, root / "data", Path("configs/service/evidence_minimal_v1.json"))
    score(config, dataset, root / "data", root / "run")
    evaluate(config, dataset, root / "data", root / "run", root / "evaluation")

    admission = root / "admission"
    admission.mkdir()
    (admission / "failure_modes.csv").write_text("reason\nsynthetic fixture\n")
    (admission / "calibration_ties.csv").write_text("service,ties\nfixture,0\n")
    (admission / "method_admission.md").write_text("Synthetic fixture, not a research result.\n")
    consumed = {
        "data": [
            "events.parquet",
            "features.parquet",
            "index.parquet",
            "evidence.parquet",
            "quality_report.json",
        ],
        "run": ["scores.parquet"],
        "evaluation": [
            "event_rankings.parquet",
            "event_comparison.csv",
            "s1_proxy_metrics.json",
            "s1_independent_metrics.json",
        ],
    }
    analysis = {
        "consumed_sha256": {
            f"{stage}/{name}": hashlib.sha256((root / stage / name).read_bytes()).hexdigest()
            for stage, names in consumed.items()
            for name in names
        },
        "output_sha256": {
            name: hashlib.sha256((admission / name).read_bytes()).hexdigest()
            for name in ["failure_modes.csv", "calibration_ties.csv", "method_admission.md"]
        },
    }
    (admission / "analysis_manifest.json").write_text(json.dumps(analysis))
    return root
