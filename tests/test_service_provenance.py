"""Producer bindings use full content identity, including relocated frozen chains."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.parametrize("mutation", ["changed", "missing", "duplicate", "bad_sha", "wrong_origin"])
def test_upstream_binding_rejects_invalid_identity(tmp_path, mutation, frozen_service_chain):
    from src.service.protocol import validate_source_bindings

    upstream = tmp_path / "data"
    upstream.mkdir()
    original = frozen_service_chain / "data" / "manifest.json"
    (upstream / "manifest.json").write_bytes(original.read_bytes())
    source = json.loads((frozen_service_chain / "run" / "source_manifest.json").read_text())
    entry = next(x for x in source["sources"] if Path(x["path"]).name == "manifest.json")
    if mutation == "changed":
        manifest = json.loads(original.read_text())
        manifest["outputs"]["features.parquet"]["sha256"] = "0" * 64
        (upstream / "manifest.json").write_text(json.dumps(manifest))
    elif mutation == "missing":
        source["sources"].remove(entry)
    elif mutation == "duplicate":
        source["sources"].append(entry.copy())
    elif mutation == "bad_sha":
        entry["sha256"] = "bad"
    else:
        entry["path"] = "/unrelated/manifest.json"
    with pytest.raises(ValueError, match="binding"):
        validate_source_bindings(source, [upstream])


def test_upstream_binding_accepts_exact_copy(tmp_path, frozen_service_chain):
    from src.service.protocol import validate_source_bindings

    upstream = tmp_path / "relocated"
    upstream.mkdir()
    (upstream / "manifest.json").write_bytes(
        (frozen_service_chain / "data" / "manifest.json").read_bytes()
    )
    source = json.loads((frozen_service_chain / "run" / "source_manifest.json").read_text())
    validate_source_bindings(source, [upstream])


@pytest.mark.parametrize("stage", ["data", "run", "evaluation", "admission"])
def test_report_rejects_mixed_frozen_content_before_output(tmp_path, stage, frozen_service_chain):
    import shutil

    from src.service.evaluation import render_report

    for name in ["data", "run", "evaluation", "admission"]:
        shutil.copytree(frozen_service_chain / name, tmp_path / name)
    if stage in {"data", "run"}:
        # An optional admission analysis must not be needed to enforce stage bindings.
        shutil.rmtree(tmp_path / "admission")
    folder = tmp_path / stage
    if stage == "admission":
        path = folder / "analysis_manifest.json"
        analysis = json.loads(path.read_text())
        analysis["consumed_sha256"]["run/scores.parquet"] = "0" * 64
        path.write_text(json.dumps(analysis))
    else:
        # A legal self-contained version with the same config/IDs but distinct content.
        filename = {
            "data": "features.parquet",
            "run": "scores.parquet",
            "evaluation": "s1_proxy_metrics.json",
        }[stage]
        path = folder / filename
        if stage in {"data", "run"}:
            frame = pd.read_parquet(path)
            column = "cpu_cores" if stage == "data" else "anomaly_score"
            selected = frame.index[~frame.case_id.str.startswith("Normal") & frame[column].notna()][
                0
            ]
            frame.loc[selected, column] += 0.1
            frame.to_parquet(path, index=False)
        else:
            value = json.loads(path.read_text())
            value["provenance_test_version"] = 2
            path.write_text(json.dumps(value))
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["outputs"][filename] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
        )
        for name in ["manifest.json", "run_manifest.json", "evaluation_manifest.json"]:
            if (folder / name).exists():
                (folder / name).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        render_report(
            tmp_path / "data", tmp_path / "run", tmp_path / "evaluation", tmp_path / "report"
        )
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("mutation", ["different_data", "missing_binding"])
def test_evaluate_rejects_cross_version_before_output(tmp_path, mutation, frozen_service_chain):
    import shutil

    from scripts.service_experiment import evaluate

    for name in ["data", "run"]:
        shutil.copytree(frozen_service_chain / name, tmp_path / name)
    folder = tmp_path / ("data" if mutation == "different_data" else "run")
    filename = "features.parquet" if mutation == "different_data" else "source_manifest.json"
    path = folder / filename
    if mutation == "different_data":
        frame = pd.read_parquet(path)
        selected = frame.index[~frame.case_id.str.startswith("Normal") & frame.cpu_cores.notna()][0]
        frame.loc[selected, "cpu_cores"] += 0.1
        frame.to_parquet(path, index=False)
    else:
        value = json.loads(path.read_text())
        value["sources"] = [x for x in value["sources"] if Path(x["path"]).name != "manifest.json"]
        path.write_text(json.dumps(value))
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["outputs"][filename] = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
    )
    for name in ["manifest.json", "run_manifest.json"]:
        if (folder / name).exists():
            (folder / name).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding"):
        evaluate(
            Path("configs/service/minimal_v1.yaml"),
            frozen_service_chain / "dataset.yaml",
            tmp_path / "data",
            tmp_path / "run",
            tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("stage", ["evaluation", "admission"])
def test_report_rejects_missing_upstream_binding(tmp_path, stage, frozen_service_chain):
    import shutil

    from src.service.evaluation import render_report

    for name in ["data", "run", "evaluation", "admission"]:
        shutil.copytree(frozen_service_chain / name, tmp_path / name)
    folder = tmp_path / stage
    filename = "source_manifest.json" if stage == "evaluation" else "analysis_manifest.json"
    path = folder / filename
    source = json.loads(path.read_text())
    if stage == "admission":
        del source["consumed_sha256"]["run/scores.parquet"]
    else:
        source["sources"] = [
            x for x in source["sources"] if Path(x["path"]).name != "manifest.json"
        ]
    path.write_text(json.dumps(source))
    if stage == "evaluation":
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest["outputs"][filename] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size
        )
        for name in ["manifest.json", "evaluation_manifest.json"]:
            (folder / name).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding"):
        render_report(
            tmp_path / "data", tmp_path / "run", tmp_path / "evaluation", tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_report_accepts_relocated_frozen_chain_with_admission(tmp_path, frozen_service_chain):
    import shutil

    from src.service.evaluation import render_report

    for name in ["data", "run", "evaluation", "admission"]:
        shutil.copytree(frozen_service_chain / name, tmp_path / name)
    out = tmp_path / "report"
    render_report(tmp_path / "data", tmp_path / "run", tmp_path / "evaluation", out)
    assert (out / "_SUCCESS").is_file()
    assert not (out / "_FAILURE").exists()
    comparison = pd.read_csv(out / "event_comparison.csv")
    assert len(comparison) == 24
    assert comparison.n_candidates.eq(25).all()
    assert len(list(out.glob("event_*.png"))) == 3
    assert (out / "method_admission.md").read_bytes() == (
        frozen_service_chain / "admission/method_admission.md"
    ).read_bytes()
