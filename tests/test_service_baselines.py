from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.service.baselines import (
    calibrate_score,
    fit_baselines,
    load_bundle,
    save_bundle,
    score_baselines,
)
from src.service.protocol import load_protocol, make_grid, make_indices


@pytest.fixture
def inputs():
    protocol = load_protocol(
        Path("configs/service/minimal_v1.yaml"), Path("configs/data/new_merge.yaml")
    )
    frames = [make_grid(c, protocol.config["services"]) for c in protocol.config["cases"]]
    features = pd.concat(frames, ignore_index=True)
    n = len(features)
    features["cpu_cores"] = np.arange(n) % 11 / 100
    features["memory_usage_ratio"] = 0.2
    features["process_count"] = 5.0
    features["score_time_lower_bound_ms"] = features.window_end_ms
    features["resource_valid"] = features.canonical_service_id.eq(protocol.config["services"][0])
    features["quality_reasons"] = [[] for _ in range(n)]
    index = make_indices(features, protocol)
    index["paired_resource_eligible"] = index.sample_id.map(
        features.set_index("sample_id").resource_valid
    )
    return features, index, protocol


def test_fit_never_sees_fault_or_tail(inputs):
    f, i, p = inputs
    b = fit_baselines(f, i, p)
    changed = f.copy()
    excluded = i.loc[i.partition.isin(["normal_tail", "fault_evaluation"]), "sample_id"]
    changed.loc[
        changed.sample_id.isin(excluded), ["cpu_cores", "memory_usage_ratio", "process_count"]
    ] = 999
    changed_index = i.copy()
    changed_index["proxy_exposure_label"] = 1
    other = fit_baselines(changed, changed_index, p)
    for key, ref in b.references.items():
        alt = other.references[key]
        np.testing.assert_array_equal(ref["mean"], alt["mean"])
        np.testing.assert_array_equal(ref["std"], alt["std"])
        for method in ref["calibration"]:
            np.testing.assert_array_equal(ref["calibration"][method], alt["calibration"][method])
        assert set(ref["fit_ids"]) <= set(i.loc[i.partition.eq("normal_fit"), "sample_id"])


def test_shared_resource_ids_and_order(inputs):
    f, i, p = inputs
    sample = f.sample_id.iloc[0]
    f.loc[f.sample_id.eq(sample), "resource_valid"] = False
    f.loc[f.sample_id.eq(sample), "memory_usage_ratio"] = np.nan
    i.loc[i.sample_id.eq(sample), "paired_resource_eligible"] = False
    b = fit_baselines(f, i.sample(frac=1, random_state=4), p)
    scores = score_baselines(f.sample(frac=1, random_state=2), i, b, p)
    assert scores.groupby("sample_id").size().eq(8).all()
    assert scores.loc[scores.sample_id.eq(sample), "score_status"].eq("Unknown").all()
    for service in p.config["services"]:
        assert (
            b.references[(service, "R3")]["fit_ids"]
            == b.references[(service, "CPU-only")]["fit_ids"]
        )
        assert (
            b.references[(service, "R3")]["calibration_ids"]
            == b.references[(service, "CPU-only")]["calibration_ids"]
        )
    cpu = scores[scores.feature_group.eq("CPU-only")].pivot(
        index="sample_id", columns="method", values="anomaly_score"
    )
    np.testing.assert_allclose(cpu.L2, cpu["max"], equal_nan=True)


def test_scores_and_constant_columns(inputs):
    f, i, p = inputs
    b = fit_baselines(f, i, p)
    ref = b.references[(p.config["services"][0], "R3")]
    assert ref["columns"] == ["cpu_cores", "memory_usage_ratio", "process_count"]
    np.testing.assert_array_equal(ref["std"][1:], [1.0, 1.0])
    scores = score_baselines(f, i, b, p)
    chosen = f[f.canonical_service_id.eq(p.config["services"][0])]
    x = (chosen[ref["columns"]].to_numpy() - ref["mean"]) / ref["std"]
    actual = scores[
        (scores.canonical_service_id.eq(p.config["services"][0])) & scores.feature_group.eq("R3")
    ]
    for method, expected in [
        ("OCSVM", -ref["models"]["OCSVM"].decision_function(x)),
        ("IF", -ref["models"]["IF"].score_samples(x)),
    ]:
        got = (
            actual[actual.method.eq(method)]
            .set_index("sample_id")
            .loc[chosen.sample_id]
            .anomaly_score
        )
        np.testing.assert_allclose(got, expected)


def test_calibration_and_untrained(inputs):
    np.testing.assert_allclose(
        calibrate_score(np.array([4.0, 3.0]), np.array([1.0, 2.0, 3.0])), [0.75, 0.5]
    )
    np.testing.assert_array_equal(calibrate_score(np.ones(3), np.ones(10)), np.zeros(3))
    f, i, p = inputs
    svc = p.config["services"][0]
    invalid = i.canonical_service_id.eq(svc) & i.partition.eq("normal_calibration")
    ids = i.loc[invalid, "sample_id"]
    f.loc[f.sample_id.isin(ids), "resource_valid"] = False
    i.loc[invalid, "paired_resource_eligible"] = False
    b = fit_baselines(f, i, p)
    assert b.references[(svc, "R3")]["status"] == "Untrained"
    s = score_baselines(f, i, b, p)
    assert s.loc[s.canonical_service_id.eq(svc), "anomaly_score"].isna().all()


def test_bundle_roundtrip(inputs, tmp_path):
    f, i, p = inputs
    b = fit_baselines(f, i, p)
    out = tmp_path / "bundle"
    save_bundle(b, out)
    other = load_bundle(out)
    a = score_baselines(f, i, b, p)
    c = score_baselines(f, i, other, p)
    np.testing.assert_allclose(a.anomaly_score, c.anomaly_score, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(a.calibrated_score, c.calibrated_score, rtol=1e-10, atol=1e-12)
    pd.testing.assert_frame_equal(
        a.drop(columns=["anomaly_score", "calibrated_score"]),
        c.drop(columns=["anomaly_score", "calibrated_score"]),
    )
    with pytest.raises(ValueError, match="exists"):
        save_bundle(b, out)


@pytest.mark.parametrize("mutation", ["partition", "eligibility", "identity", "duplicate"])
def test_index_guard(inputs, mutation):
    f, i, p = inputs
    if mutation == "partition":
        i.loc[i.case_id.ne(p.config["cases"][0]["case_id"]), "partition"] = "normal_fit"
    elif mutation == "eligibility":
        i.loc[0, "paired_resource_eligible"] = False
    elif mutation == "identity":
        i.loc[0, "canonical_service_id"] = p.config["services"][1]
    else:
        i = pd.concat([i, i.iloc[:1]])
    with pytest.raises(ValueError):
        fit_baselines(f, i, p)


def _prepared(inputs, tmp_path):
    import hashlib
    import json

    f, i, p = inputs
    data = tmp_path / "data"
    data.mkdir()
    f.to_parquet(data / "features.parquet", index=False)
    i.to_parquet(data / "index.parquet", index=False)
    from scripts.service_experiment import PREPARE_REQUIRED_ARTIFACTS

    for name in PREPARE_REQUIRED_ARTIFACTS - {"features.parquet", "index.parquet"}:
        (data / name).write_text("fixture-only provenance; no raw data\n")
    outputs = {
        name: {
            "sha256": hashlib.sha256((data / name).read_bytes()).hexdigest(),
            "size_bytes": (data / name).stat().st_size,
        }
        for name in PREPARE_REQUIRED_ARTIFACTS
    }
    (data / "manifest.json").write_text(json.dumps({"config": p.config, "outputs": outputs}))
    (data / "_SUCCESS").touch()
    return data


def test_score_tampered_input_fails_before_fit(inputs, tmp_path, monkeypatch):
    import src.service.baselines as baselines
    from scripts.service_experiment import score

    data = _prepared(inputs, tmp_path)
    with (data / "features.parquet").open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        baselines, "fit_baselines", lambda *args: pytest.fail("fit reached tampered input")
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        score(
            Path("configs/service/minimal_v1.yaml"),
            Path("configs/data/new_merge.yaml"),
            data,
            tmp_path / "run",
        )
    assert not (tmp_path / "run").exists()


def test_score_nooverwrite_and_failure_marker(inputs, tmp_path, monkeypatch):
    import src.service.baselines as baselines
    from scripts.service_experiment import score

    data = _prepared(inputs, tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()

    def broken(*args):
        raise RuntimeError("bounded simulated fit failure")

    monkeypatch.setattr(baselines, "fit_baselines", broken)
    with pytest.raises(ValueError, match="exists"):
        score(
            Path("configs/service/minimal_v1.yaml"),
            Path("configs/data/new_merge.yaml"),
            data,
            existing,
        )
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="simulated"):
        score(
            Path("configs/service/minimal_v1.yaml"), Path("configs/data/new_merge.yaml"), data, out
        )
    assert (out / "_FAILURE").is_file()
    assert not (out / "_SUCCESS").exists()


@pytest.mark.parametrize(
    "fit_count,cal_count,expected",
    [(29, 10, "Untrained"), (30, 9, "Untrained"), (30, 10, "Trained")],
)
def test_exact_reference_thresholds(inputs, fit_count, cal_count, expected):
    f, i, p = inputs
    svc = p.config["services"][0]
    keep = []
    for partition, count in [("normal_fit", fit_count), ("normal_calibration", cal_count)]:
        keep += (
            i.loc[i.canonical_service_id.eq(svc) & i.partition.eq(partition), "sample_id"]
            .iloc[:count]
            .tolist()
        )
    f["resource_valid"] = f.sample_id.isin(keep)
    i["paired_resource_eligible"] = i.sample_id.isin(keep)
    bundle = fit_baselines(f, i, p)
    assert bundle.references[(svc, "R3")]["status"] == expected
    assert bundle.references[(svc, "CPU-only")]["status"] == expected


@pytest.mark.parametrize("mutation", ["missing_features", "missing_index", "empty_outputs"])
def test_score_manifest_completeness_before_reads_or_fit(inputs, tmp_path, monkeypatch, mutation):
    import json

    import src.service.baselines as baselines
    from scripts.service_experiment import score

    data = _prepared(inputs, tmp_path)
    manifest_path = data / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "empty_outputs":
        manifest["outputs"] = {}
    else:
        del manifest["outputs"][
            "features.parquet" if mutation == "missing_features" else "index.parquet"
        ]
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        pd, "read_parquet", lambda *args, **kwargs: pytest.fail("read reached incomplete manifest")
    )
    monkeypatch.setattr(
        baselines, "fit_baselines", lambda *args: pytest.fail("fit reached incomplete manifest")
    )
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="manifest"):
        score(
            Path("configs/service/minimal_v1.yaml"), Path("configs/data/new_merge.yaml"), data, out
        )
    assert not out.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_config",
        "outputs_list",
        "root_list",
        "entry_list",
        "bad_hash",
        "bad_size",
        "unsafe_parent",
        "unsafe_absolute",
        "empty_name",
        "duplicate_entry",
        "duplicate_root",
        "missing_resolved_config",
        "symlink",
    ],
)
def test_score_rejects_malformed_manifest_before_reads(inputs, tmp_path, monkeypatch, mutation):
    import json

    import src.service.baselines as baselines
    from scripts.service_experiment import score

    data = _prepared(inputs, tmp_path)
    path = data / "manifest.json"
    manifest = json.loads(path.read_text())
    outputs = manifest["outputs"]
    if mutation == "missing_config":
        del manifest["config"]
    elif mutation == "outputs_list":
        manifest["outputs"] = []
    elif mutation == "root_list":
        manifest = []
    elif mutation == "entry_list":
        outputs["features.parquet"] = []
    elif mutation == "bad_hash":
        outputs["features.parquet"]["sha256"] = "invalid"
    elif mutation == "bad_size":
        outputs["features.parquet"]["size_bytes"] = True
    elif mutation in {"unsafe_parent", "unsafe_absolute", "empty_name"}:
        name = {"unsafe_parent": "../escape", "unsafe_absolute": "/tmp/escape", "empty_name": ""}[
            mutation
        ]
        outputs[name] = outputs["features.parquet"]
    elif mutation == "missing_resolved_config":
        del outputs["resolved_config.yaml"]
    elif mutation == "symlink":
        original = data / "features.parquet"
        external = tmp_path / "external.parquet"
        original.rename(external)
        original.symlink_to(external)
    if mutation == "duplicate_entry":
        text = json.dumps(manifest).replace('"features.parquet":', '"index.parquet":')
    elif mutation == "duplicate_root":
        text = json.dumps(manifest)[:-1] + ', "outputs": {}}'
    else:
        text = json.dumps(manifest)
    path.write_text(text)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *args, **kwargs: pytest.fail("read reached malformed manifest")
    )
    monkeypatch.setattr(
        baselines, "fit_baselines", lambda *args: pytest.fail("fit reached malformed manifest")
    )
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="manifest"):
        score(
            Path("configs/service/minimal_v1.yaml"), Path("configs/data/new_merge.yaml"), data, out
        )
    assert not out.exists()
