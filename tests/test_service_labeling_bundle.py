"""Synthetic replay and publication tests; never label production data."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.service.labeling import bundle
from src.service.labeling.evidence import EvidenceIndex


@pytest.fixture
def setup(tmp_path, monkeypatch):
    protocol = json.loads(Path("configs/service/tt_labels_a_v1.json").read_text())
    cid = next(iter(protocol["normal_splits"]))
    grid = pd.DataFrame(
        [
            dict(
                dataset_id="tt",
                run_id=cid,
                case_id=cid,
                canonical_service_id="svc",
                window_start_ms=t,
                window_end_ms=t + 15000,
                sample_id=f"s{t}",
                resource_valid=True,
            )
            for t in (0, 15000, 30000)
        ]
    )
    case = dict(
        is_fault=False,
        admission={"admitted": True},
        boundary=None,
        transport_exceptions=[],
        evidence_ids=["source"],
        candidate_services=["svc"],
    )
    index = EvidenceIndex(
        grid,
        {cid: case},
        {"source": dict(path="/machine/a", sha256="abc", source_locator="$")},
        protocol,
        "prepared",
    )
    monkeypatch.setattr(bundle, "load_evidence", lambda *args: index)
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return tmp_path, path, index, protocol


def build(setup, name):
    root, path, _, _ = setup
    return bundle.build_bundle(root / "raw", root / "prepared", path, root / name)


def test_replay_semantic_hash(setup):
    a = build(setup, "a")
    setup[2].grid.sort_values("sample_id", ascending=False, inplace=True)
    setup[2].sources["source"]["path"] = "/other/machine"
    b = build(setup, "b")
    assert a["label_version"] == b["label_version"]
    assert a["split_version"] == b["split_version"]
    assert len(list((setup[0] / "a/labels").iterdir())) == 7
    labels = pd.read_parquet(setup[0] / "a/labels/labels.parquet")
    assert str(labels.label.dtype) == "Int8"
    assert list(labels.evidence_ids.iloc[0]) == ["source"]


def test_v2_read_persists_and_checks_evaluation_membership(setup):
    setup[3]["evaluation_version"] = "evaluation-v2"
    setup[1].write_text(json.dumps(setup[3]))
    build(setup, "v2")
    data = bundle.read_bundle(setup[0] / "v2")
    assert "evaluation" in data and "evaluation_coverage" in data
    assert not data["evaluation"].evaluation_eligible.any()
    path = setup[0] / "v2/labels/evaluation.parquet"
    evaluation = pd.read_parquet(path)
    evaluation.loc[0, "evaluation_eligible"] = True
    evaluation.to_parquet(path, index=False)
    manifest_path = setup[0] / "v2/labels/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["evaluation.parquet"]["sha256"] = bundle.file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Evaluation membership"):
        bundle.read_bundle(setup[0] / "v2")


def test_no_overwrite_or_partial_publish(setup, monkeypatch):
    build(setup, "a")
    with pytest.raises(FileExistsError):
        build(setup, "a")
    with pytest.raises(ValueError):
        build(setup, "raw/new")

    def fail(*args, **kwargs):
        raise OSError("disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError):
        build(setup, "failed")
    assert not (setup[0] / "failed").exists()


def test_version_bindings(setup):
    a = build(setup, "a")
    setup[2].sources["source"]["sha256"] = "changed"
    b = build(setup, "b")
    assert a["label_version"] != b["label_version"]
    assert a["split_version"] == b["split_version"]
    setup[3]["normal_splits"][next(iter(setup[3]["normal_splits"]))] = "test"
    setup[1].write_text(json.dumps(setup[3]))
    c = build(setup, "c")
    assert b["split_version"] != c["split_version"]


def test_diff_stable_keys(setup):
    build(setup, "a")
    build(setup, "b")
    result = bundle.diff_bundles(setup[0] / "a", setup[0] / "b", setup[0] / "same")
    assert result["changed_rows"] == 0
    setup[2].cases[next(iter(setup[2].cases))]["admission"]["admitted"] = False
    build(setup, "c")
    result = bundle.diff_bundles(setup[0] / "a", setup[0] / "c", setup[0] / "diff")
    assert result["changed_rows"] == 3
    changes = pd.read_parquet(setup[0] / "diff/changes.parquet")
    assert changes.new_label.isna().all()
    with pytest.raises(FileExistsError):
        bundle.diff_bundles(setup[0] / "a", setup[0] / "c", setup[0] / "diff")
    (setup[0] / "c/labels/coverage.json").write_text("{}")
    with pytest.raises(ValueError):
        bundle.diff_bundles(setup[0] / "a", setup[0] / "c", setup[0] / "bad")


def test_diff_added_deleted_and_reason_only(setup):
    build(setup, "a")
    grid = setup[2].grid
    # One deletion, one addition, with unchanged rows retained by their stable keys.
    grid.drop(index=0, inplace=True)
    grid.loc[3] = dict(
        grid.iloc[-1], window_start_ms=45000, window_end_ms=60000, sample_id="s45000"
    )
    build(setup, "b")
    summary = bundle.diff_bundles(setup[0] / "a", setup[0] / "b", setup[0] / "delta")
    assert summary["change_types"] == {"deleted": 1, "added": 1}
    changes = pd.read_parquet(setup[0] / "delta/changes.parquet")
    assert set(changes.window_start_ms) == {0, 45000}
    case = setup[2].cases[next(iter(setup[2].cases))]
    case["admission"]["admitted"] = False
    build(setup, "c")
    case["admission"]["admitted"] = True
    case["transport_exceptions"] = [dict(start_ms=0, end_ms=60000, evidence_id="source")]
    build(setup, "d")
    summary = bundle.diff_bundles(setup[0] / "c", setup[0] / "d", setup[0] / "reasons")
    assert summary["changed_rows"] == 3
    assert summary["reason_transitions"] == {"run_admission_failed -> negative_exception": 3}
    package = bundle.read_bundle(setup[0] / "d")
    assert package["exceptions"][next(iter(setup[2].cases))]["merged_intervals"][0][
        "evidence_ids"
    ] == ["source"]


def test_rule_identity_binding_and_diff_protected_output(setup, monkeypatch):
    a = build(setup, "a")
    original = bundle.file_digest
    monkeypatch.setattr(
        bundle,
        "file_digest",
        lambda path: "changed-rule" if Path(path).name == "rules.py" else original(path),
    )
    b = build(setup, "b")
    assert a["label_version"] != b["label_version"]
    assert a["split_version"] == b["split_version"]
    with pytest.raises(ValueError, match="protected"):
        bundle.diff_bundles(setup[0] / "a", setup[0] / "b", setup[0] / "prepared/diff")


def test_preparation_tt_alias_is_protected(setup):
    root, path, _, _ = setup
    with pytest.raises(ValueError, match="protected"):
        bundle.build_bundle(root / "raw", root / "prepared/tt", path, root / "prepared/new")


def test_diff_zero_to_one(setup):
    build(setup, "normal")
    cid = next(iter(setup[2].cases))
    case = setup[2].cases[cid]
    case.update(
        is_fault=True,
        boundary={"first_intervention_requested_ms": 0, "reasons": []},
        fault={
            "effective_ms": 0,
            "removed_ms": 45000,
            "document": {
                "metadata": {"labels": {"target_service": "svc"}},
                "spec": {"action": "stress"},
            },
        },
    )
    setup[3]["normal_splits"].pop(cid)
    setup[1].write_text(json.dumps(setup[3]))
    build(setup, "fault")
    summary = bundle.diff_bundles(setup[0] / "normal", setup[0] / "fault", setup[0] / "positive")
    assert summary["changed_rows"] == 3
    changes = pd.read_parquet(setup[0] / "positive/changes.parquet")
    assert changes.old_label.eq(0).all()
    assert changes.new_label.eq(1).all()


def test_derive_evaluation_bundle_versions_scope_without_relabeling(setup, tmp_path):
    from src.service.labeling.bundle import derive_evaluation_bundle, read_bundle

    build(setup, "source")
    source = setup[0] / "source/labels"
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["evaluation_version"] = "evaluation-v3"
    protocol["evaluation_excluded_cases"] = [next(iter(protocol["normal_splits"]))]
    protocol_path = tmp_path / "evaluation-v3.json"
    protocol_path.write_text(json.dumps(protocol))
    preparation = tmp_path / "preparation/tt"
    preparation.mkdir(parents=True)
    pd.read_parquet(source / "labels.parquet").drop(
        columns=["label", "label_source"], errors="ignore"
    ).assign(resource_valid=True).to_parquet(preparation / "resource_windows.parquet", index=False)
    output = tmp_path / "derived"

    derive_evaluation_bundle(source, preparation.parent, protocol_path, output)
    result = read_bundle(output)
    from src.service.labeling.evaluation import evaluation_counts

    assert result["manifest"]["evaluation_version"] == "evaluation-v3"
    assert result["evaluation"]["evaluation_eligible"].sum() == 0
    assert result["labels"].label.equals(read_bundle(source)["labels"].label)
    assert result["splits"].split.equals(read_bundle(source)["splits"].split)
    assert result["evaluation_coverage"] == evaluation_counts(
        result["evaluation"], result["labels"], setup[2].grid
    )
    assert {
        "by_case",
        "by_source",
        "input_valid",
        "exclusion_reasons",
        "total",
    } <= set(result["evaluation_coverage"])


def test_derive_evaluation_bundle_rejects_changed_preparation(setup, tmp_path, monkeypatch):
    from dataclasses import replace

    from src.service.labeling.bundle import derive_evaluation_bundle

    build(setup, "source")
    source = setup[0] / "source/labels"
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["evaluation_version"] = "evaluation-v3"
    protocol_path = tmp_path / "evaluation-v3.json"
    protocol_path.write_text(json.dumps(protocol))
    changed = replace(setup[2], preparation_manifest_sha256="changed")
    monkeypatch.setattr(bundle, "load_evidence", lambda *args: changed)
    with pytest.raises(ValueError, match="Preparation or source evidence"):
        derive_evaluation_bundle(
            source, tmp_path / "preparation", protocol_path, tmp_path / "derived"
        )
    assert not (tmp_path / "derived").exists()


def test_derive_evaluation_bundle_rejects_label_rule_changes(setup, tmp_path):
    from src.service.labeling.bundle import derive_evaluation_bundle

    build(setup, "source")
    source = setup[0] / "source/labels"
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["evaluation_version"] = "evaluation-v3"
    protocol["positive_rule"] = "changed"
    protocol_path = tmp_path / "evaluation-v3.json"
    protocol_path.write_text(json.dumps(protocol))

    with pytest.raises(ValueError, match="label or split rules"):
        derive_evaluation_bundle(source, tmp_path / "missing", protocol_path, tmp_path / "derived")
