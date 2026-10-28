"""Independent acceptance and explicit score eligibility regression tests."""

import json

import pandas as pd
import pytest

from src.service.labeling import acceptance, bundle
from src.service.preparation import file_digest
from tests.test_service_labeling_bundle import setup  # noqa: F401


def prepared(setup, monkeypatch):
    root, protocol, index, _ = setup
    raw = root / "source.json"
    raw.write_text("{}")
    index.sources["source"].update(path=str(raw), sha256=file_digest(raw))
    monkeypatch.setattr(acceptance, "load_evidence", lambda *args: index)
    bundle.build_bundle(root / "raw", root / "prepared", protocol, root / "run")
    return root, index


def test_public_reader_preserves_versions_for_score_join(setup, monkeypatch):
    root, index = prepared(setup, monkeypatch)
    package = bundle.read_bundle(root / "run")
    labels = package["labels"]
    splits = package["splits"]
    test_ids = set(splits.loc[splits["split"].eq("test"), "sample_id"])
    expected = set(
        labels.loc[labels["sample_id"].isin(test_ids) & labels["label"].notna(), "sample_id"].head(
            1
        )
    )
    scores = labels.loc[labels.sample_id.isin(expected), ["sample_id", "label"]].copy()
    scores["score"] = 0.5
    scores.attrs.update(labels.attrs)
    joined = acceptance.join_scores(labels, splits, scores, expected)
    assert set(joined.sample_id) == expected


def test_conservation_and_tamper(setup, monkeypatch):
    root, index = prepared(setup, monkeypatch)
    assert acceptance.check_bundle(root / "run", root / "prepared")["rows"] == 3
    index.grid.drop(index=0, inplace=True)
    with pytest.raises(ValueError):
        acceptance.check_bundle(root / "run", root / "prepared")


def test_derived_bundle_replays_with_historical_rule_identity(setup, monkeypatch):
    root, _ = prepared(setup, monkeypatch)
    source = root / "run/labels"
    protocol = json.loads((source / "protocol.json").read_text())
    protocol["evaluation_version"] = "evaluation-v3"
    protocol["evaluation_excluded_cases"] = [next(iter(protocol["normal_splits"]))]
    protocol_path = root / "v3.json"
    protocol_path.write_text(json.dumps(protocol))
    bundle.derive_evaluation_bundle(source, root / "prepared", protocol_path, root / "derived")
    monkeypatch.setattr(bundle, "implementation_hash", lambda protocol: "different-current-hash")
    assert acceptance.check_bundle(root / "derived", root / "prepared")["rows"] == 3
    (source / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="source manifest mismatch"):
        acceptance.check_bundle(root / "derived", root / "prepared")


def test_check_cli_receipt_only_after_success(setup, monkeypatch, capsys):
    from scripts import label_service_data

    root, _ = prepared(setup, monkeypatch)
    receipt = root / "acceptance.json"
    args = [
        "check",
        "--bundle",
        str(root / "run"),
        "--preparation",
        str(root / "prepared"),
        "--receipt",
        str(receipt),
    ]
    assert label_service_data.main(args) == 0
    record = json.loads(receipt.read_text())
    assert record["status"] == "success"
    assert record["bundle_manifest_sha256"] == file_digest(root / "run/labels/manifest.json")
    assert record["result"]["rows"] == 3
    assert label_service_data.main(args) == 1
    receipt.unlink()
    (root / "source.json").write_text('{"changed":true}')
    assert label_service_data.main(args) == 1
    assert not receipt.exists()


def test_source_hash_revalidation(setup, monkeypatch):
    root, _ = prepared(setup, monkeypatch)
    (root / "source.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Source hash"):
        acceptance.check_bundle(root / "run", root / "prepared")


def frames():
    labels = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": pd.array([0, 1, None], dtype="Int8"),
            "label_version": ["L"] * 3,
        }
    )
    splits = pd.DataFrame(
        {"sample_id": ["a", "b", "c"], "split": ["test"] * 3, "split_version": ["S"] * 3}
    )
    scores = pd.DataFrame({"sample_id": ["a", "b", "c"], "score": [0.1, 0.9, 0.5]})
    for frame in [labels, splits, scores]:
        frame.attrs.update(label_version="L", split_version="S")
    return labels, splits, scores


def test_explicit_score_mask():
    labels, splits, scores = frames()
    assert list(acceptance.join_scores(labels, splits, scores, {"a", "b"}).sample_id) == ["a", "b"]
    for bad in [
        scores.iloc[:1],
        pd.concat([scores, scores.iloc[:1]]),
        scores.assign(score=[float("nan"), 0.9, 0.5]),
    ]:
        bad.attrs.update(scores.attrs)
        with pytest.raises(ValueError):
            acceptance.join_scores(labels, splits, bad, {"a", "b"})
    scores.attrs["label_version"] = "wrong"
    with pytest.raises(ValueError):
        acceptance.join_scores(labels, splits, scores, {"a", "b"})


def test_read_bundle_versions_join_scores(setup, monkeypatch):
    root, _ = prepared(setup, monkeypatch)
    data = bundle.read_bundle(root / "run")
    sample_ids = list(data["labels"].sample_id)
    scores = pd.DataFrame({"sample_id": sample_ids, "score": [0.1, 0.9, 0.5]})
    scores.attrs.update(
        label_version=data["manifest"]["label_version"],
        split_version=data["manifest"]["split_version"],
    )
    expected_ids = set(data["splits"].loc[data["splits"].split.eq("test"), "sample_id"])
    joined = acceptance.join_scores(data["labels"], data["splits"], scores, expected_ids)
    assert set(joined.sample_id) == expected_ids
    assert data["labels"].attrs["label_version"] == data["manifest"]["label_version"]
    assert data["splits"].attrs["split_version"] == data["manifest"]["split_version"]


def test_unknown_and_single_class():
    labels, splits, scores = frames()
    with pytest.raises(ValueError, match="Unknown"):
        acceptance.join_scores(labels, splits, scores, {"c"})
    result = acceptance.join_scores(labels, splits, scores, {"a"})
    assert result.attrs["metric_availability"] == {
        "auroc": None,
        "ap": None,
        "reason": "single_class",
    }


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "illegal", "dangling", "coverage", "provenance", "hash"]
)
def test_reject_tampered_bundle(setup, monkeypatch, mutation):
    root, _ = prepared(setup, monkeypatch)
    path = root / "run/labels"
    labels = pd.read_parquet(path / "labels.parquet")
    if mutation == "missing":
        labels = labels.iloc[1:]
    elif mutation == "duplicate":
        labels = pd.concat([labels, labels.iloc[:1]])
    elif mutation == "illegal":
        labels.loc[0, "label"] = 2
    elif mutation == "dangling":
        labels.at[0, "evidence_ids"] = ["missing-source"]
    elif mutation == "provenance":
        labels.loc[0, "label_reason"] = "forged-reason"
    if mutation in {"missing", "duplicate", "illegal", "dangling", "provenance"}:
        labels.to_parquet(path / "labels.parquet", index=False)
    elif mutation == "coverage":
        (path / "coverage.json").write_text("{}")
    else:
        (path / "exceptions.json").write_text("{}")
    # Re-sign file hashes and semantic hashes: independent replay must still fail.
    if mutation != "hash":
        import json

        manifest = json.loads((path / "manifest.json").read_text())
        for name in bundle.FILES:
            manifest["files"][name]["sha256"] = file_digest(path / name)
        if mutation in {"missing", "duplicate", "illegal", "dangling", "provenance"}:
            splits = pd.read_parquet(path / "splits.parquet")
            evidence = json.loads((path / "evidence.json").read_text())
            protocol = json.loads((path / "protocol.json").read_text())
            hashes = bundle._semantic(protocol, evidence, labels, splits)
            version = bundle._label_version(hashes)
            labels["label_version"] = version
            labels.to_parquet(path / "labels.parquet", index=False)
            manifest["semantic_hashes"] = hashes
            manifest["label_version"] = version
            manifest["files"]["labels.parquet"]["sha256"] = file_digest(path / "labels.parquet")
        (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        acceptance.check_bundle(root / "run", root / "prepared")


def test_old_gate_unchanged(setup, monkeypatch, tmp_path):
    import json

    from src.service import preparation_acceptance as old
    from src.service.modality_preparation import full_grid

    root, _ = prepared(setup, monkeypatch)
    assert (
        acceptance.check_bundle(root / "run", root / "prepared")["coverage"]["total"]["negative"]
        == 3
    )
    grid = full_grid("tt", "normal", ["svc"], 0, 15000)
    for name in [
        "resource_windows",
        "execution_windows",
        "log_windows",
        "labels",
        "splits",
        "eligibility_masks",
    ]:
        table = grid.copy()
        if name == "labels":
            table["label"] = pd.array([0], dtype="Int8")
        table.to_parquet(tmp_path / f"{name}.parquet", index=False)
        (tmp_path / f"{name}.schema.json").write_text(
            json.dumps({"fields": [{"name": c} for c in table]})
        )
    with pytest.raises(ValueError, match="all labels Unknown"):
        old.inspect_dataset(tmp_path)


def test_output_path_safety(setup, monkeypatch):
    from scripts.label_service_data import main

    root, path, _, _ = setup
    args = [
        "build",
        "--tt-root",
        str(root / "raw"),
        "--preparation",
        str(root / "prepared"),
        "--protocol",
        str(path),
        "--output",
    ]
    for target in [
        root / "raw/new",
        root / "prepared/new",
        __import__("pathlib").Path("outputs/service"),
    ]:
        assert main(args + [str(target)]) == 1
    prepared(setup, monkeypatch)
    assert main(args + [str(root / "run")]) == 1
    assert (
        main(["check", "--bundle", str(root / "run"), "--preparation", str(root / "prepared")]) == 0
    )
    assert (
        main(["check", "--bundle", str(root / "absent"), "--preparation", str(root / "prepared")])
        == 1
    )
    assert (
        main(
            [
                "diff",
                "--old",
                str(root / "run"),
                "--new",
                str(root / "run"),
                "--output",
                str(root / "delta"),
            ]
        )
        == 0
    )


def test_public_output_root_guard(setup):
    from pathlib import Path

    root, protocol, _, _ = setup
    with pytest.raises(ValueError, match="outputs/service root"):
        bundle.build_bundle(root / "raw", root / "prepared", protocol, Path("outputs/service"))


def test_version_attrs_required_and_train_not_evaluated():
    labels, splits, scores = frames()
    scores.attrs.clear()
    with pytest.raises(ValueError, match="version"):
        acceptance.join_scores(labels, splits, scores, {"a"})
    scores.attrs.update(labels.attrs)
    labels.loc[0, "label_version"] = "other"
    with pytest.raises(ValueError, match="version"):
        acceptance.join_scores(labels, splits, scores, {"a"})
    labels.loc[0, "label_version"] = "L"
    splits.loc[0, "split"] = "fit"
    with pytest.raises(ValueError, match="fit/calibration"):
        acceptance.join_scores(labels, splits, scores, {"a"})


def test_join_rejects_conflicting_stable_identity():
    labels, splits, scores = frames()
    labels["run_id"] = "case-a"
    splits["run_id"] = "case-b"
    with pytest.raises(ValueError, match="identity"):
        acceptance.join_scores(labels, splits, scores, {"a"})
