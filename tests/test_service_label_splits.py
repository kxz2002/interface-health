"""Task3: evaluation labels do not determine reference qualification."""

import importlib
import json

import pandas as pd
import pytest

from src.service.modality_preparation import full_grid


def api():
    spec = importlib.util.find_spec("src.service.label_split_preparation")
    assert spec is not None, "Independent labels/splits module is missing"
    return importlib.import_module("src.service.label_split_preparation")


def fixture():
    runs = [
        dict(run_id=f"n{i}", is_fault_run=False, split=r)
        for i, r in enumerate(["fit", "fit", "fit", "calibration", "normal_test", "normal_test"])
    ]
    runs += [dict(run_id="fault", is_fault_run=True, split="fault_test")]
    inventory = {"tt": {"dataset_id": "tt", "runs": runs}, "sn": {"runs": {}}}
    grid = pd.concat(
        [full_grid("tt", r["run_id"], ["svc", "missing"], 0, 30000) for r in runs],
        ignore_index=True,
    )
    return inventory, grid


def test_unknown_labels_allow_authorized_reference_without_leaking_fault():
    inventory, grid = fixture()
    labels, splits = api().build_label_splits(grid, inventory)
    assert len(labels) == len(splits) == len(grid)
    assert labels.label.isna().all()
    assert set(labels) == set(grid) | {"label", "label_reason"}
    assert splits.groupby("run_id").run_role.nunique().eq(1).all()
    assert splits[splits.run_id.eq("fault")].training_eligible.eq(False).all()
    assert splits[splits.run_id.eq("fault")].calibration_eligible.eq(False).all()
    assert splits[splits.run_id.eq("n0")].training_eligible.all()
    assert splits[splits.run_id.eq("n3")].calibration_eligible.all()
    assert splits[splits.run_id.eq("n4")].run_role.eq("holdout").all()


def test_bad_tt_role_is_rejected():
    inventory, grid = fixture()
    inventory["tt"]["runs"][-1]["split"] = "fit"
    with pytest.raises(ValueError, match="role"):
        api().build_label_splits(grid, inventory)


def sn_fixture():
    profile = json.loads(
        __import__("pathlib").Path("configs/service/sn_exploratory_v1.json").read_text()
    )
    runs = {
        rid: {
            "metadata": {"faults": ["exposure"] if r["is_fault_run"] else []},
            "container": r["container"],
            "is_normal_archive": not r["is_fault_run"],
        }
        for rid, r in profile["runs"].items()
    }
    grid = pd.concat(
        [full_grid("eadro_sn_v1", rid, ["svc"], 0, 15000) for rid in runs], ignore_index=True
    )
    sources = {
        "sn": {
            "status": "EXPLORATORY_ASSUMPTION_BOUND",
            "resolved_profile": profile,
            "runs": {
                rid: {"role_evidence": r, "sources": [{"sha256": "fixture"}]}
                for rid, r in profile["runs"].items()
            },
        }
    }
    return {"tt": {"dataset_id": "tt", "runs": []}, "sn": {"runs": runs}}, grid, sources


def test_sn_reference_is_explicit_and_strict_stays_blocked():
    inventory, grid, sources = sn_fixture()
    _, strict = api().build_label_splits(grid, inventory)
    assert not strict.reference_eligible.any()
    assert strict.run_role.eq("fit").sum() == 1
    assert strict.run_role.eq("calibration").sum() == 1
    labels, splits = api().build_label_splits(
        grid, inventory, sn_profile="exploratory", sources=sources
    )
    assert labels.label.isna().all()
    assert splits.training_eligible.sum() == 1
    assert splits.calibration_eligible.sum() == 1
    assert splits.reference_reason.str.contains("health_gold").any()
    normal = next(r for r, x in inventory["sn"]["runs"].items() if x["is_normal_archive"])
    inventory["sn"]["runs"][normal]["metadata"]["faults"] = ["unexpected"]
    with pytest.raises(ValueError, match="qualification"):
        api().build_label_splits(grid, inventory, sn_profile="exploratory", sources=sources)


def test_writer_preserves_grid_masks_and_exposure_separation(tmp_path):
    inventory, grid = fixture()
    grid.to_parquet(tmp_path / "windows.parquet")
    (tmp_path / "modality_sources.json").write_text("{}")
    directory = tmp_path / "tt"
    directory.mkdir()
    grid.assign(resource_valid=False, quality_status="RESOURCE_INVALID").to_parquet(
        directory / "resource_windows.parquet"
    )
    for name in ["execution", "log"]:
        grid.assign(quality_status="NO_OBSERVATIONS").to_parquet(
            directory / f"{name}_windows.parquet"
        )
    summary = api().write_label_split_tables(inventory, tmp_path)
    assert summary["labels_known"] == 0
    assert len(pd.read_parquet(directory / "splits.parquet")) == len(grid)
    masks = pd.read_parquet(directory / "eligibility_masks.parquet")
    assert not masks.resource_valid.any()
    assert "label" not in masks
    assert (directory / "exposures.json").is_file()
    with pytest.raises(FileExistsError):
        api().write_label_split_tables(inventory, tmp_path)


def test_changed_modality_identity_and_sample_id_are_rejected(tmp_path):
    inventory, grid = fixture()
    changed = grid.copy()
    changed.loc[0, "sample_id"] = "bad"
    with pytest.raises(ValueError, match="sample"):
        api().build_label_splits(changed, inventory)
    changed.to_parquet(tmp_path / "resource_windows.parquet")
    with pytest.raises(ValueError, match="identity"):
        api().modality_masks(grid, tmp_path)


def test_cli_writes_labels_and_splits(tmp_path, monkeypatch):
    from scripts import prepare_service_data as cli

    inventory, grid = fixture()
    output = tmp_path / "out"
    monkeypatch.setattr(cli, "validate_output", lambda *args: None)
    monkeypatch.setattr(cli, "selected_tt_inventory", lambda *args: inventory["tt"])
    monkeypatch.setattr(cli, "cached_sn_inventory", lambda *args, **kwargs: inventory["sn"])

    def windows(*args, **kwargs):
        grid.to_parquet(output / "windows.parquet")
        (output / "modality_sources.json").write_text("{}")
        directory = output / "tt"
        directory.mkdir()
        for name in ["resource", "execution", "log"]:
            grid.assign(quality_status="NO_OBSERVATIONS").to_parquet(
                directory / f"{name}_windows.parquet"
            )
        return {"training_ready": False}

    monkeypatch.setattr(cli, "write_window_tables", windows)
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--tt-root",
            str(tmp_path),
            "--sn-archive",
            "unused",
            "--sn-inventory",
            "unused",
            "--sn-inventory-sha256",
            "unused",
            "--output",
            str(output),
        ],
    )
    cli.main()
    assert (output / "tt/labels.parquet").is_file()
    assert json.loads((output / "summary.json").read_text())["labels_known"] == 0


def test_exposure_metadata_hash_and_declaration_do_not_label(tmp_path):
    from src.service.preparation import file_digest

    inventory, grid = fixture()
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "case_id": "fault",
                "is_anomaly_case": True,
                "inject_start_ms": 100,
                "inject_end_ms": 200,
                "planned": {"fault": "CPU"},
            }
        )
    )
    inventory["tt"]["runs"][-1]["sources"] = {
        "source_metadata": {"path": "meta.json", "sha256": file_digest(metadata)}
    }
    exposure = api().exposure_index(inventory, "tt", tmp_path)
    assert not exposure["prediction_input"]
    assert exposure["runs"][-1]["declared_exposure"]["inject_start_ms"] == 100
    labels, _ = api().build_label_splits(grid, inventory)
    assert labels.label.isna().all()
    metadata.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        api().exposure_index(inventory, "tt", tmp_path)
