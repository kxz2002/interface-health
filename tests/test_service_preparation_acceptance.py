import json

import pytest

from src.service.modality_preparation import full_grid


@pytest.mark.parametrize("manifest_format", ["new", "legacy", "summary"])
def test_acceptance_preserves_production_hash_on_revalidation(
    tmp_path, monkeypatch, manifest_format
):
    from pathlib import Path

    from src.service import preparation_acceptance as acceptance

    output = tmp_path / "production"
    output.mkdir()
    for dataset in ["tt", "sn"]:
        (output / dataset).mkdir()
        write_fixture(output / dataset)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}")
    production_hash = {"parser.py": "production-original"}
    summary = {
        "implementation_sha256": production_hash,
        "resolved_config": {
            "sn_assumption_profile": {},
            "sn_profile_config": str(profile_path),
            "sn_profile_config_sha256": acceptance.file_digest(profile_path),
            "sn_profile": "exploratory",
        },
    }
    (output / "summary.json").write_text(json.dumps(summary))
    inventory = {
        "tt": {
            "dataset_id": "dataset",
            "runs": [dict(run_id="run", candidate_services=["service"], start_ms=0, end_ms=30000)],
        },
        "sn": {"runs": {"run": dict(metrics={"service": {}}, metadata=dict(start=0, end=30))}},
    }
    (output / "inventory.json").write_text(json.dumps(inventory))
    (output / "modality_sources.json").write_text(json.dumps({"tt": {}, "sn": {"runs": {}}}))
    manual = tmp_path / "manual.json"
    manual.write_text(
        json.dumps(
            {
                "checks": [
                    dict(dataset=dataset, role=role, passed=True)
                    for dataset in ["tt", "sn"]
                    for role in ["Normal", "Fault"]
                ]
            }
        )
    )
    permissions = {
        "run": dict(
            run_role="fit",
            reference_eligible=True,
            training_eligible=True,
            calibration_eligible=False,
        )
    }
    monkeypatch.setattr(
        acceptance,
        "expected_split_permissions",
        lambda *args, **kwargs: {"tt": permissions, "sn": permissions},
    )
    monkeypatch.setattr(
        acceptance, "full_grid", lambda *args: full_grid("dataset", "run", ["service"], 0, 30000)
    )
    acceptance.write_acceptance(output, manual_evidence=manual)
    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["production_source_sha256"] == production_hash
    assert manifest["validation_source_sha256"] != production_hash
    if manifest_format == "legacy":
        manifest["source_sha256"] = manifest.pop("production_source_sha256")
    elif manifest_format == "summary":
        manifest.pop("production_source_sha256")
    manifest_path.write_text(json.dumps(manifest))
    parent_hash = acceptance.file_digest(manifest_path)
    destination = tmp_path / "revalidation"
    acceptance.write_acceptance(output, manual_evidence=manual, report_directory=destination)
    renewed = json.loads((destination / "artifact_manifest.json").read_text())
    assert renewed["production_source_sha256"] == production_hash
    assert renewed["production_manifest_sha256"] == parent_hash
    assert renewed["validation_source_sha256"][
        "src/service/preparation_acceptance.py"
    ] == acceptance.file_digest(Path("src/service/preparation_acceptance.py"))


def test_acceptance_rejects_missing_modality_grid_and_counts_unknown(tmp_path):
    from src.service.preparation_acceptance import inspect_dataset

    grid = full_grid("dataset", "run", ["service"], 0, 30000)
    for name in ["resource_windows", "execution_windows", "log_windows"]:
        grid.assign(quality_status="NO_OBSERVATIONS").to_parquet(tmp_path / f"{name}.parquet")
    grid.assign(label=None, label_reason="independent_evidence_absent").to_parquet(
        tmp_path / "labels.parquet"
    )
    grid.assign(
        run_role="fit", reference_eligible=True, training_eligible=True, calibration_eligible=False
    ).to_parquet(tmp_path / "splits.parquet")
    grid.to_parquet(tmp_path / "eligibility_masks.parquet")
    import pandas as pd

    for path in tmp_path.glob("*.parquet"):
        path.with_suffix(".schema.json").write_text(
            json.dumps({"fields": [{"name": c} for c in pd.read_parquet(path).columns]})
        )
    result = inspect_dataset(tmp_path, expected_grid=grid)
    assert result["rows"] == 2
    assert result["labels_unknown"] == 2
    assert result["training_pool_rows_before_model_masks"] == 2
    assert result["modalities"]["execution_windows"]["quality_counts"] == {"NO_OBSERVATIONS": 2}
    grid.assign(
        run_role="test",
        reference_eligible=False,
        training_eligible=True,
        calibration_eligible=False,
    ).to_parquet(tmp_path / "splits.parquet")
    with pytest.raises(ValueError, match="Fault"):
        inspect_dataset(tmp_path)
    grid.assign(
        run_role="fit", reference_eligible=True, training_eligible=True, calibration_eligible=False
    ).to_parquet(tmp_path / "splits.parquet")
    grid.iloc[:1].to_parquet(tmp_path / "execution_windows.parquet")
    with pytest.raises(ValueError, match="grid"):
        inspect_dataset(tmp_path)


def test_acceptance_requires_source_linked_manual_evidence(tmp_path):
    from src.service.preparation_acceptance import write_acceptance

    bad = tmp_path / "manual.json"
    bad.write_text('{"checks": []}')
    with pytest.raises(ValueError, match="manual"):
        write_acceptance(tmp_path, manual_evidence=bad)


def write_fixture(directory, *, role="fit", reference=True, training=True, calibration=False):
    import pandas as pd

    grid = full_grid("dataset", "run", ["service"], 0, 30000)
    for name in ["resource_windows", "execution_windows", "log_windows"]:
        grid.assign(quality_status="NO_OBSERVATIONS").to_parquet(directory / f"{name}.parquet")
    grid.assign(label=None, label_reason="unknown").to_parquet(directory / "labels.parquet")
    grid.assign(
        run_role=role,
        reference_eligible=reference,
        training_eligible=training,
        calibration_eligible=calibration,
    ).to_parquet(directory / "splits.parquet")
    grid.to_parquet(directory / "eligibility_masks.parquet")
    for p in directory.glob("*.parquet"):
        columns = pd.read_parquet(p).columns
        p.with_suffix(".schema.json").write_text(
            json.dumps({"fields": [{"name": c} for c in columns]})
        )
    return grid


@pytest.mark.parametrize(
    "expected_role,expected_reference,field,value",
    [
        ("test", False, "run_role", "fit"),
        ("holdout", True, "training_eligible", True),
        ("holdout", True, "calibration_eligible", True),
        ("test", False, "reference_eligible", True),
    ],
)
def test_acceptance_binds_role_and_all_permissions(
    tmp_path, expected_role, expected_reference, field, value
):
    import pandas as pd

    from src.service.preparation_acceptance import inspect_dataset

    grid = write_fixture(tmp_path, role=expected_role, reference=expected_reference, training=False)
    table = pd.read_parquet(tmp_path / "splits.parquet")
    table[field] = value
    if field == "run_role":
        table["training_eligible"] = True
    table.to_parquet(tmp_path / "splits.parquet")
    expected = {
        "run": dict(
            run_role=expected_role,
            reference_eligible=expected_reference,
            training_eligible=False,
            calibration_eligible=False,
        )
    }
    with pytest.raises(ValueError, match="expected split"):
        inspect_dataset(tmp_path, expected_grid=grid, expected_splits=expected)


@pytest.mark.parametrize(
    "name",
    [
        "resource_windows",
        "execution_windows",
        "log_windows",
        "labels",
        "splits",
        "eligibility_masks",
    ],
)
def test_acceptance_requires_every_schema(tmp_path, name):
    from src.service.preparation_acceptance import inspect_dataset

    write_fixture(tmp_path)
    (tmp_path / f"{name}.schema.json").unlink()
    with pytest.raises(ValueError, match="Missing schema"):
        inspect_dataset(tmp_path)


def test_expected_permissions_derive_from_inventory_and_frozen_sn_profile():
    from pathlib import Path

    from src.service.preparation_acceptance import expected_split_permissions

    profile = json.loads(Path("configs/service/sn_exploratory_v1.json").read_text())
    tt_runs = [
        dict(
            run_id=f"normal{i}",
            is_fault_run=False,
            split="fit" if i < 3 else "calibration" if i == 3 else "normal_test",
        )
        for i in range(6)
    ]
    tt_runs.append(dict(run_id="fault", is_fault_run=True, split="fault_test"))
    sn_runs = {
        rid: dict(
            container=r["container"],
            is_normal_archive=not r["is_fault_run"],
            metadata={"faults": ["declared"] if r["is_fault_run"] else []},
        )
        for rid, r in profile["runs"].items()
    }
    inventory = {"tt": {"runs": tt_runs}, "sn": {"runs": sn_runs}}
    result = expected_split_permissions(inventory, sn_profile="exploratory", profile=profile)
    assert result["tt"]["fault"] == dict(
        run_role="test",
        reference_eligible=False,
        training_eligible=False,
        calibration_eligible=False,
    )
    assert result["tt"]["normal4"] == dict(
        run_role="holdout",
        reference_eligible=True,
        training_eligible=False,
        calibration_eligible=False,
    )
    for rid, spec in profile["runs"].items():
        assert result["sn"][rid]["training_eligible"] is (spec["proposed_role"] == "fit")
        assert result["sn"][rid]["calibration_eligible"] is (spec["proposed_role"] == "calibration")
        assert result["sn"][rid]["reference_eligible"] is (not spec["is_fault_run"])
    tt_runs[-1]["split"] = "fit"
    with pytest.raises(ValueError, match="TT inventory"):
        expected_split_permissions(inventory, sn_profile="exploratory", profile=profile)
