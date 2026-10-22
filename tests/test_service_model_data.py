import json
from pathlib import Path

import pandas as pd
import pytest

from src.service.labeling import bundle
from src.service.labeling.evidence import EvidenceIndex
from src.service.modality_preparation import full_grid
from src.service.preparation import file_digest
from src.service.window_preparation import table_schema


@pytest.fixture
def inputs(tmp_path, monkeypatch, request):
    preparation = tmp_path / "prepared"
    directory = preparation / "tt"
    directory.mkdir(parents=True)
    protocol = json.loads(Path("configs/service/tt_labels_a_v2.json").read_text())
    cid = next(iter(protocol["normal_splits"]))
    dataset_id = getattr(request, "param", "tt")
    grid = full_grid(dataset_id, cid, ["svc"], 0, 45000).assign(case_id=cid)
    resource = grid.assign(
        cpu_cores=1.0,
        memory_usage_ratio=0.5,
        process_count=2.0,
        resource_valid=True,
        quality_status="RESOURCE_VALID",
    )
    execution = grid.assign(
        observation_count=1.0,
        duration_mean_ms=2.0,
        duration_p95_ms=2.0,
        duration_max_ms=2.0,
        error_known_count=1.0,
        error_count=0.0,
        error_coverage=1.0,
        quality_status="OBSERVED",
    )
    logs = grid.assign(
        messages_json='[{"message":"hello","count":1}]', event_count=1.0, quality_status="OBSERVED"
    )
    for name, frame in zip(
        ["resource_windows", "execution_windows", "log_windows"], [resource, execution, logs]
    ):
        frame.attrs.clear()
        frame.to_parquet(directory / f"{name}.parquet", index=False)
        (directory / f"{name}.schema.json").write_text(json.dumps(table_schema(frame, name, "tt")))
    grid.assign(run_role="old-role").to_parquet(directory / "splits.parquet", index=False)
    manifest = {
        "artifacts": {
            str(path.relative_to(preparation)): {"sha256": file_digest(path)}
            for path in directory.iterdir()
        }
    }
    (preparation / "artifact_manifest.json").write_text(json.dumps(manifest))
    case = dict(
        is_fault=False,
        admission={"admitted": True},
        boundary=None,
        transport_exceptions=[dict(start_ms=0, end_ms=15000, evidence_id="source")],
        evidence_ids=["source"],
        candidate_services=["svc"],
    )
    index = EvidenceIndex(
        resource,
        {cid: case},
        {"source": dict(path="synthetic", sha256="abc", source_locator="$")},
        protocol,
        file_digest(preparation / "artifact_manifest.json"),
    )
    monkeypatch.setattr(bundle, "load_evidence", lambda *args: index)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    bundle.build_bundle(tmp_path / "raw", preparation, protocol_path, tmp_path / "labels-run")
    return preparation, tmp_path / "labels-run/labels"


def load(inputs):
    from src.service.modeling.config import load_config
    from src.service.modeling.data import load_inputs

    return load_inputs(*inputs, load_config(Path("configs/service/tt_model_v1.json")))


def test_roundtrip_versions_and_grid(inputs):
    data = load(inputs)
    assert len(data.grid) == len(data.labels) == len(data.evaluation) == 3
    assert data.splits.split.eq("fit").all()
    assert data.labels.attrs["label_version"] == data.versions["label_version"]
    assert data.evaluation.attrs["evaluation_version"] == data.versions["evaluation_version"]
    assert data.hashes["preparation_manifest_sha256"] == file_digest(
        inputs[0] / "artifact_manifest.json"
    )
    assert data.hashes["evaluation_sha256"] == file_digest(inputs[1] / "evaluation.parquet")
    from src.service.modality_preparation import GRID_COLUMNS

    expected = data.grid[GRID_COLUMNS]
    for frame in [
        data.resource,
        data.execution,
        data.log,
        data.labels,
        data.splits,
        data.evaluation,
    ]:
        pd.testing.assert_frame_equal(frame[GRID_COLUMNS], expected)
        assert frame.attrs == data.versions
    manifest = json.loads((inputs[1] / "manifest.json").read_text())
    assert data.versions == {
        "label_version": manifest["label_version"],
        "split_version": manifest["split_version"],
        "evaluation_version": "tt_normal_test_and_controlled_positive_v2",
        "preparation_schema_version": "service-preparation-v1",
    }
    for name in ["label", "split", "evaluation"]:
        table = {"label": "labels", "split": "splits", "evaluation": "evaluation"}[name]
        assert data.hashes[f"{name}_sha256"] == file_digest(inputs[1] / f"{table}.parquet")
    assert data.hashes["label_manifest_sha256"] == file_digest(inputs[1] / "manifest.json")
    assert data.hashes["config_sha256"] == data.config["config_sha256"]
    for name in ["resource", "execution", "log"]:
        for suffix in ["parquet", "schema.json"]:
            relative = f"tt/{name}_windows.{suffix}"
            assert data.hashes[relative] == file_digest(inputs[0] / relative)


def test_unknown_grid_rows_retained(inputs):
    data = load(inputs)
    assert len(data.grid) == 3
    assert data.labels.label.isna().sum() == 1
    assert data.grid.sample_id.tolist() == data.labels.sample_id.tolist()


@pytest.mark.parametrize("inputs", ["tt", "service_20261017_004546"], indirect=True)
def test_tt_dataset_identity_preserved(inputs):
    data = load(inputs)
    source = pd.read_parquet(inputs[0] / "tt/resource_windows.parquet")
    pd.testing.assert_series_equal(data.grid.sample_id, source.sample_id)
    pd.testing.assert_series_equal(data.grid.dataset_id, source.dataset_id)


@pytest.mark.parametrize("inputs", ["sn", "service_20261018_unknown"], indirect=True)
def test_reject_unapproved_dataset_identity(inputs):
    with pytest.raises(ValueError, match="Only TT inputs supported"):
        load(inputs)


def test_reject_mixed_tt_dataset_identity():
    from src.service.modeling.data import _identity

    grid = pd.concat(
        [
            full_grid(dataset, "normal", ["svc"], 0, 45000)
            for dataset in ["tt", "service_20261017_004546"]
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="Only TT inputs supported"):
        _identity(grid)


def test_evaluation_content_binding(inputs):
    load(inputs)
    path = inputs[1] / "evaluation.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "evaluation_eligible"] = True
    frame.to_parquet(path, index=False)
    manifest_path = inputs[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name]["sha256"] = file_digest(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Evaluation membership"):
        load(inputs)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "extra", "sample", "end", "version", "hash"]
)
def test_identity_and_feature_allowlist(inputs, mutation):
    data = load(inputs)
    assert set(data.feature_tables["resource"]) == {
        "cpu_cores",
        "memory_usage_ratio",
        "process_count",
    }
    assert not {"case_id", "target", "phase", "probe", "label"} & set(
        data.feature_tables["execution"]
    )
    path = inputs[0] / "tt/execution_windows.parquet"
    frame = pd.read_parquet(path)
    if mutation == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif mutation == "missing":
        frame = frame.iloc[:-1]
    elif mutation == "extra":
        frame.loc[0, "canonical_service_id"] = "extra"
    elif mutation == "sample":
        frame.loc[0, "sample_id"] = "wrong"
    elif mutation == "end":
        frame.loc[0, "window_end_ms"] += 1
    elif mutation == "hash":
        frame.loc[0, "observation_count"] = 2.0
    elif mutation == "version":
        path = inputs[0] / "tt/execution_windows.schema.json"
        schema = json.loads(path.read_text())
        schema["schema_version"] = "fake"
        path.write_text(json.dumps(schema))
    if mutation != "version":
        frame.to_parquet(path, index=False)
    if mutation != "hash":
        manifest_path = inputs[0] / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"][str(path.relative_to(inputs[0]))]["sha256"] = file_digest(path)
        manifest_path.write_text(json.dumps(manifest))
        from src.service.modeling import data as model_data

        original = model_data.read_bundle

        def bound_bundle(path):
            result = original(path)
            digest = file_digest(manifest_path)
            result["manifest"]["inputs"]["preparation_manifest_sha256"] = digest
            result["evidence"]["preparation_manifest_sha256"] = digest
            return result

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(model_data, "read_bundle", bound_bundle)
            with pytest.raises(ValueError):
                load(inputs)
        return
    with pytest.raises(ValueError):
        load(inputs)


@pytest.mark.parametrize("field", ["label_version", "split_version"])
def test_reject_fake_label_versions(inputs, field):
    path = inputs[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = "fake"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load(inputs)


def test_reject_preparation_binding_change(inputs):
    path = inputs[0] / "artifact_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["unbound_change"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding mismatch"):
        load(inputs)


def test_config_expansion_and_restricted_inputs(tmp_path):
    from src.service.modeling.config import load_config

    path = tmp_path / "config.json"
    path.write_text('{"training":{"prototype_seed":1}}')
    config = load_config(path)
    assert config["training"]["prototype_seed"] == 1
    assert config["training"]["epochs"] == 20
    assert len(config["config_sha256"]) == 64
    path.write_text('{"features":{"resource":["label"]}}')
    with pytest.raises(ValueError, match="features"):
        load_config(path)


def test_message_feature_projection():
    from src.service.modeling.data import _message_features

    messages = '[{"message":"hello", "count":2, "references":["target"], "phase":"inject"}]'
    assert json.loads(_message_features(messages)) == [{"message": "hello", "count": 2}]
    with pytest.raises(ValueError):
        _message_features('[{"message":"hello", "count":true}]')


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_hash",
        "missing_section",
        "missing_nested",
        "unknown_top",
        "unknown_nested",
        "tampered",
    ],
)
def test_resolved_configuration_contract(inputs, mutation):
    from src.service.modeling.config import content_hash, load_config
    from src.service.modeling.data import load_inputs

    config = load_config(Path("configs/service/tt_model_v1.json"))
    if mutation == "missing_hash":
        del config["config_sha256"]
    elif mutation == "missing_section":
        del config["training"]
    elif mutation == "missing_nested":
        del config["training"]["epochs"]
    elif mutation == "unknown_top":
        config["unknown"] = True
    elif mutation == "unknown_nested":
        config["training"]["unknown"] = True
    else:
        config["training"]["epochs"] = 21
    if mutation not in {"missing_hash", "tampered"}:
        config["config_sha256"] = content_hash(
            {key: value for key, value in config.items() if key != "config_sha256"}
        )
    with pytest.raises(ValueError, match="[Cc]onfiguration"):
        load_inputs(*inputs, config)
