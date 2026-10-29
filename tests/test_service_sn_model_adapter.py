import pandas as pd

from src.service.modeling import sn_adapter
from src.service.modeling.sn_adapter import SN_RESOURCE_MAPPING, build_sn_prepared
from src.service.modeling.temporal import temporal_inputs


def _write_fixture(root):
    sn = root / "sn"
    sn.mkdir(parents=True)
    runs = [
        ("fit", "SN.2022-04-20T182405D2022-04-20T184806"),
        ("calibration", "SN.2022-04-21T105249D2022-04-21T111651"),
        ("holdout", "SN.2022-04-21T153302D2022-04-21T155703"),
        ("test", "SN.2022-04-17T190213D2022-04-17T192544"),
    ]
    rows = []
    for role, run_id in runs:
        for service in ["svc-a", "svc-b"]:
            for start in [0, 15000, 30000]:
                rows.append(
                    {
                        "dataset_id": "eadro_sn_v1",
                        "run_id": run_id,
                        "canonical_service_id": service,
                        "window_start_ms": start,
                        "window_end_ms": start + 15000,
                        "sample_id": f"{run_id}-{service}-{start}",
                        "role": role,
                    }
                )
    identity = pd.DataFrame(rows)
    resource = identity.drop(columns="role").copy()
    resource["cpu_usage_total_mean"] = 0.1
    resource["memory_usage_mean"] = 0.2
    resource["resource_sample_count"] = 15.0
    resource["resource_coverage"] = 1.0
    for column in [
        "cpu_usage_system_mean",
        "cpu_usage_user_mean",
        "memory_working_set_mean",
        "rx_bytes_mean",
        "tx_bytes_mean",
    ]:
        resource[column] = 0.0
    resource["quality_status"] = "OBSERVED_RAW_UNKNOWN_SEMANTICS"
    execution = identity.drop(columns="role").copy()
    execution["observation_count"] = 2.0
    execution["duration_mean_ms"] = 1.0
    execution["duration_p95_ms"] = 1.0
    execution["duration_max_ms"] = 1.0
    execution["error_known_count"] = 2.0
    execution["error_count"] = 0.0
    execution["error_coverage"] = 1.0
    execution["quality_status"] = "OBSERVED"
    logs = identity.drop(columns="role").copy()
    logs["messages_json"] = "[]"
    logs["event_count"] = 1.0
    logs["quality_status"] = "OBSERVED"
    labels = identity.drop(columns="role").copy()
    labels["label"] = pd.Series([None] * len(labels), dtype="Float64")
    labels["label_reason"] = "unknown"
    splits = identity.drop(columns="role").copy()
    role_map = identity.set_index("sample_id").role
    splits["run_role"] = splits.sample_id.map(role_map)
    splits["reference_eligible"] = splits.run_role.isin(["fit", "calibration", "holdout"])
    splits["reference_reason"] = "fixture"
    splits["training_eligible"] = splits.run_role.eq("fit")
    splits["training_reason"] = "fixture"
    splits["calibration_eligible"] = splits.run_role.eq("calibration")
    splits["calibration_reason"] = "fixture"
    splits["eligibility_scope"] = "fixture"
    resource.to_parquet(sn / "resource_windows.parquet", index=False)
    execution.to_parquet(sn / "execution_windows.parquet", index=False)
    logs.to_parquet(sn / "log_windows.parquet", index=False)
    labels.to_parquet(sn / "labels.parquet", index=False)
    splits.to_parquet(sn / "splits.parquet", index=False)
    return root


def test_sn_adapter_keeps_role_pools_and_source_mapping(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sn_adapter,
        "EXPECTED_SN_ROLES",
        {
            "SN.2022-04-20T182405D2022-04-20T184806": "fit",
            "SN.2022-04-21T105249D2022-04-21T111651": "calibration",
            "SN.2022-04-21T153302D2022-04-21T155703": "holdout",
            "SN.2022-04-17T190213D2022-04-17T192544": "test",
        },
    )
    prepared = build_sn_prepared(_write_fixture(tmp_path))
    assert len(prepared.sample_ids) == 24
    assert len(prepared.lists.fit) == 6
    assert len(prepared.lists.calibration) == 6
    assert prepared.provenance["resource_mapping"] == SN_RESOURCE_MAPPING
    assert prepared.source_labels.label.isna().all()
    assert set(prepared.labels.label.dropna().unique()) == {0, 1}


def test_sn_temporal_inputs_preserve_fixed_shape_and_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sn_adapter,
        "EXPECTED_SN_ROLES",
        {
            "SN.2022-04-20T182405D2022-04-20T184806": "fit",
            "SN.2022-04-21T105249D2022-04-21T111651": "calibration",
            "SN.2022-04-21T153302D2022-04-21T155703": "holdout",
            "SN.2022-04-17T190213D2022-04-17T192544": "test",
        },
    )
    prepared = build_sn_prepared(_write_fixture(tmp_path))
    inputs = temporal_inputs(prepared, context_windows=4)
    assert inputs["context"].shape == (24, 4, 28)
    assert inputs["values"].shape == (24, 12)
    assert inputs["observed"].dtype == bool
    assert inputs["availability"].shape == (24, 3)
