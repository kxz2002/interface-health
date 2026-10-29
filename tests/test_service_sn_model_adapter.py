from pathlib import Path

from src.service.modeling.sn_adapter import SN_RESOURCE_MAPPING, build_sn_prepared
from src.service.modeling.temporal import temporal_inputs

ROOT = Path("outputs/service/sn-preparation-20261029-task1")


def test_sn_adapter_keeps_role_pools_and_source_mapping():
    prepared = build_sn_prepared(ROOT)

    assert len(prepared.sample_ids) == 7908
    assert len(prepared.lists.fit) == 1140
    assert len(prepared.lists.calibration) == 1140
    assert prepared.provenance["resource_mapping"] == SN_RESOURCE_MAPPING
    assert prepared.source_labels.label.isna().all()
    assert set(prepared.labels.label.dropna().unique()) == {0, 1}
    assert set(prepared.lists.fit) <= set(prepared.lists.main_test) | set(prepared.sample_ids)


def test_sn_temporal_inputs_preserve_fixed_shape_and_mask():
    prepared = build_sn_prepared(ROOT)
    inputs = temporal_inputs(prepared, context_windows=4)

    assert inputs["context"].shape == (7908, 4, 28)
    assert inputs["values"].shape == (7908, 12)
    assert inputs["observed"].dtype == bool
    assert inputs["availability"].shape == (7908, 3)
    assert inputs["history_ids"][prepared.sample_ids[0]] == []
