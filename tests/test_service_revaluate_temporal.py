from types import SimpleNamespace

from scripts.service_revaluate_temporal import rebind
from src.service.modeling.data import InputTables, build_sample_lists
from tests.test_service_model_membership import membership_fixture


def test_rebind_keeps_training_membership_and_excludes_case_everywhere():
    tables, numeric, logs = membership_fixture()
    tables.labels["label_version"] = "old"
    tables.splits["split_version"] = "split"
    tables = InputTables(
        grid=tables.grid,
        resource=None,
        execution=None,
        log=None,
        labels=tables.labels,
        splits=tables.splits,
        evaluation=tables.evaluation,
        versions={"label_version": "old", "split_version": "split", "evaluation_version": "v2"},
        hashes={},
        feature_tables={},
        config={},
        protocol={},
    )
    original = build_sample_lists(tables, numeric, logs, "common")
    prepared = SimpleNamespace(
        hashes={"label_manifest_sha256": "source"},
        numeric=numeric,
        logs={"masks": logs.to_dict()},
        lists=original,
    )
    protocol = {"evaluation_version": "v3", "evaluation_excluded_cases": ["fault"]}
    labels = tables.labels.copy()
    labels["label_version"] = "new"
    bundle = {
        "manifest": {
            "derivation": {"source_manifest_sha256": "source"},
            "label_version": "new",
            "split_version": "split",
            "evaluation_version": "v3",
        },
        "labels": labels,
        "splits": tables.splits,
        "evaluation": tables.evaluation.assign(
            evaluation_eligible=tables.evaluation.evaluation_eligible
            & tables.evaluation.case_id.ne("fault")
        ),
        "protocol": protocol,
    }
    updated, lists = rebind(tables, prepared, bundle)
    assert updated.versions["evaluation_version"] == "v3"
    assert lists.fit == original.fit and lists.calibration == original.calibration
    assert lists.main_test == ("c",)
    assert lists.extended_inference == ("c",)
    assert not lists.diagnostic_groups
