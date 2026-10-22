from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.service.modeling.numeric import fit_numeric, transform_numeric


def fixture_tables():
    identity = pd.DataFrame(
        {"sample_id": list("abcdef"), "canonical_service_id": ["A"] * 2 + ["B"] * 2 + ["A"] * 2}
    )
    config = {
        "features": {
            "resource": ["cpu_cores", "memory_usage_ratio", "process_count"],
            "execution": [
                "observation_count",
                "duration_mean_ms",
                "duration_p95_ms",
                "duration_max_ms",
                "error_known_count",
                "error_count",
                "error_coverage",
            ],
        },
        "preprocessing": {
            "log1p": [
                "process_count",
                "observation_count",
                "duration_mean_ms",
                "duration_p95_ms",
                "duration_max_ms",
            ],
            "scale_epsilon": 1e-6,
        },
    }
    resource = identity.assign(
        cpu_cores=[5, 5, 3, 7, 1e9, 1e9],
        memory_usage_ratio=0.5,
        process_count=np.nan,
        resource_valid=True,
        quality_status="RESOURCE_VALID",
    )
    execution = identity.assign(
        observation_count=2.0,
        duration_mean_ms=1.0,
        duration_p95_ms=1.0,
        duration_max_ms=1.0,
        error_known_count=2.0,
        error_count=1.0,
        error_coverage=1.0,
        quality_status="OBSERVED",
    )
    return SimpleNamespace(resource=resource, execution=execution, config=config)


def test_fit_only_and_three_level_scale():
    tables = fixture_tables()
    state = fit_numeric(tables, set("abcd"), tables.config)
    assert state.fields["resource.cpu_cores"]["services"]["A"]["mean"] == 5
    assert state.fields["resource.cpu_cores"]["services"]["A"]["scale"] == pytest.approx(np.sqrt(2))
    assert state.fields["resource.memory_usage_ratio"]["services"]["A"]["scale"] == 1
    assert not state.fields["resource.process_count"]["available"]
    tables.resource.loc[4:, "cpu_cores"] = -1e9
    assert fit_numeric(tables, set("abcd"), tables.config).state_hash == state.state_hash
    tables.resource["cpu_cores"] = 5
    assert (
        fit_numeric(tables, set("abcd"), tables.config).fields["resource.cpu_cores"]["services"][
            "A"
        ]["scale"]
        == 1
    )


def test_field_quality_and_execution_error():
    tables = fixture_tables()
    tables.resource.loc[4, "resource_valid"] = False
    tables.resource["resource_valid"] = tables.resource.resource_valid.astype("boolean")
    tables.resource.loc[5, "resource_valid"] = None
    tables.execution.loc[4, ["error_known_count", "error_coverage"]] = [1, 0.5]
    tables.execution.loc[4, "quality_status"] = "OBSERVED_ERROR_PARTIAL"
    tables.execution.loc[5, "observation_count"] = 0
    result = transform_numeric(tables, fit_numeric(tables, set("abcd"), tables.config))
    masks = result["masks"].set_index("sample_id")
    assert (
        not masks.loc[list("ef"), ["cpu_cores", "memory_usage_ratio", "process_count"]].any().any()
    )
    assert masks.at["e", "duration_mean_ms"] and not masks.at["e", "error_rate"]
    assert not masks.loc[
        "f",
        [
            "observation_count",
            "duration_mean_ms",
            "duration_p95_ms",
            "duration_max_ms",
            "error_rate",
        ],
    ].any()
    assert not masks.process_count.any()


def test_invalid_domains_and_fit_leakage():
    tables = fixture_tables()
    tables.resource.loc[4, "cpu_cores"] = -1
    tables.execution.loc[4, "duration_mean_ms"] = -2
    result = transform_numeric(tables, fit_numeric(tables, set("abcd"), tables.config))
    assert not result["masks"].set_index("sample_id").at["e", "cpu_cores"]
    assert not result["masks"].set_index("sample_id").at["e", "duration_mean_ms"]


def test_service_missing_fit_field_stays_unavailable():
    tables = fixture_tables()
    tables.resource.loc[tables.resource.canonical_service_id.eq("A"), "memory_usage_ratio"] = np.nan
    state = fit_numeric(tables, set("abcd"), tables.config)
    stats = state.fields["resource.memory_usage_ratio"]["services"]["A"]
    assert stats["mean"] is None
    assert not stats["available"]
    result = transform_numeric(tables, state)
    row = result["masks"].set_index("sample_id").loc["a"]
    assert not row.memory_usage_ratio
    assert result["values"].set_index("sample_id").at["a", "memory_usage_ratio"] == 0.0
