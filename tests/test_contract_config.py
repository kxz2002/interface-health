from pathlib import Path

import pytest

from src.contracts.contract_config import load_contract_config

REPO_ROOT = Path(__file__).parents[1]


def test_load_v0_config_has_three_modalities():
    cfg = load_contract_config(REPO_ROOT / "configs/contract/v0.yaml")
    assert set(cfg.modalities.keys()) == {"endpoint_red", "service_metric", "service_log"}


def test_v0_feature_dim_is_18():
    cfg = load_contract_config(REPO_ROOT / "configs/contract/v0.yaml")
    total = sum(len(m.features) for m in cfg.modalities.values())
    assert total == 18, f"feature_dim 必须是 18，当前 {total}"


def test_v0_endpoint_red_has_10_features():
    cfg = load_contract_config(REPO_ROOT / "configs/contract/v0.yaml")
    assert len(cfg.modalities["endpoint_red"].features) == 10


def test_normalization_scope_valid():
    cfg = load_contract_config(REPO_ROOT / "configs/contract/v0.yaml")
    valid_scopes = {"per_endpoint_min_max", "per_service_min_max", "global_min_max"}
    for m in cfg.modalities.values():
        assert m.normalization in valid_scopes


def test_missing_modality_field_raises():
    with pytest.raises(ValueError, match="modalities"):
        load_contract_config(REPO_ROOT / "tests/fixtures/bad_contract_no_modalities.yaml")
