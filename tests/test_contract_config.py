from pathlib import Path

import pytest

from src.contracts.contract_config import ModalitySpec, load_contract_config

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


def test_empty_features_raises():
    """features 为空时 ModalitySpec 应抛 ValueError。"""
    with pytest.raises(ValueError, match="features"):
        ModalitySpec(
            preprocessor="TracePreprocessor",
            preprocessor_version="v0",
            features=(),
            normalization="per_endpoint_min_max",
        )


def test_invalid_normalization_raises():
    """不支持的 normalization 字符串应抛 ValueError。"""
    with pytest.raises(ValueError, match="normalization"):
        ModalitySpec(
            preprocessor="TracePreprocessor",
            preprocessor_version="v0",
            features=("feat_a",),
            normalization="unknown_norm",
        )


def test_fit_endpoint_baseline_stats_defaults_false(tmp_path):
    """新增开关默认 False(与 expand_train_pool 同款模式):config 不写该字段时,
    build 不产出 endpoint_id 列与 endpoint_baseline_stats.json。"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fit_endpoint_baseline_stats is False


def test_fit_endpoint_baseline_stats_override_true(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fit_endpoint_baseline_stats: true\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fit_endpoint_baseline_stats is True


def test_fault_baseline_train_fraction_defaults_to_0_2(tmp_path):
    """新增开关默认 0.2：config 不写该字段时取默认值。默认值只在
    expand_train_pool=true 时被 _write_v1 消费，但默认值本身必须稳定。"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_baseline_train_fraction == 0.2


def test_fault_baseline_train_fraction_override(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fault_baseline_train_fraction: 0.35\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_baseline_train_fraction == 0.35


def test_fault_baseline_train_fraction_out_of_range_raises(tmp_path):
    """越界值（<0 或 >1）必须在加载时报错，而不是悄悄产出空/全量切分污染实验。"""
    for bad in ("2.0", "-0.1"):
        cfg_path = tmp_path / f"c_{bad}.yaml"
        cfg_path.write_text(
            "contract_version: v1\n"
            "window_size_s: 15\n"
            f"fault_baseline_train_fraction: {bad}\n"
            "modalities:\n"
            "  endpoint_red:\n"
            "    preprocessor: TracePreprocessor\n"
            "    preprocessor_version: v0\n"
            "    features: [trace_request_count]\n"
            "    normalization: per_endpoint_min_max\n"
        )
        with pytest.raises(ValueError, match="fault_baseline_train_fraction"):
            load_contract_config(cfg_path)
