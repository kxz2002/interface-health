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


def test_fault_baseline_train_fraction_boundary_values_valid(tmp_path):
    """0.0 和 1.0 是闭区间的合法边界值，必须能正常加载而不报错。
    校验用的是 <=/>=（闭区间），这条测试防止未来手滑改成 </>（开区间）而不被发现。"""
    for boundary in (0.0, 1.0):
        cfg_path = tmp_path / f"c_{boundary}.yaml"
        cfg_path.write_text(
            "contract_version: v1\n"
            "window_size_s: 15\n"
            f"fault_baseline_train_fraction: {boundary}\n"
            "modalities:\n"
            "  endpoint_red:\n"
            "    preprocessor: TracePreprocessor\n"
            "    preprocessor_version: v0\n"
            "    features: [trace_request_count]\n"
            "    normalization: per_endpoint_min_max\n"
        )
        cfg = load_contract_config(cfg_path)
        assert cfg.fault_baseline_train_fraction == boundary


def test_fault_baseline_train_fraction_quoted_string_raises(tmp_path):
    """YAML 里给数值字段加引号会让 yaml.safe_load 解析成 str 而非 float，
    此时范围比较（float <= str）会抛无关的 TypeError；必须在类型检查处先堵住，
    抛出可读的 ValueError 而不是让调用方看到一个不知所云的 TypeError。"""
    cfg_path = tmp_path / "c_quoted.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        'fault_baseline_train_fraction: "0.5"\n'
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    with pytest.raises(ValueError, match="fault_baseline_train_fraction"):
        load_contract_config(cfg_path)


def test_nontarget_fractions_default_to_zero(tmp_path):
    """两个新字段默认 0.0：向后兼容——config 不写时行为与本次改动前完全一致
    （expand_train_pool=true 也只吸收 baseline，不吸收 inject/recover 非目标行）。"""
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
    assert cfg.fault_inject_nontarget_train_fraction == 0.0
    assert cfg.fault_recover_nontarget_train_fraction == 0.0


def test_nontarget_fractions_override(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fault_inject_nontarget_train_fraction: 0.75\n"
        "fault_recover_nontarget_train_fraction: 0.4\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_inject_nontarget_train_fraction == 0.75
    assert cfg.fault_recover_nontarget_train_fraction == 0.4


def test_nontarget_fractions_out_of_range_raises(tmp_path):
    """越界一律在加载期报错，不看 expand_train_pool——理由同
    fault_baseline_train_fraction：负数会触发 Python 负索引切片导致方向反转的
    错误切分，大于 1 会让 train 吞下全部窗口且不报错，两者都是"看起来合理但
    错误的数据"，必须在入口堵住。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        for bad in (1.5, -0.1):
            cfg_path = tmp_path / f"c_{field}_{bad}.yaml"
            cfg_path.write_text(
                "contract_version: v1\n"
                "window_size_s: 15\n"
                f"{field}: {bad}\n"
                "modalities:\n"
                "  endpoint_red:\n"
                "    preprocessor: TracePreprocessor\n"
                "    preprocessor_version: v0\n"
                "    features: [trace_request_count]\n"
                "    normalization: per_endpoint_min_max\n"
            )
            with pytest.raises(ValueError, match=field):
                load_contract_config(cfg_path)


def test_nontarget_fractions_boundary_values_valid(tmp_path):
    """0.0 与 1.0 是合法边界值，不能被范围校验误拒。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        for boundary in (0.0, 1.0):
            cfg_path = tmp_path / f"c_{field}_{boundary}.yaml"
            cfg_path.write_text(
                "contract_version: v1\n"
                "window_size_s: 15\n"
                f"{field}: {boundary}\n"
                "modalities:\n"
                "  endpoint_red:\n"
                "    preprocessor: TracePreprocessor\n"
                "    preprocessor_version: v0\n"
                "    features: [trace_request_count]\n"
                "    normalization: per_endpoint_min_max\n"
            )
            cfg = load_contract_config(cfg_path)
            assert getattr(cfg, field) == boundary


def test_nontarget_fractions_quoted_string_raises(tmp_path):
    """YAML 里误加引号（字符串类型）必须报出指名字段的错误，而不是在比较运算处
    抛与本意无关的 TypeError——类型检查须先于范围检查。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        cfg_path = tmp_path / f"c_{field}_str.yaml"
        cfg_path.write_text(
            "contract_version: v1\n"
            "window_size_s: 15\n"
            f'{field}: "0.5"\n'
            "modalities:\n"
            "  endpoint_red:\n"
            "    preprocessor: TracePreprocessor\n"
            "    preprocessor_version: v0\n"
            "    features: [trace_request_count]\n"
            "    normalization: per_endpoint_min_max\n"
        )
        with pytest.raises(ValueError, match=field):
            load_contract_config(cfg_path)
