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


def test_per_case_zscore_normalization_accepted():
    spec = ModalitySpec(
        preprocessor="X",
        preprocessor_version="1",
        features=("a",),
        normalization="per_case_endpoint_z_score",
    )
    assert spec.normalization == "per_case_endpoint_z_score"


def test_unknown_normalization_still_rejected():
    with pytest.raises(ValueError, match="未知 normalization"):
        ModalitySpec(
            preprocessor="X",
            preprocessor_version="1",
            features=("a",),
            normalization="per_case_endpoint_minmax_typo",
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


def _write_yaml(tmp_path, body: str, name: str = "c.yaml") -> Path:
    cfg_path = tmp_path / name
    cfg_path.write_text(body)
    return cfg_path


# 单个 modality 的最小 YAML 片段，方便按版本/normalization 拼装守卫测试。
_MODALITY_BLOCK = {
    "endpoint_red": (
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_error_rate]\n"
        "    normalization: {norm}\n"
    ),
    "service_metric": (
        "  service_metric:\n"
        "    preprocessor: MetricPreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [cpu_usage_rate]\n"
        "    normalization: {norm}\n"
    ),
}


def test_v2_config_requires_per_case_z_score_normalization(tmp_path):
    """v2×min_max 组合守卫：v2 build 路径（is_v2=True）静默跳过 rate clip 与退化
    group 告警，min_max 退化组的原始量纲会无告警进模型。错误组合必须在配置
    加载期就拒绝，而不是等到 build 跑完或产物静默错误。"""
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["endpoint_red"].format(norm="per_endpoint_min_max"),
    )
    with pytest.raises(ValueError, match="v2.*per-case z-score"):
        load_contract_config(cfg_path)


def test_v2_config_rejects_global_min_max(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["service_metric"].format(norm="global_min_max"),
    )
    with pytest.raises(ValueError, match="service_metric"):
        load_contract_config(cfg_path)


def test_v2_config_rejects_even_one_non_zscore_modality(tmp_path):
    """守卫按 modality 逐个检查：不能因多数 modality 正确就放行单个错误组合，
    且报错必须点名违规 modality 与其实际 normalization。"""
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n"
        + _MODALITY_BLOCK["endpoint_red"].format(norm="per_case_endpoint_z_score")
        + _MODALITY_BLOCK["service_metric"].format(norm="per_service_min_max"),
    )
    with pytest.raises(ValueError, match=r"service_metric.*per_service_min_max"):
        load_contract_config(cfg_path)


def test_v2_config_with_per_case_z_score_loads(tmp_path):
    """合法的 v2 组合（全部 modality 走 per-case z-score）必须正常加载。"""
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n"
        + _MODALITY_BLOCK["endpoint_red"].format(norm="per_case_endpoint_z_score")
        + _MODALITY_BLOCK["service_metric"].format(norm="per_case_service_z_score"),
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.contract_version == "v2"


def test_v1_config_with_min_max_still_loads(tmp_path):
    """守卫只约束 v2：v1/v0 的 min_max 是既有唯一合法形态，不能被新守卫误伤。"""
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["endpoint_red"].format(norm="per_endpoint_min_max"),
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.contract_version == "v1"


def test_v1_config_with_per_case_z_score_rejected_at_load(tmp_path):
    """对称守卫：per-case z-score 是 v2 专属，v1 配它必须在加载期拒绝。

    实测（v2_fit_scope_mini fixture + v1 config 跑 build_contract.py）：该组合
    并非"静默跑通后被 clip 截断"，而是在归一化阶段抛裸 KeyError: 'case_id'——
    per-case scope 的分组键含 case_id（normalization.py 的 _GROUP_COLS），而 v1
    路径传给 Normalizer.transform 的 group_cols 不含该列。裸 KeyError 报错质量
    差，守卫把它换成带说明的 ValueError。
    """
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["endpoint_red"].format(norm="per_case_endpoint_z_score"),
    )
    with pytest.raises(ValueError, match="v2"):
        load_contract_config(cfg_path)


def test_v0_config_with_per_case_z_score_rejected_at_load(tmp_path):
    """v0 同样不接受 per-case z-score，且报错必须点出 case_id/KeyError 根因。"""
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v0\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["service_metric"].format(norm="per_case_service_z_score"),
    )
    with pytest.raises(ValueError, match="case_id"):
        load_contract_config(cfg_path)


def test_v2_endpoint_red_requires_endpoint_scope(tmp_path):
    """modality↔scope 配对：endpoint_red 必须配 per_case_endpoint_z_score。

    错配成 service 版会静默按 (case_id, service_name) 聚合——当前数据集
    service↔endpoint 1:1 数字不变，但未来一个 service 多 endpoint 的数据集上
    会把不同 endpoint 的窗混在一起算 z 值，且没有任何报错。
    """
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n" + _MODALITY_BLOCK["endpoint_red"].format(norm="per_case_service_z_score"),
    )
    with pytest.raises(ValueError, match=r"endpoint_red.*per_case_endpoint_z_score"):
        load_contract_config(cfg_path)


def test_v2_service_metric_requires_service_scope(tmp_path):
    cfg_path = _write_yaml(
        tmp_path,
        "contract_version: v2\n"
        "window_size_s: 15\n"
        "modalities:\n"
        + _MODALITY_BLOCK["service_metric"].format(norm="per_case_endpoint_z_score"),
    )
    with pytest.raises(ValueError, match=r"service_metric.*per_case_service_z_score"):
        load_contract_config(cfg_path)


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
