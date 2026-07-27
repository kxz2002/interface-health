from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir

from src.fusion.early_concat import EarlyConcatFusion
from src.models.deep_svdd import DeepSVDD

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_DIR = str(REPO_ROOT / "configs")


def test_fusion_concat_instantiate_returns_early_concat_fusion():
    # compose(config_name="fusion/concat") 走 config-group 加载路径，Hydra 按惯例把结果
    # 挂在跟组名同名的 key 下（cfg.fusion），不是顶层 cfg 本身——这是 Task 3 组合进
    # base.yaml 的 defaults 列表（- fusion: concat）时 cfg.fusion._target_ 能直接生效的前提，
    # 保持配置文件本身不加 @package _global_，两处用法才不冲突。
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/concat")
    fusion = hydra.utils.instantiate(
        cfg.fusion,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
    )
    assert isinstance(fusion, EarlyConcatFusion)
    assert fusion.output_dim == 18


def test_model_deep_svdd_instantiate_returns_deep_svdd():
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="model/deep_svdd")
    svdd = hydra.utils.instantiate(cfg.model, input_dim=18)
    assert isinstance(svdd, DeepSVDD)
    assert svdd.rep_dim == 32  # deep_svdd.yaml 默认值


def test_fusion_output_dim_feeds_svdd_input_dim():
    """fusion.output_dim 必须能直接喂给 svdd 的 input_dim，不需要手动改代码对齐维度。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        fusion_cfg = compose(config_name="fusion/concat")
        model_cfg = compose(config_name="model/deep_svdd")
    fusion = hydra.utils.instantiate(
        fusion_cfg.fusion,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
    )
    svdd = hydra.utils.instantiate(model_cfg.model, input_dim=fusion.output_dim)
    assert svdd.encoder[0].in_features == fusion.output_dim


def test_fusion_reliability_gate_instantiate_returns_reliability_gated_fusion():
    import pandas as pd

    from src.data.endpoint_baseline_stats import EndpointBaselineStats
    from src.fusion.reliability_gate import ReliabilityGatedFusion

    red_cols = [f"endpoint_red__f{i}" for i in range(10)]
    svc_cols = [f"service_metric__g{i}" for i in range(5)] + [
        f"service_log__h{i}" for i in range(3)
    ]
    stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    rows = [{"endpoint_key": "epA", **{c: 1.0 for c in red_cols + svc_cols}} for _ in range(20)]
    stats.fit(pd.DataFrame(rows))

    # compose(config_name="fusion/reliability_gate") 同样走 config-group 加载路径，
    # 结果挂在 cfg.fusion 下（与上面 concat 的注释同一原因），不是顶层 cfg 本身。
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/reliability_gate")
    fusion = hydra.utils.instantiate(
        cfg.fusion,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA"},
    )
    assert isinstance(fusion, ReliabilityGatedFusion)
    assert fusion.output_dim == 16  # yaml 里 branch_dim 默认值


def test_fusion_deviation_weighted_instantiate_returns_deviation_weighted_fusion():
    import pandas as pd

    from src.data.endpoint_baseline_stats import EndpointBaselineStats
    from src.fusion.deviation_weighted import DeviationWeightedFusion

    red_cols = [f"endpoint_red__f{i}" for i in range(10)]
    svc_cols = [f"service_metric__g{i}" for i in range(5)] + [
        f"service_log__h{i}" for i in range(3)
    ]
    stats = EndpointBaselineStats(red_cols=red_cols, svc_cols=svc_cols)
    rows = [{"endpoint_key": "epA", **{c: 1.0 for c in red_cols + svc_cols}} for _ in range(20)]
    stats.fit(pd.DataFrame(rows))

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="fusion/deviation_weighted")
    fusion = hydra.utils.instantiate(
        cfg.fusion,
        modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3},
        endpoint_baseline_stats=stats,
        id_to_endpoint_key={0: "epA"},
        red_cols=red_cols,
        svc_cols=svc_cols,
    )
    assert isinstance(fusion, DeviationWeightedFusion)
    assert fusion.output_dim == 18
