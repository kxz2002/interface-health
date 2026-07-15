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
