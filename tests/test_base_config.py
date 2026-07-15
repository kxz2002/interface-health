from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf.errors import MissingMandatoryValue

REPO_ROOT = Path(__file__).parents[1]
CONFIGS_DIR = str(REPO_ROOT / "configs")


def test_base_config_composes_with_fusion_and_model_defaults():
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="base", overrides=["contract_dir=/tmp/c", "out=/tmp/o.parquet"])
    assert cfg.fusion._target_ == "src.fusion.early_concat.EarlyConcatFusion"
    assert cfg.model._target_ == "src.models.deep_svdd.DeepSVDD"
    assert cfg.contract_dir == "/tmp/c"
    assert cfg.out == "/tmp/o.parquet"


def test_base_config_missing_contract_dir_raises():
    """contract_dir/out 未通过 CLI override 填充时，访问该字段必须报错，而不是静默返回 None。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="base")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.contract_dir


def test_base_config_fusion_switchable_via_override():
    """本轮只新增 concat，但必须验证 config-group 切换机制本身工作——
    这是消融实验用 fusion=xxx 切换机制的前提。"""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="base",
            overrides=["contract_dir=/tmp/c", "out=/tmp/o.parquet", "fusion=concat"],
        )
    assert cfg.fusion._target_ == "src.fusion.early_concat.EarlyConcatFusion"
