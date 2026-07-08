from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[1]


def test_dvc_yaml_has_v0_stages():
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stages = set(pipeline["stages"].keys())
    assert {"build_contract", "train_v0", "eval_v0"}.issubset(stages)


def test_dvc_v0_stages_have_required_fields():
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    for stage_name in ("build_contract", "train_v0", "eval_v0"):
        stage = pipeline["stages"][stage_name]
        assert "cmd" in stage
        assert "deps" in stage


def test_build_contract_uses_dataset_config():
    """build_contract stage 必须用 --dataset 指向真实存在的 config，
    且该 config 列入 deps（防"config 改了但没进 deps 导致 dvc 不重跑"的隐蔽坑）。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stage = pipeline["stages"]["build_contract"]

    cmd = stage["cmd"]
    assert "--dataset" in cmd
    assert "--data-root" not in cmd

    # 从 cmd 抽出 --dataset 后面的路径，断言文件存在
    tokens = cmd.split()
    ds_path = tokens[tokens.index("--dataset") + 1]
    assert (REPO_ROOT / ds_path).exists(), f"{ds_path} 不存在"

    # 该 config 必须在 deps 里
    deps = stage.get("deps", [])
    assert ds_path in deps, f"{ds_path} 未列入 build_contract 的 deps"
