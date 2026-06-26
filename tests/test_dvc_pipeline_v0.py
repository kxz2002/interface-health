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
