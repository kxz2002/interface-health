"""锁定 dvc.yaml 里 v1 契约两条链路（未扩容 vs RG 扩容）的 --config 绑定。

expand_train_pool 开关（history/entries/014）的核心保证是 build_contract_v1
（L0/L1/L2 对比基线用）与 build_contract_v1_expanded（RG 专属）分别指向
configs/contract/v1.yaml（expand_train_pool=false）与
configs/contract/v1_expanded_pool.yaml（expand_train_pool=true）。这个绑定
只存在于 dvc.yaml 里，tests/test_contract_v1_train_pool.py 等测试直接调
build_contract.py --config，完全绕开 dvc.yaml，不会捕获"两个 stage 的
--config 被改反"这类回归——这正是本开关要防的那类问题，因此单独锁定。
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[1]


def _stage_config_path(stage: dict) -> str:
    cmd = stage["cmd"]
    tokens = cmd.split()
    return tokens[tokens.index("--config") + 1]


def test_build_contract_v1_uses_non_expanded_config():
    """build_contract_v1（L0/L1/L2 对比基线用）必须指向 v1.yaml，
    且该 config 的 expand_train_pool 必须为 False——否则 L0/L1/L2 的
    eval 集合会被 RG 的训练池扩容逻辑悄悄污染，复现本开关修复前的问题。
    """
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stage = pipeline["stages"]["build_contract_v1"]

    config_path = _stage_config_path(stage)
    assert config_path == "configs/contract/v1.yaml"
    assert config_path in stage.get("deps", [])

    cfg = yaml.safe_load((REPO_ROOT / config_path).read_text())
    assert cfg.get("expand_train_pool", False) is False


def test_build_contract_v1_expanded_uses_expanded_config():
    """build_contract_v1_expanded(RG 专属)已隔离到 dvc_reliability_gate/dvc.yaml,
    必须指向 v1_expanded_pool.yaml 且 expand_train_pool=true。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc_reliability_gate/dvc.yaml").read_text())
    stage = pipeline["stages"]["build_contract_v1_expanded"]

    config_path = _stage_config_path(stage)
    assert config_path == "configs/contract/v1_expanded_pool.yaml"
    assert config_path in stage.get("deps", [])

    cfg = yaml.safe_load((REPO_ROOT / config_path).read_text())
    assert cfg.get("expand_train_pool", False) is True
    assert cfg.get("fit_endpoint_baseline_stats", False) is True


def test_train_v1_reliability_gate_reads_expanded_contract_dir():
    """train_v1_reliability_gate 必须消费 build_contract_v1_expanded 的产物
    （contract_v1_expanded/），不能悄悄指回 contract_v1/（未扩容），
    否则 RG 会训练在与 L0/L1/L2 相同的小样本上，失去扩容训练池的意义。
    """
    pipeline = yaml.safe_load((REPO_ROOT / "dvc_reliability_gate/dvc.yaml").read_text())
    stage = pipeline["stages"]["train_v1_reliability_gate"]

    cmd = stage["cmd"]
    assert "contract_dir=artifacts/contract_v1_expanded" in cmd

    deps = stage.get("deps", [])
    assert any(d.startswith("artifacts/contract_v1_expanded/") for d in deps)
    assert not any(
        d.startswith("artifacts/contract_v1/") for d in deps
    ), "train_v1_reliability_gate 的 deps 不应指向未扩容的 contract_v1/"


def test_v1_and_v1_expanded_configs_share_identical_modalities():
    """两份 v1 系 config 除 expand_train_pool 外必须共享同一份 modalities 定义，
    否则 L0/L1/L2 与 RG 会在 feature_dim/schema 上悄悄分叉，破坏两条链路的可比性。
    该断言没有 YAML include 机制兜底，纯靠测试锁住同步。
    """
    v1 = yaml.safe_load((REPO_ROOT / "configs/contract/v1.yaml").read_text())
    v1_expanded = yaml.safe_load((REPO_ROOT / "configs/contract/v1_expanded_pool.yaml").read_text())

    assert v1["modalities"] == v1_expanded["modalities"]
    assert v1["window_size_s"] == v1_expanded["window_size_s"]
    assert v1["contract_version"] == v1_expanded["contract_version"]
    assert v1["expand_train_pool"] is False
    assert v1_expanded["expand_train_pool"] is True


def test_root_dvc_excludes_rg_stages_and_stale_sidecar_output():
    """3 个 RG 专属 stage 必须从根 dvc.yaml 移除(裸 dvc repro 不再触发它们);
    build_contract_v1 的 outs 不再声明 endpoint_baseline_stats.json
    (v1.yaml flag=false 后不再产出该文件)。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stages = pipeline["stages"]
    for rg_stage in (
        "build_contract_v1_expanded",
        "train_v1_reliability_gate",
        "eval_v1_reliability_gate",
    ):
        assert rg_stage not in stages
    outs = stages["build_contract_v1"].get("outs", [])
    assert "artifacts/contract_v1/endpoint_baseline_stats.json" not in outs
