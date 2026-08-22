import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.contracts.contract_config import load_contract_config
from src.contracts.contract_v0 import ContractV0Error
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.preprocessors.api_preprocessor import ApiPreprocessor
from src.preprocessors.log_preprocessor import LogPreprocessor
from src.preprocessors.metric_preprocessor import MetricPreprocessor
from src.preprocessors.trace_preprocessor import TracePreprocessor

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(out_dir: Path) -> None:
    # 这些测试验证"开启 fit_endpoint_baseline_stats 时产出 endpoint_id 列与
    # endpoint_baseline_stats.json"。该开关现由 config 声明，v1_expanded_pool.yaml
    # 打开它；v1.yaml 关闭(见 test_v1_without_flag_omits_endpoint_stats)。train_fit
    # 的构成不受 expand_train_pool 影响，故 sidecar 数值重算断言仍成立。
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1_expanded_pool.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_endpoint_id_column_present_and_matches_sorted_mapping(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    expected_mapping = {ep: i for i, ep in enumerate(sorted(ep_to_svc.keys()))}

    for name in ("train_fit", "eval_all"):
        df = pd.read_parquet(out_dir / f"{name}.parquet")
        assert "endpoint_id" in df.columns
        assert df["endpoint_id"].equals(df["endpoint_key"].map(expected_mapping))


def test_endpoint_baseline_stats_sidecar_written_and_loadable(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    sidecar = out_dir / "endpoint_baseline_stats.json"
    assert sidecar.exists()
    stats = EndpointBaselineStats.load(sidecar)

    # sidecar 覆盖的 endpoint 必须是 train_fit 里实际出现过的 endpoint_key 子集
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    fit_endpoints = set(train_fit["endpoint_key"].unique())
    assert stats.fitted_endpoints() == fit_endpoints


def test_endpoint_baseline_stats_values_match_train_fit_only_recomputation(tmp_path):
    """比集合相等更强的回归测试：不仅覆盖的 endpoint KEY 要对，均值/方差 VALUE 本身
    也必须只由 train_fit 决定——不能悄悄混入 Task 6 训练池扩容吸收的 fault_baseline
    行，也不能混入 holdout。直接照抄 EndpointBaselineStats.fit() 里的公式
    （global 统计量 → 退化判定 → shrinkage 混合）在 train_fit.parquet 上重新算一遍，
    与 sidecar 持久化的值逐项比对。若未来 fit() 的调用点被改成传入 train_pool
    （train_fit + fault_baseline）而不是 train_fit，这里会因为均值/方差数值不再
    匹配而失败——单纯比较 fitted_endpoints() 的集合相等不会发现这类回归，因为
    train_pool 里出现的 endpoint_key 集合可能与 train_fit 完全一样。

    mini fixture 的 Normal case 只有 3 个时间窗，fit_frac=0.6 → int(3*0.6)=1 个窗
    进 train_fit，即该 endpoint 在 train_fit 里样本量 n=1，触发 std 的零方差退化
    路径（单样本 std 恒为 NaN）——这正好覆盖了退化分支的公式（退化到 global
    统计量，不走 shrinkage 混合），比挑一个 n>=2 的 endpoint 更能验证兜底链正确。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    ep = "POST:/api/v1/travelservice/trips/left"
    assert ep in set(
        train_fit["endpoint_key"].unique()
    ), "fixture 结构变了，需要更新本测试的目标 endpoint"

    # red_cols/svc_cols 必须锁定 v1_expanded_pool.yaml 声明的特征列，不能扫描
    # dataframe 列名前缀——ApiPreprocessor 现在无条件产出 content_length/body_hash
    # 派生列，即使该 config 未声明它们，train_fit 里也会多出这几列
    # endpoint_red__ 前缀列。按前缀扫描会把这些未声明列纳入重算，与
    # EndpointBaselineStats.fit() 实际使用的列集合（scripts/build_contract.py 同样
    # 改为从 cfg 声明派生，见该文件 _write_v1() 注释）不一致，产出假阴性。
    cfg = load_contract_config(str(REPO_ROOT / "configs/contract/v1_expanded_pool.yaml"))
    red_cols = [f"endpoint_red__{f}" for f in cfg.modalities["endpoint_red"].features]
    svc_cols = [f"service_metric__{f}" for f in cfg.modalities["service_metric"].features] + [
        f"service_log__{f}" for f in cfg.modalities["service_log"].features
    ]

    stats = EndpointBaselineStats.load(out_dir / "endpoint_baseline_stats.json")
    persisted_mean, persisted_std = stats.branch_stats(ep, "ep")

    expected_mean, expected_std = _recompute_branch_stats(train_fit, ep, red_cols)

    assert persisted_mean == pytest.approx(expected_mean, rel=1e-6)
    assert persisted_std == pytest.approx(expected_std, rel=1e-6)

    # svc 分支同理，用同一份重算逻辑再验证一遍，覆盖两个分支各自独立的列集合。
    persisted_svc_mean, persisted_svc_std = stats.branch_stats(ep, "svc")
    expected_svc_mean, expected_svc_std = _recompute_branch_stats(train_fit, ep, svc_cols)
    assert persisted_svc_mean == pytest.approx(expected_svc_mean, rel=1e-6)
    assert persisted_svc_std == pytest.approx(expected_svc_std, rel=1e-6)


def _recompute_branch_stats(
    train_fit: pd.DataFrame, endpoint_key: str, cols: list[str]
) -> tuple[list[float], list[float]]:
    """独立照抄 EndpointBaselineStats.fit() 的公式，只用 pandas 直接在
    train_fit.parquet 上重新计算——不复用被测代码本身，避免测试和实现共用一个
    可能同时出错的路径。公式细节（ddof=1、退化判定阈值、shrinkage 权重）
    必须与 src/data/endpoint_baseline_stats.py 的 fit() 保持一致，否则测试会
    产出假阳性/假阴性。
    """
    shrinkage_k = 10.0
    eps = 1e-9

    g_mean = train_fit[cols].mean()
    g_std = train_fit[cols].std()  # ddof=1，pandas 默认，与 fit() 一致
    g_degenerate = g_std.isna() | (g_std < eps)
    g_std = g_std.where(~g_degenerate, eps)
    g_mean = g_mean.fillna(0.0)

    group = train_fit[train_fit["endpoint_key"] == endpoint_key]
    n = len(group)
    local_mean = group[cols].mean()
    local_std = group[cols].std()
    degenerate = local_std.isna() | (local_std < eps)

    mean = local_mean.copy()
    std = local_std.copy()
    std = std.where(~degenerate, g_std)
    mean = mean.where(~(degenerate & mean.isna()), g_mean)

    w = n / (n + shrinkage_k)
    std = std.where(degenerate, w * std + (1 - w) * g_std)
    mean = mean.where(degenerate, w * mean + (1 - w) * g_mean)

    return mean.tolist(), std.tolist()


def test_process_one_case_raises_when_endpoint_key_missing_from_id_map(tmp_path):
    """endpoint_id_map 缺某个 case 实际用到的 endpoint_key 时（真实场景：数据里出现的
    endpoint 未登记进 configs/contract/endpoint_to_service.yaml），.map() 原本会静默产出
    NaN，一路流到 ReliabilityGatedFusion.gate_weights() 的 int(endpoint_id[i].item())
    才崩，报错信息（"cannot convert float NaN to integer"）完全不指向真正病灶。
    这里直接调 _process_one_case（不经过完整 subprocess 构建，避免为了测这一条
    分支去改共享 fixture 数据），验证现在会在 build 阶段就地报出清晰错误。
    """
    from scripts.build_contract import _process_one_case

    cfg = load_contract_config(str(REPO_ROOT / "configs/contract/v1.yaml"))
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    case_dir = REPO_ROOT / "tests/fixtures/mini_data_root/Lv_P_DISKIO_preserve"

    # 故意漏掉该 case 实际用到的 endpoint_key，模拟"数据里出现但映射表里没有"的场景。
    missing_key = "POST:/api/v1/travelservice/trips/left"
    incomplete_id_map = {
        ep: i for i, ep in enumerate(sorted(ep_to_svc.keys())) if ep != missing_key
    }

    with pytest.raises(ContractV0Error, match=missing_key.replace("/", r"\/")):
        _process_one_case(
            case_dir,
            cfg,
            ep_to_svc,
            TracePreprocessor(),
            ApiPreprocessor(),
            MetricPreprocessor(intermediate_dir=tmp_path / "intermediate"),
            LogPreprocessor(drain3_state_path=tmp_path / "drain3.bin"),
            endpoint_id_map=incomplete_id_map,
        )


def test_endpoint_id_not_added_in_v0(tmp_path):
    """v0 路径不受影响——endpoint_id 是 v1 新增列，不应该悄悄出现在 v0 产物里
    （否则 v0 的既有契约测试的列集合断言会被意外改变行为）。"""
    out_dir = tmp_path / "contract_v0"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "endpoint_id" not in train.columns
    assert not (out_dir / "endpoint_baseline_stats.json").exists()


def test_v1_without_flag_omits_endpoint_stats(tmp_path):
    """v1.yaml 现在 fit_endpoint_baseline_stats=false:不产出 endpoint_id 列，
    也不产出 endpoint_baseline_stats.json。这是把"是否产出 RG 统计量"从
    contract_version=="v1" 解耦成显式开关后的核心不变量——L0/L1/L2 基线
    (走 v1.yaml)不再无谓 fit 一份没人消费的统计量。"""
    out_dir = tmp_path / "contract_v1_noflag"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v1.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    for name in ("train_fit", "eval_all"):
        df = pd.read_parquet(out_dir / f"{name}.parquet")
        assert "endpoint_id" not in df.columns
    assert not (out_dir / "endpoint_baseline_stats.json").exists()
