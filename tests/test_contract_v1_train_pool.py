import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[1]


def _run_v1_build(
    out_dir: Path,
    config: str = "configs/contract/v1_expanded_pool.yaml",
    dataset: str = "tests/fixtures/split_fraction_mini.yaml",
) -> None:
    # 这些测试验证训练池扩容 + baseline 时序切分行为，故默认走 v1_expanded_pool.yaml
    # （expand_train_pool=true）+ split_fraction_mini（故障 case 有 5 个 baseline 窗口，
    # int(5*0.2)=1，能验证非零分数切分）。mini_dataset.yaml 的故障 case 每个只有
    # 1 个 baseline 窗口，int(1*0.2)=0，无法验证分数切分，不能用于这批测试。
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / config),
            "--dataset",
            str(REPO_ROOT / dataset),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_train_pool_includes_fault_baseline_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)

    train = pd.read_parquet(out_dir / "train.parquet")
    # mini fixture 含故障 case（如 Lv_E_SPLITTEST_travel），其 baseline 阶段行
    # 现在应出现在 train.parquet 里（扩容前 train 只含 Normal case）。
    assert (
        train["anomaly_type"] != "Normal"
    ).any(), "train.parquet 未吸收任何故障 case 的 baseline 行，训练池扩容未生效"
    # 吸收的行必须确实是 baseline 阶段，不能混入 inject/recover
    non_normal = train[train["anomaly_type"] != "Normal"]
    assert (non_normal["phase"] == "baseline").all()


def test_train_pool_excludes_inject_and_recover_rows(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert not (train["phase"] == "recover").any()
    assert not (train["phase"] == "inject").any()


def test_source_phase_column_present_and_correct(tmp_path):
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "source_phase" in train.columns
    normal_rows = train[train["anomaly_type"] == "Normal"]
    fault_rows = train[train["anomaly_type"] != "Normal"]
    assert (normal_rows["source_phase"] == "normal_case").all()
    # mini fixture 含故障 case，fault_rows 不应为空——若为空说明训练池扩容未生效，
    # 这里必须硬断言，不能因为 len(fault_rows)==0 而静默跳过下面的校验
    assert len(fault_rows) > 0
    assert (fault_rows["source_phase"] == "fault_baseline").all()


def test_eval_all_and_train_pool_sample_id_disjoint_no_leakage(tmp_path):
    """核心防泄漏不变量（issue #16 修复后仍必须成立且加强）：train_pool 与 eval_all
    的 sample_id 必须互斥。issue #16 前的实现靠"baseline 整段摘出 eval"来保证互斥；
    修复后 baseline 按时间窗时序切分，一部分进 train、其余留 eval，互斥性改由
    split_fault_baseline_temporal 的整窗切分保证（同一窗不会既在 train 又在 eval）。
    这条断言是唯一的硬约束，语义比"eval 不含 baseline 行"更本质。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()


def test_eval_all_retains_majority_of_fault_baseline_rows(tmp_path):
    """issue #16 修复的正向断言：故障 baseline 行不再被整段摘出 eval，大部分（新 fixture
    里 5 窗中的 4 窗 = 80%）应留在 eval_all 维持负样本类别平衡。train 只吸收最早 1 窗。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    train_baseline = train[(train["anomaly_type"] != "Normal") & (train["phase"] == "baseline")]
    eval_baseline = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    train_windows = train_baseline["timestamp_window_ms"].nunique()
    eval_windows = eval_baseline["timestamp_window_ms"].nunique()
    assert train_windows == 1, "fraction=0.2 × 5 窗应吸收最早 1 窗进 train"
    assert eval_windows == 4, "其余 4 窗 baseline 应留在 eval_all"


def test_default_v1_config_does_not_expand_train_pool(tmp_path):
    """configs/contract/v1.yaml 默认 expand_train_pool=false（本次新增开关的
    默认值），train.parquet 必须与 train_fit.parquet 完全一致（不吸收故障
    baseline 行），eval_all 必须含故障 case 的全部阶段（baseline 不摘除）。
    这是与 entry 012 既有实验数字保持可比的行为，必须锁住不被悄悄改回扩容。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config="configs/contract/v1.yaml")

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    # mini fixture 含故障 case，baseline 阶段行必须仍留在 eval_all 里（未被摘除）
    fault_baseline_in_eval = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    assert len(fault_baseline_in_eval) > 0


def test_contract_v1_838_variant_is_pure_normal(tmp_path):
    """2x2 归因实验依赖的手工构造 contract_v1_838：确认 train.parquet 替换为
    train_fit.parquet 后确实是纯 Normal（无 source_phase 列或全为 normal_case），
    防止归因实验的"原始838行"对照组混入扩容数据。"""
    out_dir = tmp_path / "contract_v1"
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
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert (train_fit["anomaly_type"] == "Normal").all()
