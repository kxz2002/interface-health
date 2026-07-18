import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from scripts.train_baseline_v0 import _collate
from src.contracts import validate_scores_df
from src.fusion.base import MODALITY_ORDER

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


def _fake_sample(sample_id: str, endpoint_id: int | None) -> dict:
    sample = {m: torch.zeros(2) for m in MODALITY_ORDER}
    sample["label"] = {"phase": "normal", "is_anomaly": False}
    sample["meta"] = {"sample_id": sample_id, "endpoint_key": "epA"}
    if endpoint_id is not None:
        sample["meta"]["endpoint_id"] = endpoint_id
    return sample


def test_collate_stacks_endpoint_id_when_first_sample_has_it():
    batch = [_fake_sample("s1", endpoint_id=0), _fake_sample("s2", endpoint_id=1)]
    out = _collate(batch)
    assert "endpoint_id" in out
    assert out["endpoint_id"].tolist() == [0, 1]


def test_collate_omits_endpoint_id_when_first_sample_lacks_it():
    """v0 数据没有 endpoint_id 列，_row_to_sample 不产出该 key；_collate 只看第一个
    样本判断整批（同一 parquet 内列集合一致，v0/v1 不混跑），输出里不应出现这个 key，
    下游 fusion 调用走 endpoint_id=None 的默认参数路径。"""
    batch = [_fake_sample("s1", endpoint_id=None), _fake_sample("s2", endpoint_id=None)]
    out = _collate(batch)
    assert "endpoint_id" not in out


def _build_contract(contract_dir: Path, config: str = "configs/contract/v0.yaml") -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / config),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/mini_dataset.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _train(contract_dir: Path, out: Path, seed: int = 42, epochs: int = 2) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={out}",
            f"seed={seed}",
            f"training.epochs={epochs}",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_train_baseline_v0_writes_scores_contract(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    validate_scores_df(df)
    assert "case_id" in df.columns
    assert "anomaly_type" in df.columns
    # 行数应与 eval_all 一致（每个评估样本输出一个 score）
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)


def test_train_baseline_v0_is_reproducible(tmp_path):
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out1 = tmp_path / "scores1.parquet"
    out2 = tmp_path / "scores2.parquet"
    for out in (out1, out2):
        _train(contract_dir, out)

    df1 = pd.read_parquet(out1).sort_values("sample_id").reset_index(drop=True)
    df2 = pd.read_parquet(out2).sort_values("sample_id").reset_index(drop=True)
    pd.testing.assert_series_equal(df1["score"], df2["score"])


def test_train_baseline_v0_scores_carry_endpoint_label_columns(tmp_path):
    """scores.parquet 必须显式带上 is_endpoint_anomaly / label_granularity，
    否则四层 eval 分层无法计算——这是历史上 is_target_endpoint 被静默丢弃的同一种坑。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert "is_endpoint_anomaly" in df.columns
    assert "label_granularity" in df.columns

    merged = df.merge(
        eval_all[["sample_id", "is_endpoint_anomaly", "label_granularity"]],
        on="sample_id",
        suffixes=("_out", "_src"),
    )
    assert (
        merged["is_endpoint_anomaly_out"].astype(bool)
        == merged["is_endpoint_anomaly_src"].astype(bool)
    ).all()
    assert (merged["label_granularity_out"] == merged["label_granularity_src"]).all()


def test_train_baseline_v0_works_end_to_end_on_v1_contract(tmp_path):
    """v1 训练路径此前无任何端到端覆盖——v1 专属问题（如 train_fit 太小、
    eval_all schema 漂移）此前会绿着上线。复用同一份训练脚本对 v1 contract 跑一遍。"""
    contract_dir = tmp_path / "contract_v1"
    _build_contract(contract_dir, config="configs/contract/v1.yaml")

    out = tmp_path / "scores_v1.parquet"
    _train(contract_dir, out)

    df = pd.read_parquet(out)
    validate_scores_df(df)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert len(df) == len(eval_all)


def test_train_baseline_v0_does_not_save_checkpoint_by_default(tmp_path):
    """fusion_checkpoint 未指定时保持默认行为：不写任何权重文件。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    _train(contract_dir, out)

    assert list(tmp_path.glob("*.pt")) == []


def test_train_baseline_v0_saves_fusion_checkpoint_when_requested(tmp_path):
    """analyze_gate_weights.py（Task 9）需要真实训练出的 fusion 权重而非随机初始化——
    训练脚本需支持可选的 fusion_checkpoint override 落盘 state_dict。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    checkpoint = tmp_path / "fusion.pt"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            f"contract_dir={contract_dir}",
            f"out={out}",
            "seed=42",
            "training.epochs=2",
            f"fusion_checkpoint={checkpoint}",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    assert checkpoint.exists()
    state_dict = torch.load(checkpoint)
    assert isinstance(state_dict, dict) and len(state_dict) == 0  # concat 融合无可训练参数
