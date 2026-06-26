import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.data.contract_dataloader import ContractDataset

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


@pytest.fixture(scope="module")
def tiny_contract(tmp_path_factory):
    out = tmp_path_factory.mktemp("contract_seq")
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--data-root",
            str(MINI_DATA_ROOT),
            "--out-dir",
            str(out),
            "--seed",
            "1",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return out


def test_sequence_dataset_shape(tiny_contract):
    """sequence 模式每个 modality 返回 (seq_len, D) tensor。"""
    ds = ContractDataset(
        parquet_path=tiny_contract / "train.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="sequence",
        sequence_length=4,
    )
    if len(ds) == 0:
        pytest.skip("mini data too short for sequence_length=4")
    sample = ds[0]
    assert sample["endpoint_red"].shape == (4, 10)
    assert sample["service_metric"].shape == (4, 5)
    assert sample["service_log"].shape == (4, 3)


def test_sequence_no_cross_endpoint_boundary(tiny_contract):
    """序列窗口不跨 endpoint 边界。"""
    ds = ContractDataset(
        parquet_path=tiny_contract / "train.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="sequence",
        sequence_length=4,
    )
    for i in range(len(ds)):
        sample = ds[i]
        eps = sample["meta"]["endpoint_key_per_step"]
        assert len(set(eps)) == 1, f"序列跨了 endpoint 边界: {eps}"


def test_nan_mean_fit_on_separate_parquet(tiny_contract):
    """用 fit_on_parquet 注入外部均值，避免 eval 路径信息泄漏。"""
    import pandas as pd

    df = pd.read_parquet(tiny_contract / "eval_all.parquet")
    df.loc[df.index[0], "service_log__event_rate"] = float("nan")
    tmp = tiny_contract / "eval_nan.parquet"
    df.to_parquet(tmp, index=False)
    ds = ContractDataset(
        parquet_path=tmp,
        schema_path=tiny_contract / "schema.json",
        mode="point",
        nan_strategy="mean",
        fit_on_parquet=tiny_contract / "train.parquet",
    )
    sample = ds[0]
    assert not torch.isnan(sample["service_log"]).any()
