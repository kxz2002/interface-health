import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.contract_dataloader import ContractDataset

REPO_ROOT = Path(__file__).parents[1]
MINI_DATA_ROOT = REPO_ROOT / "tests/fixtures/mini_data_root"


@pytest.fixture(scope="module")
def tiny_contract(tmp_path_factory):
    out = tmp_path_factory.mktemp("contract")
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


def test_point_dataset_returns_modality_dict(tiny_contract):
    ds = ContractDataset(
        parquet_path=tiny_contract / "train.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="point",
    )
    sample = ds[0]
    assert set(sample.keys()) >= {
        "endpoint_red",
        "service_metric",
        "service_log",
        "label",
        "meta",
    }
    assert sample["endpoint_red"].shape == (10,)
    assert sample["service_metric"].shape == (5,)
    assert sample["service_log"].shape == (3,)
    assert isinstance(sample["endpoint_red"], torch.Tensor)


def test_point_nan_mean_impute(tiny_contract):
    """NaN 用训练均值填补，输出不再有 NaN。"""
    df = pd.read_parquet(tiny_contract / "train.parquet")
    df.loc[0, "service_log__event_rate"] = float("nan")
    tmp = tiny_contract / "train_nan.parquet"
    df.to_parquet(tmp, index=False)
    ds = ContractDataset(
        parquet_path=tmp,
        schema_path=tiny_contract / "schema.json",
        mode="point",
        nan_strategy="mean",
    )
    sample = ds[0]
    assert not torch.isnan(sample["service_log"]).any()


def test_point_label_carries_phase_and_anomaly(tiny_contract):
    ds = ContractDataset(
        parquet_path=tiny_contract / "eval_all.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="point",
    )
    sample = ds[0]
    assert "phase" in sample["label"]
    assert "is_anomaly" in sample["label"]
