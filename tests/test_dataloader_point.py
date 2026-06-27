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


def test_invalid_nan_strategy_raises(tiny_contract):
    """未知 nan_strategy 应立即抛 ValueError。"""
    with pytest.raises(ValueError, match="nan_strategy"):
        ContractDataset(
            parquet_path=tiny_contract / "train.parquet",
            schema_path=tiny_contract / "schema.json",
            nan_strategy="drop",
        )


def test_invalid_sequence_length_raises(tiny_contract):
    """sequence_length < 1 应立即抛 ValueError。"""
    with pytest.raises(ValueError, match="sequence_length"):
        ContractDataset(
            parquet_path=tiny_contract / "train.parquet",
            schema_path=tiny_contract / "schema.json",
            sequence_length=0,
        )


def test_all_nan_mean_fill_falls_back_to_zero(tiny_contract, caplog):
    """fit 源数据某特征列全 NaN 时，nan_strategy='mean' 应回退到 0 填充并发出 warning。"""
    import logging

    df = pd.read_parquet(tiny_contract / "train.parquet")
    df["service_log__event_rate"] = float("nan")
    tmp = tiny_contract / "train_all_nan.parquet"
    df.to_parquet(tmp, index=False)
    with caplog.at_level(logging.WARNING, logger="src.data.contract_dataloader"):
        ds = ContractDataset(
            parquet_path=tmp,
            schema_path=tiny_contract / "schema.json",
            nan_strategy="mean",
        )
    assert "service_log__event_rate" in caplog.text
    # 全 NaN 列应被 0 填充，不影响 dataset 正常使用
    assert len(ds) > 0
    sample = ds[0]
    svc_log_tensor = sample["service_log"]
    # service_log__event_rate 是 service_log modality 的第一个特征
    assert svc_log_tensor[0].item() == pytest.approx(0.0)
