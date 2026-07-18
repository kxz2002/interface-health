#!/usr/bin/env python
"""按 anomaly_type 分层统计 ReliabilityGatedFusion 学到的 w_ep/w_svc 均值，
用于 spec §4.4 可解释性证据（HTTP 类故障预期 w_ep 高，资源类故障预期 w_svc 高）。
一次性分析脚本，不是训练/评估管线的一部分。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from src.data.contract_dataloader import ContractDataset
from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER
from src.fusion.reliability_gate import ReliabilityGatedFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-dir", required=True)
    parser.add_argument("--checkpoint", required=False, help="若训练脚本保存了权重则传入")
    args = parser.parse_args()

    contract_dir = Path(args.contract_dir)
    schema = json.loads((contract_dir / "schema.json").read_text())
    modality_dims = {n: len(g["columns"]) for n, g in schema["feature_groups"].items()}
    baseline_stats = EndpointBaselineStats.load(contract_dir / "endpoint_baseline_stats.json")
    ep_to_svc = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    id_to_key = {i: ep for i, ep in enumerate(sorted(ep_to_svc.keys()))}

    fusion = ReliabilityGatedFusion(
        modality_dims=modality_dims,
        endpoint_baseline_stats=baseline_stats,
        id_to_endpoint_key=id_to_key,
        branch_dim=16,
    )
    if args.checkpoint:
        fusion.load_state_dict(torch.load(args.checkpoint))
    fusion.eval()

    ds = ContractDataset(
        parquet_path=contract_dir / "eval_all.parquet",
        schema_path=contract_dir / "schema.json",
        nan_strategy="mean",
        fit_on_parquet=contract_dir / "train.parquet",
    )
    eval_df = pd.read_parquet(contract_dir / "eval_all.parquet")

    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            batch = {m: sample[m].unsqueeze(0) for m in MODALITY_ORDER}
            eid = torch.tensor([sample["meta"]["endpoint_id"]])
            w_ep, w_svc = fusion.gate_weights(batch, eid)
            records.append(
                {
                    "sample_id": sample["meta"]["sample_id"],
                    "w_ep": w_ep.item(),
                    "w_svc": w_svc.item(),
                }
            )

    weights_df = pd.DataFrame(records).merge(eval_df[["sample_id", "anomaly_type"]], on="sample_id")
    summary = weights_df.groupby("anomaly_type")[["w_ep", "w_svc"]].mean()
    print(summary.to_string())


if __name__ == "__main__":
    main()
