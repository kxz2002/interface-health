"""Score an existing TT shared prior-GRU checkpoint on SN as a diagnostic transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.service.modeling.availability import fit_availability
from src.service.modeling.sn_adapter import build_sn_prepared
from src.service.modeling.temporal import TemporalPredictor, score_temporal, temporal_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = build_sn_prepared(args.prepared_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = TemporalPredictor(input_dim=28, hidden_dim=int(config["training"]["hidden_dim"]))
    model.load_state_dict(checkpoint["model_state"])
    prior = fit_availability(prepared, prepared.lists.fit)
    inputs = temporal_inputs(prepared, int(config["context_windows"]), availability_prior=prior)
    scores = score_temporal(model, inputs, prepared.lists.main_test)
    labels = prepared.labels.set_index("sample_id").label.reindex(scores.sample_id).to_numpy()
    known = pd.notna(labels)
    y = labels[known].astype(int)
    values = scores.score.to_numpy(float)[known]
    metrics = {
        "rows": int(len(y)),
        "positive_prevalence": float((y == 1).mean()),
        "auroc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
        "ap": float(average_precision_score(y, values)) if len(np.unique(y)) == 2 else None,
        "label_source": "controlled_exposure_proxy; not independent Service gold",
    }
    args.output.mkdir(parents=True)
    scores.to_parquet(args.output / "scores.parquet", index=False)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "protocol": "sn_frozen_tt_prior_gru_diagnostic_v1",
                "checkpoint": str(args.checkpoint.resolve()),
                "metrics": metrics,
                "limitations": [
                    "TT weights are frozen; SN fit-only normalization and availability prior are adapted.",
                    "SN labels remain controlled-exposure proxies rather than Service gold.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
