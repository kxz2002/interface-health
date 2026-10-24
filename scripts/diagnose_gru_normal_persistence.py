"""Compare causal persistence with the GRU on the inner Normal validation Case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import LocalRuntime, read_prepared
from src.service.modeling.data import load_inputs
from src.service.modeling.temporal import temporal_inputs
from src.service.modeling.temporal_tuning import prepare_inner


def persistence_diagnostic(inputs, validation_ids):
    positions = {sample_id: index for index, sample_id in enumerate(inputs["ids"])}
    selected = [positions[sample_id] for sample_id in sorted(validation_ids)]
    context = inputs["context"][selected]
    history_observed = (context[..., 12:24] > 0) & (context[..., -1:] > 0)
    predictions = np.zeros((len(selected), 12), dtype=np.float32)
    for lag in range(context.shape[1]):
        predictions = np.where(history_observed[:, lag], context[:, lag, :12], predictions)
    observed = inputs["observed"][selected]
    squared = np.where(observed, (predictions - inputs["values"][selected]) ** 2, 0)
    losses = squared.sum(-1) / np.maximum(observed.sum(-1), 1)
    return dict(
        validation_count=len(selected),
        value_loss=float(losses.mean()),
        observed_without_history=int((observed & ~history_observed.any(1)).sum()),
        observed_cells=int(observed.sum()),
        per_feature={
            name: dict(
                observed=int(observed[:, index].sum()),
                mse=float(squared[:, index].sum() / max(observed[:, index].sum(), 1)),
            )
            for index, name in enumerate(inputs["feature_order"])
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["prepared", "preparation", "label-bundle"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    outer_tables, outer, _ = read_prepared(args.prepared)
    tables = load_inputs(args.preparation, args.label_bundle, outer_tables.config)
    if tables.hashes != outer_tables.hashes or tables.versions != outer_tables.versions:
        raise ValueError("Normal diagnostic input binding mismatch")
    inner, audit = prepare_inner(tables, outer, LocalRuntime().tokenizer(tables.config))
    print(
        json.dumps(
            dict(
                selection_scope="outer_fit_normal_only",
                preprocessor_hash=audit["preprocessor_hash"],
                validation_case=audit["validation_case"],
                persistence=persistence_diagnostic(
                    temporal_inputs(inner, 4), audit["validation_ids"]
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
