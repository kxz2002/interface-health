"""Closed-window sequence inputs shared by neural anomaly baselines."""

import numpy as np

from .temporal import temporal_inputs


def closed_window_sequences(prepared, sequence_length):
    if type(sequence_length) is not int or sequence_length < 2:
        raise ValueError("Sequence length must be an integer of at least two")

    inputs = temporal_inputs(prepared, context_windows=sequence_length - 1)
    target = np.concatenate(
        [inputs["features"], np.ones((len(inputs["ids"]), 1), dtype=np.float32)], axis=1
    )
    sequences = np.concatenate([inputs["context"], target[:, None, :]], axis=1)
    present = sequences[:, :, -1].astype(bool)
    observed = sequences[:, :, 12:24].astype(bool) & present[:, :, None]
    return {
        "ids": inputs["ids"],
        "sequences": sequences,
        "present": present,
        "observed": observed,
        "history_ids": inputs["history_ids"],
        "feature_order": inputs["feature_order"],
    }
