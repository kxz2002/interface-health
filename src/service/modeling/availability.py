"""Normal-only likelihood of observed modality availability patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def availability_patterns(prepared, ids):
    ids = list(ids)
    masks = prepared.numeric["masks"].set_index("sample_id").loc[ids]
    logs = prepared.logs.get("masks")
    if logs is None or any(sample_id not in logs for sample_id in ids):
        raise ValueError("Missing declared log availability")
    log_valid = np.array([logs[sample_id] for sample_id in ids])
    if log_valid.dtype != bool and len(ids):
        raise ValueError("Log availability must be boolean")
    resource = masks[["cpu_cores", "memory_usage_ratio", "process_count"]].any(axis=1)
    execution = masks.drop(columns=["cpu_cores", "memory_usage_ratio", "process_count"]).any(axis=1)
    return resource.to_numpy(int) * 4 + execution.to_numpy(int) * 2 + log_valid.astype(int)


def fit_availability(prepared, ids):
    ids = sorted(ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(prepared.lists.fit):
        raise ValueError("Availability fitting requires nonempty frozen fit IDs")
    labels = prepared.labels.set_index("sample_id").label.reindex(ids)
    splits = prepared.splits.set_index("sample_id").split.reindex(ids)
    if not labels.eq(0).fillna(False).all() or not splits.eq("fit").all():
        raise ValueError("Availability fitting requires known-normal fit samples")
    identity = prepared.identity.set_index("sample_id").loc[ids]
    frame = pd.DataFrame(
        dict(
            service=identity.canonical_service_id.to_numpy(),
            pattern=availability_patterns(prepared, ids),
        )
    )
    counts = {
        str(service): np.bincount(group.pattern, minlength=8).tolist()
        for service, group in frame.groupby("service")
    }
    return dict(schema_version="service-availability-v1", alpha=1.0, counts=counts, fit_ids=ids)


def score_availability(state, prepared, ids):
    if state.get("schema_version") != "service-availability-v1" or state.get("alpha") != 1.0:
        raise ValueError("Unsupported availability likelihood state")
    ids = list(ids)
    services = prepared.identity.set_index("sample_id").loc[ids, "canonical_service_id"]
    patterns = availability_patterns(prepared, ids)
    scores = []
    for service, pattern in zip(services, patterns):
        counts = np.asarray(state["counts"].get(str(service), []))
        if counts.shape != (8,) or not np.issubdtype(counts.dtype, np.integer):
            raise ValueError("Missing or invalid service availability counts")
        if (counts < 0).any() or counts.sum() <= 0:
            raise ValueError("Invalid availability fit counts")
        scores.append(-np.log((counts[pattern] + 1.0) / (counts.sum() + 8.0)))
    return np.asarray(scores, dtype=float)
