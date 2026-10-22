"""Fixed equal-weight score fusion with service-specific Normal-fit references."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(prepared, frames, ids):
    ids = sorted(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Fusion requires nonempty unique IDs")
    if set(frames) != {"value", "availability"}:
        raise ValueError("Fusion requires value and availability channels")
    result = prepared.identity.set_index("sample_id").loc[ids].copy()
    for name, frame in frames.items():
        if frame.sample_id.duplicated().any() or set(frame.sample_id) != set(ids):
            raise ValueError("Fusion channel IDs must exactly match frozen scope")
        aligned = frame.set_index("sample_id").loc[ids]
        values = aligned.score.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite fusion input")
        for column in result.columns.intersection(aligned.columns):
            if not result[column].equals(aligned[column]):
                raise ValueError("Conflicting fusion sample identity")
        result[name] = values
    return result


def fit_tail_fusion(prepared, frames, ids):
    ids = sorted(ids)
    if ids != sorted(prepared.lists.fit):
        raise ValueError("Fusion fitting requires the complete frozen fit scope")
    frame = _aligned(prepared, frames, ids)
    labels = prepared.labels.set_index("sample_id").label.reindex(ids)
    splits = prepared.splits.set_index("sample_id").split.reindex(ids)
    if not labels.eq(0).fillna(False).all() or not splits.eq("fit").all():
        raise ValueError("Fusion fitting requires known-normal fit samples")
    references = {
        str(service): {name: sorted(group[name].tolist()) for name in ["value", "availability"]}
        for service, group in frame.groupby("canonical_service_id")
    }
    return dict(
        schema_version="service-tail-fusion-v1",
        formula="sum_negative_log_smoothed_upper_tail",
        weights=dict(value=1.0, availability=1.0),
        fit_ids=ids,
        references=references,
    )


def score_tail_fusion(state, prepared, frames, ids):
    if (
        state.get("schema_version") != "service-tail-fusion-v1"
        or state.get("formula") != "sum_negative_log_smoothed_upper_tail"
        or state.get("weights") != dict(value=1.0, availability=1.0)
    ):
        raise ValueError("Unsupported fixed fusion state")
    frame = _aligned(prepared, frames, ids)
    frame["score"] = 0.0
    for service, group in frame.groupby("canonical_service_id"):
        reference = state["references"].get(str(service))
        if reference is None:
            raise ValueError("Missing service Normal-fit fusion reference")
        for name in ["value", "availability"]:
            values = np.asarray(reference.get(name, []), dtype=float)
            if (
                values.ndim != 1
                or not len(values)
                or not np.isfinite(values).all()
                or (np.diff(values) < 0).any()
            ):
                raise ValueError("Invalid sorted Normal-fit fusion reference")
            greater_equal = len(values) - np.searchsorted(values, group[name], side="left")
            tail = (greater_equal + 1.0) / (len(values) + 1.0)
            frame.loc[group.index, name + "_tail"] = tail
            frame.loc[group.index, "score"] += -np.log(tail)
    return frame.reset_index()
