"""Masked Normal state distances and fixed robust max score fusion."""

from __future__ import annotations

import numpy as np


def _values(values, observed, services):
    values = np.asarray(values, dtype=float)
    observed = np.asarray(observed)
    services = np.asarray(services)
    if (
        values.ndim != 2
        or values.shape[1] != 12
        or observed.shape != values.shape
        or observed.dtype != bool
        or services.shape != (len(values),)
    ):
        raise ValueError("Normal state shapes or masks are invalid")
    if not np.isfinite(values[observed]).all():
        raise ValueError("Observed Normal state values must be finite")
    if any(not isinstance(service, str) or not service for service in services.tolist()):
        raise ValueError("Normal state requires service IDs")
    return values, observed, services


def fit_normal_state(values, observed, services, sample_ids, permitted_ids):
    values, observed, services = _values(values, observed, services)
    sample_ids = list(sample_ids)
    if (
        not sample_ids
        or len(sample_ids) != len(values)
        or len(set(sample_ids)) != len(sample_ids)
        or set(sample_ids) != set(permitted_ids)
    ):
        raise ValueError("Normal state requires exact permitted fit IDs")
    global_center, global_scale = [], []
    for feature in range(12):
        selected = values[observed[:, feature], feature]
        if not len(selected):
            raise ValueError("Normal state requires global field coverage")
        center = float(np.median(selected))
        global_center.append(center)
        global_scale.append(max(float(np.median(np.abs(selected - center))) * 1.4826, 0.05))
    references = {}
    for service in sorted(set(services)):
        centers, scales, counts = [], [], []
        for feature in range(12):
            selected = values[(services == service) & observed[:, feature], feature]
            counts.append(len(selected))
            if len(selected) < 32:
                centers.append(global_center[feature])
                scales.append(global_scale[feature])
            else:
                center = float(np.median(selected))
                centers.append(center)
                scales.append(max(float(np.median(np.abs(selected - center))) * 1.4826, 0.05))
        references[service] = dict(center=centers, scale=scales, counts=counts)
    return dict(
        schema_version="service-normal-state-v1",
        fit_ids=sorted(sample_ids),
        global_center=global_center,
        global_scale=global_scale,
        services=references,
        minimum_count=32,
        scale_floor=0.05,
    )


def score_normal_state(state, values, observed, services):
    values, observed, services = _values(values, observed, services)
    if (
        state.get("schema_version") != "service-normal-state-v1"
        or state.get("minimum_count") != 32
        or state.get("scale_floor") != 0.05
    ):
        raise ValueError("Unsupported Normal state schema")
    centers, scales = [], []
    for service in services:
        reference = state["services"].get(service)
        centers.append(reference["center"] if reference else state["global_center"])
        scales.append(reference["scale"] if reference else state["global_scale"])
    centers, scales = np.asarray(centers).reshape(-1, 12), np.asarray(scales).reshape(-1, 12)
    if not np.isfinite(centers).all() or not np.isfinite(scales).all() or (scales < 0.05).any():
        raise ValueError("Invalid Normal state reference")
    residual = np.where(observed, values - centers, 0) / scales
    return np.square(residual).sum(1) / np.maximum(observed.sum(1), 1)


def fit_fusion_reference(scores, sample_ids, permitted_ids):
    scores = np.asarray(scores, dtype=float)
    sample_ids = list(sample_ids)
    if scores.shape != (len(sample_ids), 2) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Fusion reference requires finite paired scores")
    if len(set(sample_ids)) != len(sample_ids) or set(sample_ids) != set(permitted_ids):
        raise ValueError("Fusion reference requires exact held-out Normal IDs")
    center = np.median(scores, axis=0)
    scale = np.maximum(
        np.median(np.abs(scores - center), axis=0) * 1.4826, np.maximum(scores.std(0) * 0.1, 1e-4)
    )
    return dict(
        schema_version="service-state-temporal-max-v1",
        center=center.tolist(),
        scale=scale.tolist(),
        fit_ids=sorted(sample_ids),
    )


def fuse_state_temporal(state, scores):
    scores = np.asarray(scores, dtype=float)
    center, scale = np.asarray(state["center"]), np.asarray(state["scale"])
    if (
        state.get("schema_version") != "service-state-temporal-max-v1"
        or scores.ndim != 2
        or scores.shape[1] != 2
        or center.shape != (2,)
        or scale.shape != (2,)
    ):
        raise ValueError("Invalid fixed fusion shape or schema")
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale < 1e-4).any()
    ):
        raise ValueError("Invalid fusion values or scale")
    return ((scores - center) / scale).max(1)
