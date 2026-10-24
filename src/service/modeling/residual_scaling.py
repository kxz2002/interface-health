"""Normal held-out residual scales with service shrinkage and explicit masks."""

from __future__ import annotations

import numpy as np


def _arrays(errors, observed, services):
    errors = np.asarray(errors, dtype=float)
    observed = np.asarray(observed)
    services = np.asarray(services)
    if (
        errors.ndim != 2
        or errors.shape[1] != 12
        or observed.shape != errors.shape
        or observed.dtype != bool
        or services.shape != (len(errors),)
    ):
        raise ValueError("Residual scale shapes or boolean masks are invalid")
    if not np.isfinite(errors[observed]).all() or (errors[observed] < 0).any():
        raise ValueError("Observed squared residuals must be finite and nonnegative")
    if any(not isinstance(service, str) or not service for service in services.tolist()):
        raise ValueError("Residual scales require nonempty service IDs")
    return np.where(observed, errors, 0), observed, services


def fit_residual_scales(errors, observed, services, sample_ids, permitted_ids):
    errors, observed, services = _arrays(errors, observed, services)
    sample_ids = list(sample_ids)
    if (
        len(sample_ids) != len(errors)
        or len(set(sample_ids)) != len(sample_ids)
        or not sample_ids
        or set(sample_ids) != set(permitted_ids)
    ):
        raise ValueError("Residual scale fitting requires exact held-out Normal IDs")
    counts = observed.sum(0)
    if (counts == 0).any():
        raise ValueError("Every residual feature requires global Normal coverage")
    global_scale = errors.sum(0) / counts
    floor = np.maximum(global_scale * 0.1, 1e-4)
    scales, service_counts = {}, {}
    for service in sorted(set(services)):
        selected = services == service
        count = observed[selected].sum(0)
        estimate = (errors[selected].sum(0) + 32 * global_scale) / (count + 32)
        scales[service] = np.maximum(estimate, floor).tolist()
        service_counts[service] = count.tolist()
    return dict(
        schema_version="service-residual-scale-v1",
        global_scale=np.maximum(global_scale, floor).tolist(),
        service_scales=scales,
        service_counts=service_counts,
        global_counts=counts.tolist(),
        fit_ids=sorted(sample_ids),
        shrinkage_count=32,
        relative_floor=0.1,
        absolute_floor=1e-4,
    )


def scaled_value_loss(state, errors, observed, services):
    errors, observed, services = _arrays(errors, observed, services)
    if (
        state.get("schema_version") != "service-residual-scale-v1"
        or state.get("shrinkage_count") != 32
        or state.get("relative_floor") != 0.1
        or state.get("absolute_floor") != 1e-4
    ):
        raise ValueError("Unsupported residual scale state")
    scales = np.asarray(
        [state["service_scales"].get(service, state["global_scale"]) for service in services],
        dtype=float,
    ).reshape(-1, 12)
    if not np.isfinite(scales).all() or (scales < 1e-4).any():
        raise ValueError("Residual scale denominators must be finite and positive")
    return (errors / scales).sum(1) / np.maximum(observed.sum(1), 1)
