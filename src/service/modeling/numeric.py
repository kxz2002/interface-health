"""Fit-only numeric preprocessing with source-quality masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericState:
    fields: dict
    fit_ids: tuple[str, ...]
    config: dict
    state_hash: str


def _quality_mask(frame, modality):
    mask = pd.Series(True, index=frame.index)
    if "quality_status" in frame:
        mask &= (
            frame.quality_status.astype(str)
            .str.upper()
            .isin({"OBSERVED", "OBSERVED_ERROR_PARTIAL", "RESOURCE_VALID"})
        )
    if modality == "resource" and "resource_valid" in frame:
        mask &= frame.resource_valid.eq(True).fillna(False)
    return mask


def _numeric_frames(tables, prep):
    values = pd.DataFrame(index=pd.Index(tables.resource.sample_id, name="sample_id"))
    masks = pd.DataFrame(index=values.index)
    for modality in ("resource", "execution"):
        frame = getattr(tables, modality).set_index("sample_id").reindex(values.index)
        fields = (
            ["cpu_cores", "memory_usage_ratio", "process_count"]
            if modality == "resource"
            else [
                "observation_count",
                "duration_mean_ms",
                "duration_p95_ms",
                "duration_max_ms",
                "error_rate",
            ]
        )
        quality = _quality_mask(frame, modality)
        if modality == "execution":
            quality &= frame.observation_count.gt(0)
        for field in fields:
            if field == "error_rate":
                raw = frame.error_count / frame.error_known_count.where(
                    frame.error_known_count.gt(0)
                )
                valid = (
                    quality
                    & frame.error_coverage.eq(1)
                    & frame.error_known_count.eq(frame.observation_count)
                    & frame.error_count.ge(0)
                    & frame.error_count.le(frame.error_known_count)
                )
            else:
                raw = pd.to_numeric(frame[field], errors="coerce")
                valid = quality & raw.ge(0)
            valid &= np.isfinite(raw)
            values[field] = raw.where(valid)
            if field in prep.get("log1p", []):
                values[field] = np.log1p(values[field])
            masks[field] = valid.fillna(False).astype(bool)
    return values, masks


def fit_numeric(tables, fit_ids, config):
    from .config import content_hash

    prep = dict(config["preprocessing"])
    values, masks = _numeric_frames(tables, prep)
    if set(fit_ids) - set(values.index):
        raise ValueError("Unknown fit IDs")
    if hasattr(tables, "splits"):
        split = tables.splits.set_index("sample_id").split
        label = tables.labels.set_index("sample_id").label
        allowed = set(split.index[split.eq("fit")]) & set(label.index[label.eq(0).fillna(False)])
        if not set(fit_ids) <= allowed:
            raise ValueError("Fit requires known-normal fit IDs")
    fit = values.loc[sorted(fit_ids)]
    services = tables.resource.set_index("sample_id").canonical_service_id
    fields = {}
    epsilon = prep.get("scale_epsilon", 1e-6)
    for field in values:
        observed = fit[field].dropna()
        mean = float(observed.mean()) if len(observed) else None
        std = float(observed.std(ddof=0)) if len(observed) else 0.0
        scale = std if std > epsilon else 1.0
        service_stats = {}
        for service in sorted(services.unique()):
            subset = fit.loc[services.reindex(fit.index).eq(service), field].dropna()
            service_std = float(subset.std(ddof=0)) if len(subset) else 0.0
            service_stats[str(service)] = dict(
                mean=float(subset.mean()) if len(subset) else None,
                scale=service_std if service_std > epsilon else scale,
                fallback=(
                    "service" if service_std > epsilon else "global" if std > epsilon else "unit"
                ),
                available=len(subset) > 0,
            )
        modality = (
            "resource"
            if field in ["cpu_cores", "memory_usage_ratio", "process_count"]
            else "execution"
        )
        fields[f"{modality}.{field}"] = dict(
            services=service_stats, global_mean=mean, global_scale=scale, available=mean is not None
        )
    payload = dict(fields=fields, fit_ids=sorted(fit_ids), config=prep)
    return NumericState(fields, tuple(sorted(fit_ids)), prep, content_hash(payload))


def transform_numeric(tables, state):
    values, masks = _numeric_frames(tables, state.config)
    services = tables.resource.set_index("sample_id").canonical_service_id
    for field in values:
        modality = (
            "resource"
            if field in ["cpu_cores", "memory_usage_ratio", "process_count"]
            else "execution"
        )
        stats = state.fields[f"{modality}.{field}"]
        for sample_id in values.index:
            parameters = stats["services"].get(str(services.at[sample_id]))
            if parameters is None:
                parameters = dict(mean=None, scale=stats["global_scale"])
            if parameters["mean"] is None:
                masks.at[sample_id, field] = False
                values.at[sample_id, field] = 0.0
            elif masks.at[sample_id, field]:
                values.at[sample_id, field] = (
                    values.at[sample_id, field] - parameters["mean"]
                ) / parameters["scale"]
    quality = pd.DataFrame(index=values.index)
    quality["resource_valid"] = masks[["cpu_cores", "memory_usage_ratio", "process_count"]].any(
        axis=1
    )
    quality["execution_valid"] = masks[
        [
            "observation_count",
            "duration_mean_ms",
            "duration_p95_ms",
            "duration_max_ms",
            "error_rate",
        ]
    ].any(axis=1)
    return {
        "values": values.reset_index(),
        "masks": masks.reset_index(),
        "quality": quality.reset_index(),
    }
