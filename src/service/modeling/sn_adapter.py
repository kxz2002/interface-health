"""Build a model-facing SN view without changing the source preparation tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .data import SampleLists

SN_RESOURCE_MAPPING = {
    "cpu_cores": "cpu_usage_total_mean",
    "memory_usage_ratio": "memory_usage_mean",
    "process_count": None,
}
SN_EXECUTION_COLUMNS = [
    "observation_count",
    "duration_mean_ms",
    "duration_p95_ms",
    "duration_max_ms",
    "error_rate",
]
SN_LOG_COLUMNS = [
    "event_count",
    "matched_template_count",
    "max_matched_fraction",
    "unmatched_count",
]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ids(frame: pd.DataFrame) -> list[str]:
    return frame.sample_id.astype(str).tolist()


def _numeric(resource: pd.DataFrame, execution: pd.DataFrame) -> tuple[dict, dict]:
    resource = resource.set_index("sample_id")
    execution = execution.set_index("sample_id")
    values = pd.DataFrame(index=resource.index)
    masks = pd.DataFrame(index=resource.index)
    for target, source in SN_RESOURCE_MAPPING.items():
        if source is None:
            values[target] = np.nan
            masks[target] = False
            continue
        value = pd.to_numeric(resource[source], errors="coerce")
        valid = value.notna() & np.isfinite(value)
        values[target] = value.where(valid)
        masks[target] = valid.astype(bool)
    for column in SN_EXECUTION_COLUMNS:
        if column == "error_rate":
            known = execution.error_known_count
            value = execution.error_count / known.where(known.gt(0))
            valid = (
                execution.error_coverage.eq(1)
                & known.eq(execution.observation_count)
                & value.between(0, 1)
            )
        else:
            value = pd.to_numeric(execution[column], errors="coerce")
            valid = value.notna() & np.isfinite(value) & value.ge(0)
        values[column] = value.where(valid)
        masks[column] = valid.fillna(False).astype(bool)
    return values.reset_index(), masks.reset_index()


def _lists(identity: pd.DataFrame, splits: pd.DataFrame, labels: pd.DataFrame) -> SampleLists:
    fit = tuple(sorted(splits.loc[splits.training_eligible, "sample_id"].astype(str)))
    calibration = tuple(sorted(splits.loc[splits.calibration_eligible, "sample_id"].astype(str)))
    test = tuple(
        sorted(splits.loc[splits.run_role.isin(["test", "holdout"]), "sample_id"].astype(str))
    )
    all_ids = tuple(sorted(identity.sample_id.astype(str)))
    proxy_positive = tuple(sorted(labels.loc[labels.label.eq(1), "sample_id"].astype(str)))
    reasons = {
        "fit": "fixed SN Normal fit run role",
        "calibration": "fixed SN Normal calibration run role",
        "proxy_positive": "fault_test run exposure proxy; not independent Service gold",
    }
    payload = dict(
        common_fit=fit,
        common_calibration=calibration,
        common_main_test=test,
        extended_inference=all_ids,
        extended_main_test=test,
        diagnostic_pre=(),
        diagnostic_positive=proxy_positive,
        diagnostic_groups={},
        excluded=(),
        reasons=reasons,
        mode="sn_role_proxy",
    )
    return SampleLists(**payload, lists_hash=_hash(payload))


def _fit_normalize(values: pd.DataFrame, identity: pd.DataFrame, fit_ids: tuple[str, ...]):
    result = values.copy().set_index("sample_id")
    identity_index = identity.set_index("sample_id")
    fit = set(fit_ids)
    state = {}
    for column in result.columns:
        stats = {}
        for service in sorted(identity.canonical_service_id.astype(str).unique()):
            ids = [
                sample_id
                for sample_id in fit
                if str(identity_index.at[sample_id, "canonical_service_id"]) == service
            ]
            observed = result.loc[ids, column].dropna()
            mean = float(observed.mean()) if len(observed) else 0.0
            scale = float(observed.std(ddof=0)) if len(observed) else 1.0
            stats[service] = {"mean": mean, "scale": scale if scale > 1e-6 else 1.0}
        state[column] = stats
        for sample_id in result.index:
            service = str(identity_index.at[sample_id, "canonical_service_id"])
            parameter = stats[service]
            if pd.notna(result.at[sample_id, column]):
                result.at[sample_id, column] = (
                    result.at[sample_id, column] - parameter["mean"]
                ) / parameter["scale"]
    return result.reset_index(), state


def build_sn_prepared(root: Path):
    """Return a model-facing object while preserving source labels as Unknown."""
    root = Path(root)
    sn = root / "sn"
    resource = pd.read_parquet(sn / "resource_windows.parquet")
    execution = pd.read_parquet(sn / "execution_windows.parquet")
    logs = pd.read_parquet(sn / "log_windows.parquet")
    source_labels = pd.read_parquet(sn / "labels.parquet")
    splits = pd.read_parquet(sn / "splits.parquet")
    frames = {
        "resource": resource,
        "execution": execution,
        "logs": logs,
        "labels": source_labels,
        "splits": splits,
    }
    expected_ids = set(source_labels.sample_id)
    if not expected_ids or any(frame.sample_id.duplicated().any() for frame in frames.values()):
        raise ValueError("SN model adapter requires nonempty unique sample IDs")
    if any(set(frame.sample_id) != expected_ids for frame in frames.values()):
        raise ValueError("SN model adapter table ID sets differ")
    if source_labels.label.notna().any():
        raise ValueError("SN source labels must remain Unknown")
    identity = source_labels[
        [
            "sample_id",
            "dataset_id",
            "run_id",
            "canonical_service_id",
            "window_start_ms",
            "window_end_ms",
        ]
    ].copy()
    identity["case_id"] = identity["run_id"]
    values, masks = _numeric(resource, execution)
    values = values.set_index("sample_id").reindex(identity.sample_id).reset_index()
    masks = masks.set_index("sample_id").reindex(identity.sample_id).reset_index()
    quality = pd.DataFrame(
        {
            "sample_id": identity.sample_id,
            "resource_valid": masks[list(SN_RESOURCE_MAPPING)].any(axis=1).to_numpy(bool),
            "execution_valid": masks[SN_EXECUTION_COLUMNS].any(axis=1).to_numpy(bool),
        }
    )
    log_values = logs[["sample_id", "event_count"]].copy()
    log_values["matched_template_count"] = 0.0
    log_values["max_matched_fraction"] = 0.0
    log_values["unmatched_count"] = 0.0
    log_values = (
        log_values.set_index("sample_id").reindex(identity.sample_id).fillna(0).reset_index()
    )
    log_masks = pd.DataFrame({"sample_id": identity.sample_id})
    log_masks["event_count"] = log_values.event_count.gt(0)
    for column in SN_LOG_COLUMNS[1:]:
        log_masks[column] = False
    log_available = log_values.event_count.gt(0).to_numpy(bool)
    split_roles = splits.set_index("sample_id").run_role.reindex(source_labels.sample_id)
    if split_roles.isna().any():
        raise ValueError("SN split roles do not cover source labels")
    model_labels = source_labels[["sample_id"]].copy()
    model_labels["run_role"] = split_roles.to_numpy()
    model_labels["label"] = model_labels.run_role.eq("test").astype("Int64")
    model_labels["label_source"] = np.where(
        model_labels.label.eq(1), "controlled_exposure_proxy", "reference_role_proxy"
    )
    model_labels["label_scope"] = np.where(
        model_labels.label.eq(1), "fault_run_service_window_proxy", "normal_run_reference"
    )
    model_labels.loc[model_labels.run_role.eq("holdout"), "label"] = 0
    model_labels = model_labels.drop(columns="run_role")
    lists = _lists(identity, splits, model_labels)
    values, numeric_state = _fit_normalize(values, identity, lists.fit)
    versions = {
        "label_version": "sn_role_proxy_v1",
        "split_version": "sn_exploratory_roles_v1",
        "evaluation_version": "sn_controlled_exposure_proxy_v1",
        "preparation_schema_version": "service-preparation-v1",
    }
    provenance = {
        "source_labels": "all Unknown; model labels are derived role/proxy metadata",
        "resource_mapping": SN_RESOURCE_MAPPING,
        "execution_columns": SN_EXECUTION_COLUMNS,
        "log_statistics": SN_LOG_COLUMNS,
        "numeric_state": numeric_state,
        "alignment_assumption": "fixed_minus_28800_seconds_not_proven_timezone",
        "lists_hash": lists.lists_hash,
        "log_statistics_state_hash": _hash(SN_LOG_COLUMNS),
    }
    return SimpleNamespace(
        sample_ids=tuple(identity.sample_id.astype(str)),
        numeric={"values": values, "masks": masks, "quality": quality},
        logs={
            "statistics": {
                "values": log_values,
                "masks": log_masks,
                "feature_order": SN_LOG_COLUMNS,
                "state_hash": provenance["log_statistics_state_hash"],
            },
            "masks": dict(zip(identity.sample_id, log_available)),
            "cache": {"log_config": {"max_tokens": 1, "max_messages": 1}},
        },
        lists=lists,
        identity=identity,
        labels=model_labels,
        source_labels=source_labels,
        splits=splits.assign(
            split=splits.run_role.map(
                {"fit": "fit", "calibration": "calibration", "holdout": "test", "test": "test"}
            )
        ),
        versions=versions,
        hashes={"preparation_root": str(root.resolve())},
        provenance=provenance,
    )
