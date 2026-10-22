"""Read immutable TT preparation and schema-2 label bundles on their full grid."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from src.service.labeling.bundle import read_bundle
from src.service.labeling.rules import IDENTITY_COLUMNS
from src.service.modality_preparation import GRID_COLUMNS, KEYS, stable_ids, validate_grid
from src.service.preparation import checked_path, file_digest

from .config import content_hash, validate_config


@dataclass(frozen=True)
class InputTables:
    grid: pd.DataFrame
    resource: pd.DataFrame
    execution: pd.DataFrame
    log: pd.DataFrame
    labels: pd.DataFrame
    splits: pd.DataFrame
    evaluation: pd.DataFrame
    versions: dict
    hashes: dict
    feature_tables: dict[str, pd.DataFrame]
    config: dict
    protocol: dict | None = None


@dataclass(frozen=True)
class SampleLists:
    common_fit: tuple[str, ...]
    common_calibration: tuple[str, ...]
    common_main_test: tuple[str, ...]
    extended_inference: tuple[str, ...]
    extended_main_test: tuple[str, ...]
    diagnostic_pre: tuple[str, ...]
    diagnostic_positive: tuple[str, ...]
    diagnostic_groups: dict
    excluded: tuple[str, ...]
    reasons: dict
    lists_hash: str
    mode: str

    @property
    def fit(self):
        return self.common_fit

    @property
    def calibration(self):
        return self.common_calibration

    @property
    def main_test(self):
        return self.common_main_test

    @property
    def full_test(self):
        return self.extended_main_test

    @property
    def fit_diagnostic(self):
        return self.common_fit

    @property
    def exclusions(self):
        return self.excluded


@dataclass(frozen=True)
class PreparedData:
    sample_ids: tuple[str, ...]
    numeric: dict
    logs: dict
    lists: SampleLists
    identity: pd.DataFrame | None = None
    labels: pd.DataFrame | None = None
    splits: pd.DataFrame | None = None
    versions: dict = field(default_factory=dict)
    hashes: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_prepared(self)


def validate_sample_lists(lists: SampleLists) -> str:
    payload = asdict(lists)
    declared = payload.pop("lists_hash")
    actual = content_hash(payload)
    if actual != declared:
        raise ValueError("Sample lists content/hash mismatch")
    return actual


def _validate_prepared(prepared: PreparedData) -> None:
    ids = set(prepared.sample_ids)
    if len(ids) != len(prepared.sample_ids) or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in ids
    ):
        raise ValueError("Invalid prepared sample IDs")
    numeric = prepared.numeric
    for name in ["values", "masks", "quality"]:
        frame = numeric.get(name)
        if (
            not isinstance(frame, pd.DataFrame)
            or "sample_id" not in frame
            or frame.sample_id.duplicated().any()
            or set(frame.sample_id) != ids
            or frame.columns.duplicated().any()
        ):
            raise ValueError(f"Invalid prepared numeric {name} IDs/shape")
    values = numeric["values"].set_index("sample_id").reindex(prepared.sample_ids)
    masks = numeric["masks"].set_index("sample_id").reindex(prepared.sample_ids)
    if list(values) != list(masks) or not len(values.columns):
        raise ValueError("Prepared numeric values/masks feature mismatch")
    for column in masks:
        if not pd.api.types.is_bool_dtype(masks[column]) or masks[column].isna().any():
            raise ValueError("Prepared numeric masks must be nonnull booleans")
        try:
            observed = values.loc[masks[column], column].to_numpy(dtype=float)
        except (ValueError, TypeError) as error:
            raise ValueError("Prepared numeric observations must be numeric") from error
        if not np.isfinite(observed).all():
            raise ValueError("Prepared valid numeric observations must be finite")
    quality = numeric["quality"].drop(columns="sample_id")
    if not {"resource_valid", "execution_valid"} <= set(quality):
        raise ValueError("Missing prepared numeric quality flags")
    if any(
        not pd.api.types.is_bool_dtype(quality[column]) or quality[column].isna().any()
        for column in quality
    ):
        raise ValueError("Prepared numeric quality flags must be nonnull booleans")
    logs = prepared.logs
    names = [
        "input_ids",
        "attention_mask",
        "special_mask",
        "group_ids",
        "texts",
        "counts",
        "masks",
        "stats",
        "coverage",
    ]
    for name in names:
        if not isinstance(logs.get(name), dict) or set(logs[name]) != ids:
            raise ValueError(f"Prepared log {name} sample IDs mismatch")
    cache = logs.get("cache", {})
    log_config = cache.get("log_config", {})
    max_tokens, max_messages = log_config.get("max_tokens"), log_config.get("max_messages")
    if type(max_tokens) is not int or type(max_messages) is not int:
        raise ValueError("Missing prepared log token/message budgets")
    for sample_id in prepared.sample_ids:
        count = logs["counts"][sample_id]
        valid = logs["masks"][sample_id]
        if (
            type(count) is not int
            or not 0 <= count <= max_messages
            or type(valid) is not bool
            or valid != bool(count)
        ):
            raise ValueError("Prepared log count/validity mismatch")
        for name in ["input_ids", "attention_mask", "special_mask", "group_ids", "texts"]:
            if not isinstance(logs[name][sample_id], list) or len(logs[name][sample_id]) != count:
                raise ValueError("Prepared log message axis mismatch")
        groups, texts = logs["group_ids"][sample_id], logs["texts"][sample_id]
        if any(
            not isinstance(group, str) or not group.startswith(("template:", "unmatched:"))
            for group in groups
        ):
            raise ValueError("Invalid prepared log group ID")
        if any(not isinstance(text, str) or not text for text in texts) or len(set(texts)) != count:
            raise ValueError("Invalid/duplicate prepared log message")
        for message_index in range(count):
            token_ids = logs["input_ids"][sample_id][message_index]
            attention = logs["attention_mask"][sample_id][message_index]
            special = logs["special_mask"][sample_id][message_index]
            if any(
                not isinstance(array, list) or len(array) != max_tokens
                for array in [token_ids, attention, special]
            ):
                raise ValueError("Prepared log token axis mismatch")
            if any(
                not isinstance(value, Integral) or isinstance(value, bool) or value < 0
                for value in token_ids
            ):
                raise ValueError("Invalid prepared log token IDs")
            if any(value not in (0, 1) for value in attention + special):
                raise ValueError("Invalid prepared log token masks")
            if not any(active and not excluded for active, excluded in zip(attention, special)):
                raise ValueError("Prepared log message has no body token")
        stats = logs["stats"][sample_id]
        if set(stats) != {
            "event_count",
            "matched_template_count",
            "max_matched_fraction",
            "unmatched_count",
        }:
            raise ValueError("Prepared log statistics schema mismatch")
        for name in ["event_count", "matched_template_count", "unmatched_count"]:
            if type(stats[name]) is not int or stats[name] < 0:
                raise ValueError("Invalid prepared log event/template count")
        fraction = stats["max_matched_fraction"]
        if (
            bool(stats["event_count"]) != valid
            or stats["unmatched_count"] > stats["event_count"]
            or stats["matched_template_count"] > stats["event_count"]
        ):
            raise ValueError("Prepared log full-window statistics/validity mismatch")
        if (
            valid and (fraction is None or not np.isfinite(fraction) or not 0 <= fraction <= 1)
        ) or (not valid and fraction is not None):
            raise ValueError("Invalid prepared matched-template fraction")
    list_names = [
        "common_fit",
        "common_calibration",
        "common_main_test",
        "extended_inference",
        "extended_main_test",
        "diagnostic_pre",
        "diagnostic_positive",
        "excluded",
    ]
    for name in list_names:
        members = getattr(prepared.lists, name)
        if len(set(members)) != len(members) or set(members) - ids:
            raise ValueError(f"Prepared membership {name} sample IDs mismatch")


def assemble_prepared(
    tables: InputTables,
    numeric: dict,
    logs: dict,
    lists: SampleLists,
    provenance: dict | None = None,
) -> PreparedData:
    """Assemble modality outputs while preserving stable sample identity."""
    ids = tuple(tables.grid.sort_values("sample_id").sample_id)
    return PreparedData(
        sample_ids=ids,
        numeric=numeric,
        logs=logs,
        lists=lists,
        identity=tables.grid.copy(),
        labels=tables.labels.copy() if hasattr(tables, "labels") else None,
        splits=tables.splits.copy() if hasattr(tables, "splits") else None,
        versions=dict(getattr(tables, "versions", {})),
        hashes=dict(getattr(tables, "hashes", {})),
        provenance=dict(provenance or {}),
    )


def build_sample_lists(
    tables: InputTables, numeric: dict, log_valid: pd.Series, mode: str
) -> SampleLists:
    if mode not in {"common", "extended"}:
        raise ValueError("Unknown membership mode")
    frame = tables.grid.set_index("sample_id").sort_index()
    masks = numeric["masks"].set_index("sample_id").reindex(frame.index)
    valid = masks.fillna(False).any(axis=1)
    logs = log_valid.reindex(frame.index, fill_value=False).fillna(False).astype(bool)
    labels = tables.labels.set_index("sample_id").reindex(frame.index)
    splits = tables.splits.set_index("sample_id").reindex(frame.index).split
    eligible = (
        tables.evaluation.set_index("sample_id")
        .reindex(frame.index)
        .evaluation_eligible.fillna(False)
    )
    known = labels.label.notna()
    normal = labels.label.eq(0).fillna(False)
    positive = labels.label.eq(1).fillna(False)

    def select(mask):
        return tuple(frame.index[mask.fillna(False)])

    fit = select(valid & splits.eq("fit") & normal)
    calibration = select(valid & splits.eq("calibration") & normal)
    pre_reference = labels.label_source.eq("pre_intervention_reference")
    main = select(valid & eligible & known & splits.eq("test") & ~pre_reference)
    extended = select((valid | logs) & splits.eq("test"))
    extended_main = select((valid | logs) & eligible & known & splits.eq("test") & ~pre_reference)
    pre_mask = (
        valid & splits.eq("test") & normal & labels.label_source.eq("pre_intervention_reference")
    )
    positive_mask = valid & splits.eq("test") & positive
    groups = {}
    for (case, service), group in frame.groupby(["case_id", "canonical_service_id"]):
        pre = tuple(group.index[pre_mask.reindex(group.index)])
        pos = tuple(group.index[positive_mask.reindex(group.index)])
        if pre or pos:
            groups[f"{case}|{service}"] = dict(pre=pre, positive=pos, reference_available=bool(pre))
    reasons = {}
    for sample_id in frame.index:
        row_reasons = []
        if not valid.at[sample_id]:
            row_reasons.append("log_only" if logs.at[sample_id] else "all_modalities_invalid")
        if not known.at[sample_id]:
            row_reasons.append("unknown_label")
        if not eligible.at[sample_id]:
            row_reasons.append("evaluation_ineligible")
        reasons[sample_id] = row_reasons
    payload = dict(
        common_fit=fit,
        common_calibration=calibration,
        common_main_test=main,
        extended_inference=extended,
        extended_main_test=extended_main,
        diagnostic_pre=select(pre_mask),
        diagnostic_positive=select(positive_mask),
        diagnostic_groups=groups,
        excluded=select(~valid & ~logs),
        reasons=reasons,
        mode=mode,
    )
    return SampleLists(**payload, lists_hash=content_hash(payload))


def _message_features(value: str) -> str:
    messages = json.loads(value)
    if not isinstance(messages, list):
        raise ValueError("Invalid prepared log messages")
    clean = []
    for item in messages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("message"), str)
            or type(item.get("count")) is not int
            or item["count"] <= 0
        ):
            raise ValueError("Invalid prepared log message/count")
        clean.append({"message": item["message"], "count": item["count"]})
    return json.dumps(clean, ensure_ascii=False)


def _identity(frame: pd.DataFrame) -> pd.DataFrame:
    if set(GRID_COLUMNS) - set(frame) or frame[GRID_COLUMNS].isna().any().any():
        raise ValueError("Missing input identity")
    for name in ["window_start_ms", "window_end_ms"]:
        if not pd.api.types.is_integer_dtype(frame[name]):
            raise ValueError("Noninteger input interval")
    validate_grid(frame)
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id")
    expected = stable_ids(frame)
    if not expected.sample_id.equals(frame.sample_id):
        raise ValueError("Invalid stable sample_id")
    if "case_id" in frame and not frame.case_id.eq(frame.run_id).all():
        raise ValueError("Case/run mismatch")
    dataset_ids = set(frame.dataset_id)
    if len(dataset_ids) != 1 or not dataset_ids <= {"tt", "service_20261017_004546"}:
        raise ValueError("Only TT inputs supported")
    return frame[GRID_COLUMNS].sort_values(KEYS).reset_index(drop=True)


def load_inputs(preparation: Path, label_bundle: Path, config: dict) -> InputTables:
    validate_config(config)
    resolved = {key: value for key, value in config.items() if key != "config_sha256"}
    preparation = Path(preparation)
    if preparation.name == "tt" and not (preparation / "artifact_manifest.json").is_file():
        preparation = preparation.parent
    manifest_path = preparation / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prepared_hash = file_digest(manifest_path)
    bundle = read_bundle(Path(label_bundle))
    label_manifest = bundle["manifest"]
    if label_manifest["schema_version"] != 2:
        raise ValueError("Model requires schema 2 evaluation bundle")
    if (
        label_manifest["inputs"]["preparation_manifest_sha256"] != prepared_hash
        or bundle["evidence"]["preparation_manifest_sha256"] != prepared_hash
    ):
        raise ValueError("Label/preparation manifest binding mismatch")
    versions = {key: label_manifest[key] for key in ["label_version", "split_version"]}
    versions["evaluation_version"] = bundle["protocol"]["evaluation_version"]
    versions["preparation_schema_version"] = config["preparation_schema_version"]
    grid = _identity(bundle["labels"])
    tables, features = {}, {}
    hashes = {"preparation_manifest_sha256": prepared_hash, "config_sha256": content_hash(resolved)}
    bundle_directory = Path(label_bundle)
    if not (bundle_directory / "manifest.json").is_file():
        bundle_directory /= "labels"
    hashes["label_manifest_sha256"] = file_digest(bundle_directory / "manifest.json")
    for name in ["labels", "splits", "evaluation"]:
        frame = bundle[name].sort_values(KEYS).reset_index(drop=True)
        bundle[name] = frame
        if not _identity(frame).equals(grid):
            raise ValueError(f"{name}: full grid mismatch")
        frame.attrs.update(versions)
        hash_name = {"labels": "label", "splits": "split", "evaluation": "evaluation"}[name]
        hashes[f"{hash_name}_sha256"] = label_manifest["files"][f"{name}.parquet"]["sha256"]
    for modality in ["resource", "execution", "log"]:
        stem = f"tt/{modality}_windows"
        paths = []
        for suffix in [".parquet", ".schema.json"]:
            relative = stem + suffix
            path = checked_path(preparation, relative)
            expected = manifest.get("artifacts", {}).get(relative, {}).get("sha256")
            if expected is None or file_digest(path) != expected:
                raise ValueError(f"Preparation artifact hash mismatch: {relative}")
            hashes[relative] = expected
            paths.append(path)
        frame = pd.read_parquet(paths[0])
        schema = json.loads(paths[1].read_text())
        if schema.get("schema_version") != config["preparation_schema_version"]:
            raise ValueError("Unsupported preparation schema version")
        if schema.get("profile") != "tt_source_verified" or schema.get("key") != KEYS:
            raise ValueError("Unsupported preparation profile/key")
        fields = schema.get("fields")
        if not isinstance(fields, list) or any(not isinstance(field, dict) for field in fields):
            raise ValueError("Malformed preparation schema fields")
        if [field.get("name") for field in fields] != list(frame) or any(
            field.get("dtype") != str(frame[field["name"]].dtype) for field in fields
        ):
            raise ValueError("Preparation schema fields/dtypes mismatch")
        allowed = config["features"][modality]
        if schema.get("feature_fields") != allowed or set(allowed) - set(frame):
            raise ValueError("Preparation feature allowlist mismatch")
        required = ["quality_status"] + (["resource_valid"] if modality == "resource" else [])
        if set(required) - set(frame):
            raise ValueError("Missing source quality columns")
        if not _identity(frame).equals(grid):
            raise ValueError(f"{modality}: full grid mismatch")
        frame = frame.sort_values(KEYS).reset_index(drop=True)
        frame.attrs.update(versions)
        tables[modality] = frame
        features[modality] = frame[allowed].copy()
        if modality == "log":
            features[modality]["messages_json"] = frame.messages_json.map(_message_features)
    grid = bundle["labels"][IDENTITY_COLUMNS].sort_values(KEYS).reset_index(drop=True)
    grid.attrs.update(versions)
    return InputTables(
        grid,
        tables["resource"],
        tables["execution"],
        tables["log"],
        bundle["labels"],
        bundle["splits"],
        bundle["evaluation"],
        versions,
        hashes,
        features,
        config,
        bundle["protocol"],
    )
