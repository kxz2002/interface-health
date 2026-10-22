"""Fit-only Drain3 templates and deterministic, per-message text inputs."""

from __future__ import annotations

import importlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.service.preparation import file_digest

from .config import content_hash
from .data import InputTables

CLEANING_VERSION = "conservative-log-body-v1"
TOKENIZATION_VERSION = "per-message-head-tail-v1"
LOG_STAT_FIELDS = (
    "event_count",
    "matched_template_count",
    "max_matched_fraction",
    "unmatched_count",
)
_DEFAULTS = dict(
    max_messages=16,
    error_quota=4,
    unmatched_quota=4,
    max_tokens=128,
    head_fraction=2 / 3,
    freeze_templates_after_fit=True,
)
_TIMESTAMP = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
    r"(?:[.,]\d{1,9})?(?:Z|[+-]\d{2}:\d{2})?)(?:[ \t]+|$)"
)
_LEVELS = r"TRACE|DEBUG|INFO|WARN|ERROR|FATAL"
_SPRING = re.compile(
    rf"^(?P<level>{_LEVELS})\s+\d+\s+---\s+\[[^\]\r\n]+\]"
    r"\s+[A-Za-z_$][\w.$-]*\s*:\s*(?P<body>.*)$",
    re.DOTALL,
)
_LEVEL = re.compile(rf"^(?:\[(?P<bracket>{_LEVELS})\]|(?P<plain>{_LEVELS}))(?:[ \t]+|$)")
_THREAD = re.compile(
    r"^\[(?:main|Thread-\d+|pool-\d+-thread-\d+|http-nio-[\w-]+|"
    r"nioEventLoopGroup-[\w-]+)\](?:[ \t]+|$)"
)


@dataclass(frozen=True)
class LogState:
    templates: dict[str, tuple[str, ...]]
    config: dict
    fit_ids: tuple[str, ...]
    fit_input_hash: str
    snapshot_hash: str
    state_hash: str
    miners: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class LogStatisticsState:
    fields: dict
    fit_ids: tuple[str, ...]
    config: dict
    source_binding: dict
    fit_input_hash: str
    state_hash: str


def _check_fit_ids(tables: InputTables, fit_ids: set[str], ids: set[str]) -> None:
    if set(fit_ids) - ids:
        raise ValueError("Unknown fit IDs")
    if hasattr(tables, "splits") and hasattr(tables, "labels"):
        split = tables.splits.set_index("sample_id").split
        label = tables.labels.set_index("sample_id").label
        allowed = set(split.index[split.eq("fit")]) & set(label.index[label.eq(0).fillna(False)])
        if not set(fit_ids) <= allowed:
            raise ValueError("Log fitting requires known-normal fit IDs")


def _log_config(config: dict) -> dict:
    supplied = config.get("log", config)
    if not isinstance(supplied, dict):
        raise ValueError("Expected log configuration mapping")
    if set(supplied) - set(_DEFAULTS) - {"model", "revision", "tokenizer_revision"}:
        raise ValueError("Unknown log configuration key")
    resolved = {**_DEFAULTS, **supplied}
    for name, upper in [("max_messages", 16), ("max_tokens", 128)]:
        if type(resolved[name]) is not int or not 0 < resolved[name] <= upper:
            raise ValueError(f"Invalid log budget: {name}")
    for name in ["error_quota", "unmatched_quota"]:
        if type(resolved[name]) is not int or not 0 <= resolved[name] <= 16:
            raise ValueError(f"Invalid log quota: {name}")
    if resolved["head_fraction"] != 2 / 3 or resolved["freeze_templates_after_fit"] is not True:
        raise ValueError(
            "Log processing requires frozen templates and the specified head/tail rule"
        )
    return resolved


def _clean(text: str) -> tuple[str, bool]:
    body = text.strip()
    stamp = _TIMESTAMP.match(body)
    if stamp:
        try:
            datetime.fromisoformat(stamp["stamp"].replace(",", ".").replace("Z", "+00:00"))
        except ValueError:
            return body, False
        body = body[stamp.end() :]
        spring = _SPRING.match(body)
        if spring:
            return spring["body"].strip(), spring["level"] in {"ERROR", "FATAL"}
    thread = _THREAD.match(body)
    if thread and _LEVEL.match(body[thread.end() :]):
        body = body[thread.end() :]
    level = _LEVEL.match(body)
    error = False
    if level:
        error = (level["plain"] or level["bracket"]) in {"ERROR", "FATAL"}
        body = body[level.end() :]
        thread = _THREAD.match(body)
        if thread:
            body = body[thread.end() :]
    return body.strip(), error


def _frame(tables: InputTables) -> pd.DataFrame:
    frame = tables.log
    required = {"sample_id", "canonical_service_id", "messages_json", "quality_status"}
    if required - set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Missing log identity/messages/source quality")
    if frame.sample_id.duplicated().any():
        raise ValueError("Duplicate log sample_id")
    if not frame.sample_id.map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Invalid log sample_id")
    if hasattr(tables, "grid"):
        ids = tables.grid.sample_id
        if ids.duplicated().any() or set(ids) != set(frame.sample_id):
            raise ValueError("Log/full grid sample IDs mismatch")
    return frame.sort_values("sample_id")


def _messages(row: pd.Series) -> tuple[list[dict], int]:
    try:
        raw = (
            json.loads(row.messages_json)
            if isinstance(row.messages_json, str)
            else row.messages_json
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid log JSON for {row.sample_id}") from error
    if not isinstance(raw, list):
        raise ValueError(f"Invalid log messages for {row.sample_id}")
    merged = {}
    excluded = 0
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("message"), str)
            or type(item.get("count")) is not int
            or item["count"] <= 0
        ):
            raise ValueError(f"Invalid log message/count for {row.sample_id}")
        text, error = _clean(item["message"])
        if not text:
            excluded += item["count"]
            continue
        message = merged.setdefault(text, dict(text=text, count=0, error=False))
        message["count"] += item["count"]
        message["error"] |= error
    return [merged[text] for text in sorted(merged)], excluded


def _snapshot(miners: dict) -> dict:
    return {
        service: [
            dict(cluster_id=cluster.cluster_id, template=cluster.get_template(), size=cluster.size)
            for cluster in sorted(miner.drain.clusters, key=lambda cluster: cluster.cluster_id)
        ]
        for service, miner in sorted(miners.items())
    }


def _state_payload(state: LogState) -> dict:
    return dict(
        templates=state.templates,
        config=state.config,
        fit_ids=state.fit_ids,
        fit_input_hash=state.fit_input_hash,
        snapshot_hash=state.snapshot_hash,
    )


def fit_logs(tables: InputTables, fit_ids: set[str], config: dict) -> LogState:
    resolved = _log_config(config)
    frame = _frame(tables)
    _check_fit_ids(tables, fit_ids, set(frame.sample_id))
    try:
        miner_class = importlib.import_module("drain3.template_miner").TemplateMiner
        config_class = importlib.import_module("drain3.template_miner_config").TemplateMinerConfig
    except ImportError as error:
        raise RuntimeError(
            "Drain3 is required for log template fitting; no substitute parser is allowed"
        ) from error
    drain_config = config_class()
    drain_config.masking_instructions = []
    miners = {}
    fit_records = []
    characters = []
    for _, row in frame[frame.sample_id.isin(fit_ids)].iterrows():
        quality = str(row.quality_status)
        messages, excluded = _messages(row) if quality == "OBSERVED" else ([], 0)
        fit_records.append(
            dict(
                sample_id=row.sample_id,
                service=str(row.canonical_service_id),
                quality=quality,
                messages=messages,
                excluded=excluded,
            )
        )
        if not messages:
            continue
        service = str(row.canonical_service_id)
        if service not in miners:
            service_config = config_class()
            service_config.masking_instructions = []
            miners[service] = miner_class(config=service_config)
        for message in messages:
            miners[service].add_log_message(message["text"])
            characters.append(len(message["text"]))
    templates = {
        service: tuple(sorted(cluster.get_template() for cluster in miner.drain.clusters))
        for service, miner in sorted(miners.items())
    }
    try:
        parser_version = version("drain3")
    except PackageNotFoundError:
        parser_version = None
    parser_sources = {
        module_name: file_digest(Path(importlib.import_module(f"drain3.{module_name}").__file__))
        for module_name in [
            "template_miner",
            "template_miner_config",
            "drain",
            "masking",
            "simple_profiler",
        ]
    }
    parser_config = dict(
        parser="drain3",
        parser_version=parser_version,
        parser_version_status=(
            "distribution_metadata" if parser_version else "distribution_metadata_unavailable"
        ),
        parser_sources_sha256=parser_sources,
        drain=vars(drain_config).copy(),
        match_full_search_strategy="never",
        cleaning_version=CLEANING_VERSION,
        log_config=resolved,
        fit_message_diagnostics=dict(
            distinct_messages_fed=len(characters),
            max_characters=max(characters, default=0),
            messages_over_2000_characters=sum(length > 2000 for length in characters),
            character_truncation=False,
        ),
    )
    state = LogState(
        templates,
        parser_config,
        tuple(sorted(fit_ids)),
        content_hash(dict(records=fit_records)),
        content_hash(_snapshot(miners)),
        "",
        miners,
    )
    return LogState(
        templates,
        parser_config,
        state.fit_ids,
        state.fit_input_hash,
        state.snapshot_hash,
        content_hash(_state_payload(state)),
        miners,
    )


def _tokenizer_identity(tokenizer: object) -> dict:
    required = [
        "encode",
        "num_special_tokens_to_add",
        "build_inputs_with_special_tokens",
        "get_special_tokens_mask",
        "get_vocab",
    ]
    if any(not callable(getattr(tokenizer, name, None)) for name in required):
        raise TypeError(
            "The tokenizer must expose encoding, special-token construction/masking and vocabulary APIs"
        )
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("The tokenizer vocabulary must be nonempty")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    metadata = json.loads(
        json.dumps(
            dict(
                tokenizer_class=f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                name_or_path=getattr(tokenizer, "name_or_path", None),
                init_kwargs=getattr(tokenizer, "init_kwargs", {}),
                special_tokens_map=getattr(tokenizer, "special_tokens_map", {}),
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                padding_side=getattr(tokenizer, "padding_side", "right"),
                backend=backend.to_str() if backend is not None else None,
            ),
            default=str,
        )
    )
    metadata["vocabulary_sha256"] = content_hash(vocab)
    return metadata


def _tokenize(tokenizer: object, text: str, maximum: int) -> dict | None:
    body = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if not isinstance(body, list) or any(
        not isinstance(token, Integral) or isinstance(token, bool) or token < 0 for token in body
    ):
        raise ValueError("The tokenizer returned invalid body token IDs")
    if not body:
        return None
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    if type(special_count) is not int or not 0 <= special_count < maximum:
        raise ValueError("The tokenizer special tokens leave no body budget")
    budget = maximum - special_count
    truncated = len(body) > budget
    if truncated:
        head = math.ceil(2 * budget / 3)
        tail = budget - head
        body = body[:head] + (body[-tail:] if tail else [])
    ids = list(tokenizer.build_inputs_with_special_tokens(body))
    special = list(tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True))
    body_special = list(tokenizer.get_special_tokens_mask(body, already_has_special_tokens=True))
    if (
        len(ids) != len(body) + special_count
        or len(special) != len(ids)
        or len(body_special) != len(body)
        or any(value not in (0, 1) for value in special)
        or any(value not in (0, 1) for value in body_special)
        or [token for token, masked in zip(ids, special) if not masked]
        != [token for token, masked in zip(body, body_special) if not masked]
    ):
        raise ValueError("The tokenizer special-token construction/mask is inconsistent")
    if all(special):
        return None
    padding = maximum - len(ids)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(pad_id, Integral) or isinstance(pad_id, bool) or pad_id < 0:
        raise ValueError("The tokenizer requires a valid pad_token_id")
    return dict(
        input_ids=ids + [int(pad_id)] * padding,
        attention_mask=[1] * len(ids) + [0] * padding,
        special_mask=special + [1] * padding,
        truncated=truncated,
    )


def _select(messages: list[dict], config: dict) -> list[dict]:
    selected = []
    selected_texts = set()
    group_counts = Counter()

    def take(candidates: list[dict], quota: int) -> None:
        remaining = {
            message["text"]: message
            for message in candidates
            if message["text"] not in selected_texts
        }
        for _ in range(min(quota, config["max_messages"] - len(selected))):
            if not remaining:
                break
            message = min(
                remaining.values(),
                key=lambda item: (group_counts[item["group"]], item["group"], item["text"]),
            )
            selected.append(message)
            selected_texts.add(message["text"])
            group_counts[message["group"]] += 1
            del remaining[message["text"]]

    take([message for message in messages if message["error"]], config["error_quota"])
    take([message for message in messages if not message["matched"]], config["unmatched_quota"])
    take(messages, config["max_messages"])
    return selected


def _source_count(row: pd.Series) -> int | None:
    value = row.get("event_count")
    if value is None or pd.isna(value):
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, Integral))
        or not math.isfinite(value)
        or value < 0
        or int(value) != value
    ):
        raise ValueError("Invalid source log event_count")
    return int(value)


def prepare_logs(tables: InputTables, state: LogState, tokenizer: object, config: dict) -> dict:
    resolved = _log_config(config)
    frame = _frame(tables)
    tokenizer_identity = _tokenizer_identity(tokenizer)
    if (
        content_hash(_snapshot(state.miners)) != state.snapshot_hash
        or content_hash(_state_payload(state)) != state.state_hash
    ):
        raise ValueError("Frozen Drain3 template state changed")
    output = {
        name: {}
        for name in [
            "input_ids",
            "attention_mask",
            "special_mask",
            "group_ids",
            "texts",
            "counts",
            "masks",
            "coverage",
            "stats",
        ]
    }
    tokens_by_text = {}
    groups_by_text = {}
    input_records = []
    for _, row in frame.iterrows():
        sample_id = row.sample_id
        service = str(row.canonical_service_id)
        quality = str(row.quality_status)
        messages, cleaning_excluded = _messages(row) if quality == "OBSERVED" else ([], 0)
        source_count = _source_count(row)
        input_records.append(
            dict(
                sample_id=sample_id,
                service=service,
                quality=quality,
                messages=messages,
                cleaning_excluded=cleaning_excluded,
                source_event_count=source_count,
            )
        )
        available = []
        token_excluded = 0
        for message in messages:
            text = message["text"]
            if text not in tokens_by_text:
                tokens_by_text[text] = _tokenize(tokenizer, text, resolved["max_tokens"])
            tokens = tokens_by_text[text]
            if tokens is None:
                token_excluded += message["count"]
                continue
            key = (service, text)
            if key not in groups_by_text:
                miner = state.miners.get(service)
                cluster = (
                    miner.match(text, full_search_strategy="never") if miner is not None else None
                )
                groups_by_text[key] = (
                    cluster is not None,
                    (
                        "template:" + cluster.get_template()
                        if cluster is not None
                        else "unmatched:" + text
                    ),
                )
            matched, group = groups_by_text[key]
            available.append({**message, "matched": matched, "group": group, "tokens": tokens})
        selected = _select(available, resolved)
        total = sum(message["count"] for message in available)
        matched_counts = Counter()
        for message in available:
            if message["matched"]:
                matched_counts[message["group"]] += message["count"]
        selected_events = sum(message["count"] for message in selected)
        if selected:
            reason = "observed"
        elif quality == "NO_OBSERVATIONS_COLLECTION_PRESENT":
            reason = "no_events_collection_present"
        elif quality == "COLLECTION_MISSING":
            reason = "collection_missing"
        elif quality != "OBSERVED":
            reason = "source_quality_invalid"
        else:
            reason = "messages_unusable"
        for name in ["input_ids", "attention_mask", "special_mask"]:
            output[name][sample_id] = [list(message["tokens"][name]) for message in selected]
        output["group_ids"][sample_id] = [message["group"] for message in selected]
        output["texts"][sample_id] = [message["text"] for message in selected]
        output["counts"][sample_id] = len(selected)
        output["masks"][sample_id] = bool(selected)
        output["stats"][sample_id] = dict(
            event_count=total,
            matched_template_count=len(matched_counts),
            max_matched_fraction=max(matched_counts.values(), default=0) / total if total else None,
            unmatched_count=sum(
                message["count"] for message in available if not message["matched"]
            ),
        )
        output["coverage"][sample_id] = dict(
            reason=reason,
            source_quality=quality,
            source_event_count=source_count,
            source_event_count_difference=(
                source_count - total if source_count is not None else None
            ),
            cleaning_excluded_event_count=cleaning_excluded,
            token_excluded_event_count=token_excluded,
            excluded_event_count=cleaning_excluded + token_excluded,
            quality_excluded_event_count=source_count if quality != "OBSERVED" else 0,
            available_event_count=total,
            available_message_count=len(available),
            available_group_count=len({message["group"] for message in available}),
            selected_message_count=len(selected),
            selected_event_count=selected_events,
            unselected_message_count=len(available) - len(selected),
            unselected_event_count=total - selected_events,
            selected_group_count=len({message["group"] for message in selected}),
            selected_matched_template_count=len(
                {message["group"] for message in selected if message["matched"]}
            ),
            selected_event_fraction=selected_events / total if total else None,
            selected_message_fraction=len(selected) / len(available) if available else None,
            selected_group_fraction=(
                len({message["group"] for message in selected})
                / len({message["group"] for message in available})
                if available
                else None
            ),
            error_message_count=sum(message["error"] for message in available),
            selected_error_message_count=sum(message["error"] for message in selected),
            token_truncated_message_count=sum(
                message["tokens"]["truncated"] for message in available
            ),
            selected_token_truncated_message_count=sum(
                message["tokens"]["truncated"] for message in selected
            ),
        )
    if content_hash(_snapshot(state.miners)) != state.snapshot_hash:
        raise RuntimeError("Drain3 matching modified the frozen templates")
    cache = dict(
        version=2,
        cleaning_version=CLEANING_VERSION,
        tokenization_version=TOKENIZATION_VERSION,
        state_hash=state.state_hash,
        fit_ids=list(state.fit_ids),
        tokenizer=tokenizer_identity,
        log_config=resolved,
        configuration_sha256=content_hash(config),
        inputs_sha256=content_hash(dict(records=input_records)),
        tokenized=True,
    )
    cache["cache_key"] = content_hash(cache)
    output["cache"] = cache
    return output


def _statistics_config(config: dict) -> dict:
    prep = config.get("preprocessing", {})
    count_fields = ["event_count", "matched_template_count", "unmatched_count"]
    resolved = dict(
        log1p=[field for field in LOG_STAT_FIELDS if field in prep.get("log1p", count_fields)],
        std_ddof=prep.get("std_ddof", 0),
        scale_epsilon=prep.get("scale_epsilon", 1e-6),
        fallback_scale=prep.get("fallback_scale", 1.0),
    )
    if (
        resolved["std_ddof"] != 0
        or resolved["fallback_scale"] != 1
        or not np.isfinite(resolved["scale_epsilon"])
        or resolved["scale_epsilon"] <= 0
    ):
        raise ValueError("Invalid log statistics preprocessing configuration")
    return resolved


def _statistics_binding(logs: dict) -> dict:
    cache = logs["cache"]
    return dict(
        log_state_hash=cache["state_hash"],
        cleaning_version=cache["cleaning_version"],
        tokenization_version=cache["tokenization_version"],
        tokenizer_sha256=content_hash(cache["tokenizer"]),
    )


def _statistics_frames(
    tables: InputTables, logs: dict, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    frame = _frame(tables).set_index("sample_id")
    ids = set(frame.index)
    if set(logs["stats"]) != ids or set(logs["masks"]) != ids:
        raise ValueError("Log statistics/full grid sample IDs mismatch")
    if any(set(record) != set(LOG_STAT_FIELDS) for record in logs["stats"].values()):
        raise ValueError("Invalid log statistics fields")
    available = pd.Series(logs["masks"]).reindex(frame.index)
    if not pd.api.types.is_bool_dtype(available) or available.isna().any():
        raise ValueError("Invalid log statistics source masks")
    values = pd.DataFrame.from_dict(logs["stats"], orient="index").reindex(
        index=frame.index, columns=LOG_STAT_FIELDS
    )
    masks = pd.DataFrame(index=frame.index)
    for name in LOG_STAT_FIELDS:
        raw = pd.to_numeric(values[name], errors="coerce")
        valid = available & frame.quality_status.eq("OBSERVED") & raw.ge(0) & np.isfinite(raw)
        if name == "max_matched_fraction":
            valid &= raw.le(1)
        values[name] = raw.where(valid)
        if name in config["log1p"]:
            values[name] = np.log1p(values[name])
        masks[name] = valid.astype(bool)
    return values, masks, frame.canonical_service_id.astype(str)


def fit_log_statistics(
    tables: InputTables, logs: dict, fit_ids: set[str], config: dict
) -> LogStatisticsState:
    """Fit full-window statistics after the common fit membership is frozen."""
    resolved = _statistics_config(config)
    values, _, services = _statistics_frames(tables, logs, resolved)
    _check_fit_ids(tables, fit_ids, set(values.index))
    fit = values.loc[sorted(fit_ids)]
    fields = {}
    epsilon = resolved["scale_epsilon"]
    for name in LOG_STAT_FIELDS:
        observed = fit[name].dropna()
        global_std = float(observed.std(ddof=0)) if len(observed) else 0.0
        global_scale = global_std if global_std > epsilon else 1.0
        parameters = {}
        for service in sorted(services.unique()):
            subset = fit.loc[services.reindex(fit.index).eq(service), name].dropna()
            service_std = float(subset.std(ddof=0)) if len(subset) else 0.0
            parameters[service] = dict(
                mean=float(subset.mean()) if len(subset) else None,
                scale=service_std if service_std > epsilon else global_scale,
                scale_source=(
                    "service"
                    if service_std > epsilon
                    else "global" if global_std > epsilon else "unit"
                ),
                available=bool(len(subset)),
                fit_count=len(subset),
            )
        fields[name] = dict(
            services=parameters,
            global_mean=float(observed.mean()) if len(observed) else None,
            global_scale=global_scale,
            global_fit_count=len(observed),
        )
    fit_records = [
        dict(
            sample_id=sample_id,
            service=services.at[sample_id],
            values=[None if pd.isna(value) else float(value) for value in fit.loc[sample_id]],
        )
        for sample_id in fit.index
    ]
    payload = dict(
        fields=fields,
        fit_ids=tuple(sorted(fit_ids)),
        config=resolved,
        source_binding=_statistics_binding(logs),
        fit_input_hash=content_hash(dict(records=fit_records)),
    )
    return LogStatisticsState(**payload, state_hash=content_hash(payload))


def transform_log_statistics(tables: InputTables, logs: dict, state: LogStatisticsState) -> dict:
    """Transform using fit-only service means; fill masked placeholders with zero."""
    payload = dict(
        fields=state.fields,
        fit_ids=state.fit_ids,
        config=state.config,
        source_binding=state.source_binding,
        fit_input_hash=state.fit_input_hash,
    )
    if (
        content_hash(payload) != state.state_hash
        or _statistics_binding(logs) != state.source_binding
    ):
        raise ValueError("Log statistics state/source binding mismatch")
    values, masks, services = _statistics_frames(tables, logs, state.config)
    for name in LOG_STAT_FIELDS:
        for sample_id in values.index:
            parameters = state.fields[name]["services"].get(services.at[sample_id])
            if parameters is None or not parameters["available"]:
                masks.at[sample_id, name] = False
            elif masks.at[sample_id, name]:
                values.at[sample_id, name] = (
                    values.at[sample_id, name] - parameters["mean"]
                ) / parameters["scale"]
    values = values.where(masks, 0.0)
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite normalized log statistics")
    return dict(
        values=values.reset_index(),
        masks=masks.reset_index(),
        state_hash=state.state_hash,
        feature_order=list(LOG_STAT_FIELDS),
    )
