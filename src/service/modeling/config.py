"""Resolve explicit model defaults without touching data or model weights."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs/service/tt_model_v1.json"


def content_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _merge(defaults: dict, overrides: dict) -> dict:
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            raise ValueError(f"Unknown configuration key: {key}")
        if key == "auxiliary_losses":
            _validate_structure(value, {}, "configuration.model.auxiliary_losses")
            result[key] = copy.deepcopy(value)
            continue
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected configuration mapping: {key}")
            result[key] = _merge(defaults[key], value)
        else:
            result[key] = value
    return result


def _validate_structure(value: object, schema: object, path: str = "configuration") -> None:
    if path == "configuration.model.auxiliary_losses":
        if value == {}:
            return
        if not isinstance(value, dict) or set(value) != {"reconstruction"}:
            raise ValueError("Unsupported auxiliary objective")
        weight = value["reconstruction"]
        if type(weight) not in {int, float} or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Reconstruction weight must be finite and positive")
        return
    if isinstance(schema, dict):
        if not isinstance(value, dict) or set(value) != set(schema):
            raise ValueError(f"Incomplete or unknown configuration keys: {path}")
        for key, expected in schema.items():
            _validate_structure(value[key], expected, f"{path}.{key}")
    elif isinstance(schema, list):
        if not isinstance(value, list):
            raise ValueError(f"Expected configuration list: {path}")
        if schema:
            for item in value:
                _validate_structure(item, schema[0], path)
    elif schema is None:
        if value is not None and not isinstance(value, str):
            raise ValueError(f"Expected nullable configuration string: {path}")
    elif isinstance(schema, float):
        if type(value) not in {int, float}:
            raise ValueError(f"Expected configuration number: {path}")
    elif type(value) is not type(schema):
        raise ValueError(f"Invalid configuration type: {path}")


def validate_config(config: dict) -> None:
    defaults = json.loads(DEFAULT_CONFIG.read_text())
    if not isinstance(config, dict) or "config_sha256" not in config:
        raise ValueError("Missing resolved configuration hash")
    resolved = {key: value for key, value in config.items() if key != "config_sha256"}
    _validate_structure(resolved, defaults)
    if not isinstance(config["config_sha256"], str) or config["config_sha256"] != content_hash(
        resolved
    ):
        raise ValueError("Configuration hash mismatch")
    for key in [
        "schema_version",
        "dataset",
        "preparation_schema_version",
        "label_schema_version",
        "window_ms",
        "features",
    ]:
        if config.get(key) != defaults[key]:
            raise ValueError(f"Unsupported input configuration: {key}")
    for key in ["baselines", "calibration"]:
        if config[key] != defaults[key]:
            raise ValueError(f"Unsupported fixed protocol configuration: {key}")
    for key in ["std_ddof", "fallback_scale"]:
        if config["preprocessing"][key] != defaults["preprocessing"][key]:
            raise ValueError(f"Unsupported fixed preprocessing configuration: {key}")


def load_config(path: Path) -> dict:
    config = _merge(json.loads(DEFAULT_CONFIG.read_text()), json.loads(Path(path).read_text()))
    config["config_sha256"] = content_hash(config)
    validate_config(config)
    return config
