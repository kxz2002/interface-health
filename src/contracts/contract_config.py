from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModalitySpec:
    preprocessor: str
    preprocessor_version: str
    features: list[str]
    normalization: str
    candidates_pool: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContractConfig:
    contract_version: str
    window_size_s: int
    modalities: dict[str, ModalitySpec]


def load_contract_config(path: str | Path) -> ContractConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if "modalities" not in raw:
        raise ValueError("contract config 缺少 'modalities' 字段")
    modalities = {name: ModalitySpec(**spec) for name, spec in raw["modalities"].items()}
    return ContractConfig(
        contract_version=raw["contract_version"],
        window_size_s=raw["window_size_s"],
        modalities=modalities,
    )
