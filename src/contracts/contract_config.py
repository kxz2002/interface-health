from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import yaml


@dataclass(frozen=True)
class ModalitySpec:
    preprocessor: str
    preprocessor_version: str
    features: tuple[str, ...]
    normalization: str
    candidates_pool: tuple[str, ...] = ()

    _VALID_NORMALIZATIONS: ClassVar[frozenset[str]] = frozenset(
        {"per_endpoint_min_max", "per_service_min_max", "global_min_max"}
    )

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("ModalitySpec.features 不能为空")
        if self.normalization not in self._VALID_NORMALIZATIONS:
            raise ValueError(
                f"未知 normalization: {self.normalization!r}，"
                f"支持 {sorted(self._VALID_NORMALIZATIONS)}"
            )


@dataclass(frozen=True)
class ContractConfig:
    contract_version: str
    window_size_s: int
    modalities: dict[str, ModalitySpec]
    # v1 专属：是否把故障 case 的 baseline 阶段行吸收进训练池（838→~6249 行）。
    # 默认 False 保持 Task 6 之前的行为（train=纯 train_fit，eval_all 含全部
    # 故障阶段），与 entry 012 的既有实验数字可比；RG 专属实验用
    # v1_expanded_pool.yaml 显式打开。v0 不消费此字段。
    expand_train_pool: bool = False


def load_contract_config(path: str | Path) -> ContractConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if "modalities" not in raw:
        raise ValueError("contract config 缺少 'modalities' 字段")
    modalities = {
        name: ModalitySpec(
            **{
                **spec,
                "features": tuple(spec.get("features", [])),
                "candidates_pool": tuple(spec.get("candidates_pool", [])),
            }
        )
        for name, spec in raw["modalities"].items()
    }
    return ContractConfig(
        contract_version=raw["contract_version"],
        window_size_s=raw["window_size_s"],
        modalities=modalities,
        expand_train_pool=raw.get("expand_train_pool", False),
    )
