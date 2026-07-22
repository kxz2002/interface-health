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
    # v1 专属(RG 消费):是否 fit 并落盘 per-endpoint 双分支(ep/svc)normal-only
    # 基线统计量(endpoint_baseline_stats.json)，并同步产出 endpoint_id 列。默认
    # False——L0/L1/L2 对比基线不需要这份统计量，不产出可避免 build_contract_v1
    # 无谓多算一遍并少一个产物依赖。ReliabilityGatedFusion 用 v1_expanded_pool.yaml
    # 显式打开。v0 不消费此字段(走默认 False)。取值与 expand_train_pool 相互独立。
    fit_endpoint_baseline_stats: bool = False
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 baseline
    # 阶段行按时间窗时序切分时，最早 fraction 比例的窗口进训练池，其余留在 eval_all。
    # 默认 0.2——把 eval_all 正负比从扩容导致的 ~84:16 拉回接近原始 ~50:50，同时训练池
    # 仍有实质扩容（见 issue #16 / history/entries/016）。expand_train_pool=false 时
    # 该字段被忽略（但仍参与下方 __post_init__ 范围校验，配错值一律在加载期报错）。
    fault_baseline_train_fraction: float = 0.2

    def __post_init__(self) -> None:
        # 恒校验（不看 expand_train_pool）：越界比例是配置错误，即便当前未被消费也应
        # 在加载期暴露，而不是等到 expand_train_pool 被打开时才悄悄产出空/全量切分。
        if not 0.0 <= self.fault_baseline_train_fraction <= 1.0:
            raise ValueError(
                "fault_baseline_train_fraction 必须落在 [0.0, 1.0]，"
                f"实际 {self.fault_baseline_train_fraction}"
            )


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
        fit_endpoint_baseline_stats=raw.get("fit_endpoint_baseline_stats", False),
        fault_baseline_train_fraction=raw.get("fault_baseline_train_fraction", 0.2),
    )
