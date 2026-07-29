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
    # v1 专属：是否把故障 case 的 baseline 阶段行部分吸收进训练池（吸收比例由下方
    # fault_baseline_train_fraction 控制，默认吸收后训练池 838→~1787 行，其余 baseline
    # 窗口留在 eval_all，见 issue #16 / history/entries/016）。默认 False 保持 Task 6
    # 之前的行为（train=纯 train_fit，eval_all 含全部故障阶段），与 entry 012 的既有
    # 实验数字可比；RG 专属实验用 v1_expanded_pool.yaml 显式打开。v0 不消费此字段。
    expand_train_pool: bool = False
    # v1 专属(RG 消费):是否 fit 并落盘 per-endpoint 双分支(ep/svc)normal-only
    # 基线统计量(endpoint_baseline_stats.json)，并同步产出 endpoint_id 列。默认
    # False——L0/L1/L2 对比基线不需要这份统计量，不产出可避免 build_contract_v1
    # 无谓多算一遍并少一个产物依赖。ReliabilityGatedFusion 用 v1_expanded_pool.yaml
    # 显式打开。v0 不消费此字段(走默认 False)。取值与 expand_train_pool 相互独立。
    fit_endpoint_baseline_stats: bool = False
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 baseline
    # 阶段行按时间窗时序切分时，最早 fraction 比例的窗口进训练池，其余留在 eval_all。
    # 默认 0.2——真实 29-case 数据实测 eval_all 正负比（is_endpoint_anomaly 口径）约
    # 24.62%:75.38%，较修复前的 ~84:16 有实质改善，但未完全回到原始 ~50:50（规划阶段
    # 的 45:55 预估未按 is_endpoint_anomaly 的 endpoint 级粒度折算，偏乐观，见
    # history/entries/016）。expand_train_pool=false 时该字段被忽略（但仍参与下方
    # __post_init__ 范围校验，配错值一律在加载期报错）。
    fault_baseline_train_fraction: float = 0.2
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 inject
    # 阶段"非目标 endpoint"行按时间窗时序切分，最早 fraction 比例的窗口进训练池，
    # 其余留在 eval_all。判据是 is_endpoint_anomaly == False，不是 is_target_endpoint
    # ——前者天然兼容两种 label_granularity：case 级标签下 is_endpoint_anomaly
    # fallback 等于 is_anomaly，inject 阶段恒 True，该判据自动选不出任何行，
    # 不会把正样本误吸收进训练池。默认 0.0（不吸收，向后兼容本字段引入前的行为）。
    fault_inject_nontarget_train_fraction: float = 0.0
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 recover
    # 阶段"非目标 endpoint"行按时间窗时序切分。判据必须是 is_target_endpoint == False
    # 而非 is_endpoint_anomaly——is_anomaly 定义为 phase == "inject"，recover 阶段
    # 恒 False，导致 is_endpoint_anomaly 对 recover 阶段所有 endpoint（含目标）恒为
    # False，拿它筛"非目标"是空操作。且该判据只对 label_granularity == "endpoint"
    # 的 case 有精确含义：case 级标签的 case 没有 target_endpoint 字段、
    # is_target_endpoint 全填 False，无法区分目标/非目标，其 recover 行整段排除在
    # 本切分外、原样留在 eval_all（见 _write_v1）。默认 0.0（不吸收，向后兼容）。
    fault_recover_nontarget_train_fraction: float = 0.0

    def __post_init__(self) -> None:
        # 恒校验（不看 expand_train_pool）：越界/非数值比例是配置错误，即便当前未被
        # 消费也应在加载期暴露，而不是等到 expand_train_pool 被打开时才在
        # split_fault_phase_temporal 内部产出反直觉结果——大于 1.0 的正数会让 train
        # 吞下全部窗口而不报错，负数触发 Python 负索引切片导致方向反转的错误切分（都
        # 不是"崩溃"而是"看起来合理但错误的数据"）；YAML 里若误加引号（字符串类型）
        # 则会在比较运算处抛出与本意无关的 TypeError，因此类型检查须先于范围检查。
        for field_name in (
            "fault_baseline_train_fraction",
            "fault_inject_nontarget_train_fraction",
            "fault_recover_nontarget_train_fraction",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{field_name} 必须是数值类型，"
                    f"实际类型 {type(value).__name__}"
                    f"（值={value!r}，检查 YAML 中是否误加了引号）"
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} 必须落在 [0.0, 1.0]，实际 {value}")


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
        fault_inject_nontarget_train_fraction=raw.get("fault_inject_nontarget_train_fraction", 0.0),
        fault_recover_nontarget_train_fraction=raw.get(
            "fault_recover_nontarget_train_fraction", 0.0
        ),
    )
