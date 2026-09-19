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

    # per_case_endpoint_z_score / per_case_service_z_score 只被 v2 build 路径消费
    # （contract_version == "v2" 的 per-case 归一化，见 Task 11/14）。双向组合
    # 守卫均在 ContractConfig.__post_init__ 加载期执行：v2 配 min_max 直接拒绝
    # （is_v2 路径跳过 clip 与退化告警）；v0/v1 配 per-case z-score 也拒绝——
    # 实测（v2_fit_scope_mini fixture + v1 config 实跑 build_contract.py）该组合
    # 在归一化阶段抛裸 KeyError: 'case_id'（per-case 分组键含 case_id，v1 路径的
    # group_cols 不含该列），并非"静默跑通后被 clip 截断"；守卫把裸 KeyError
    # 换成带说明的 ValueError。
    _VALID_NORMALIZATIONS: ClassVar[frozenset[str]] = frozenset(
        {
            "per_endpoint_min_max",
            "per_service_min_max",
            "global_min_max",
            "per_case_endpoint_z_score",
            "per_case_service_z_score",
        }
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
    # fault_baseline_train_fraction 控制；838→~1787 行是 entry 016 时点数字，已随
    # 数据集变化 drift，不再准确，见 issue #16 / history/entries/016）。默认 False 保持 Task 6
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
    # 的 case 有精确含义：case 级标签的 case 没有 target_endpoint 字段，
    # is_target_endpoint 在真实数据里全为 0.0 或 NaN（缺列时 trace_preprocessor.py
    # 填 False，Normal case 该列存在但全 NaN，两条不同路径），无法区分目标/非目标，
    # 其 recover 行整段排除在本切分外、原样留在 eval_all（见 _write_v1）。默认 0.0
    # （不吸收，向后兼容）。
    fault_recover_nontarget_train_fraction: float = 0.0

    # 两个 per-case z-score 值是 v2 专属，v2 下每个 modality 还必须与自己的
    # 聚合 scope 正确配对（组合键决定 z 值在哪些窗之间计算，错配不报错但算错）。
    _PER_CASE_Z_SCORE_NORMALIZATIONS: ClassVar[frozenset[str]] = frozenset(
        {"per_case_endpoint_z_score", "per_case_service_z_score"}
    )
    # v2 下 modality → 唯一合法 normalization。endpoint_red 的特征是 per-endpoint
    # 粒度，必须按 (case_id, endpoint_key) 聚合；service_metric/service_log 只有
    # service 粒度，按 (case_id, service_name) 聚合。当前数据集 service↔endpoint
    # 1:1，错配数字不变；未来一个 service 多 endpoint 时错配会静默把不同 endpoint
    # 的窗混在一起算 z 值，故在加载期按配对表拒绝。
    _V2_MODALITY_NORMALIZATION: ClassVar[dict[str, str]] = {
        "endpoint_red": "per_case_endpoint_z_score",
        "service_metric": "per_case_service_z_score",
        "service_log": "per_case_service_z_score",
    }

    def __post_init__(self) -> None:
        # 配置加载期守卫（早于 build）。两个方向的错误组合都不是"立即崩溃"型，
        # 而是静默产出错误数据，故必须在这里指名道姓地拒绝。
        if self.contract_version == "v2":
            for mod_name, spec in self.modalities.items():
                expected = self._V2_MODALITY_NORMALIZATION.get(mod_name)
                if spec.normalization not in self._PER_CASE_Z_SCORE_NORMALIZATIONS:
                    # v2×min_max：is_v2 路径整体跳过 rate clip 与退化 group 告警
                    # （设计文档 §4.3：clip 治的是 min-max 跨批次参照系错位的症状），
                    # min_max 退化组的原始量纲会无告警、无裁剪地进入模型。
                    raise ValueError(
                        "contract_version='v2' 要求所有 modality 使用 per-case z-score 归一化"
                        f"（{sorted(self._PER_CASE_Z_SCORE_NORMALIZATIONS)}），"
                        f"但 modality {mod_name!r} 配的是 {spec.normalization!r}："
                        "v2 路径会跳过 clip(0,1) 与退化 group 告警，min_max 退化组的"
                        "原始量纲会无告警进入模型（见设计文档 §4.3）"
                    )
                if expected is None or spec.normalization != expected:
                    expected_desc = (
                        expected
                        if expected is not None
                        else f"未登记的 modality，无法判定 scope 配对（已知配对：{self._V2_MODALITY_NORMALIZATION}）"
                    )
                    raise ValueError(
                        f"contract_version='v2' 下 modality {mod_name!r} 的 normalization "
                        f"必须是 {expected_desc}，实际配的是 {spec.normalization!r}："
                        "endpoint 粒度与 service 粒度特征必须各按自己的 (case_id, key) "
                        "组合键聚合，错配会静默按错误的分组算 z 值"
                    )
        elif self.contract_version in ("v0", "v1"):
            # 对称守卫：这两个值仅 v2 可用。实测 v1 config 配 per-case z-score 并非
            # "静默跑通后被 clip 截断"，而是在归一化阶段抛裸 KeyError: 'case_id'——
            # per-case scope 的分组键含 case_id，而 v0/v1 路径传给
            # Normalizer.transform 的 group_cols 不含该列。在这里换成可读错误。
            for mod_name, spec in self.modalities.items():
                if spec.normalization in self._PER_CASE_Z_SCORE_NORMALIZATIONS:
                    raise ValueError(
                        f"normalization {spec.normalization!r} 仅 contract_version='v2' 可用，"
                        f"modality {mod_name!r} 在 {self.contract_version!r} 配置里使用了它："
                        "v0/v1 路径的归一化分组不含 case_id，构建会在归一化阶段抛裸 "
                        "KeyError: 'case_id'；请改用 per_endpoint/per_service/global min_max"
                    )

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

    def endpoint_baseline_columns(self) -> tuple[list[str], list[str]]:
        """EndpointBaselineStats 消费的 (red_cols, svc_cols)，从 modalities 声明派生。

        必须从这里派生、不能按 dataframe 列名前缀扫描——预处理器可能产出比 contract
        声明更多的原始列，按前缀扫描会把未声明列也吸收进来，与 schema.json 的
        feature_groups（同样严格按 modalities 声明派生）产生静默维度错位。
        """
        red_cols = [f"endpoint_red__{f}" for f in self.modalities["endpoint_red"].features]
        svc_cols = [f"service_metric__{f}" for f in self.modalities["service_metric"].features] + [
            f"service_log__{f}" for f in self.modalities["service_log"].features
        ]
        return red_cols, svc_cols


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
