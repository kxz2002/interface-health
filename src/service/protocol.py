"""service_minimal_v1 冻结协议：实体解析、网格/阶段、Schema 校验与输出保护。

设计目标（T0）：让错误实体、错误信息预算或非法输出路径在数据处理之前失败。
本模块不读取 metric/log 原始序列（T1/T2 才做），只消费冻结配置与 case 元数据。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.contracts.split_v1 import split_normal_rows_temporal
from src.data.dataset_config import load_dataset_config

LOG = logging.getLogger(__name__)

PROTOCOL_ID = "service_minimal_v1"
# 与 configs/service/minimal_v1.yaml 的 window_ms 镜像：load_protocol 校验两者一致，
# make_grid/validate_features 直接引用本常量，避免代码与配置漂移
WINDOW_MS = 15_000
CASE_ROLES = ("normal", "cpu", "kill")
REVIEW_STATUS_ENUM = ("pending_human_review", "confirmed", "rejected")
# 每个 case 条目允许的键集合（per-case 封闭，未知键报错）
CASE_KEYS = frozenset(
    {
        "case_id",
        "role",
        "is_anomaly_case",
        "window_start_ms",
        "window_end_ms",
        "inject_start_ms",
        "inject_end_ms",
        "target_service",
        "recoverable",
        "chaos_kind",
        "chaos_yaml",
    }
)

FROZEN_CASE_IDS = (
    "Normal_20260728T161044Z",
    "Lv_P_CPU_preserve_20260728T164711Z_em",
    "Lv_S_KILLPOD_order_20260728T203705Z_em",
    "Lv_S_KILLPOD_preserve_20260728T200124Z_em",
)
FROZEN_CASE_IDENTITIES = dict(
    zip(
        FROZEN_CASE_IDS,
        (
            ("normal", False, None),
            ("cpu", True, "ts-preserve-service"),
            ("kill", True, "ts-order-service"),
            ("kill", True, "ts-preserve-service"),
        ),
    )
)
FROZEN_SERVICES = (
    "ts-assurance-service",
    "ts-auth-service",
    "ts-basic-service",
    "ts-cancel-service",
    "ts-config-service",
    "ts-contacts-service",
    "ts-execute-service",
    "ts-gateway-service",
    "ts-inside-payment-service",
    "ts-notification-service",
    "ts-order-other-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-preserve-other-service",
    "ts-preserve-service",
    "ts-price-service",
    "ts-rebook-service",
    "ts-route-service",
    "ts-seat-service",
    "ts-security-service",
    "ts-station-service",
    "ts-train-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-user-service",
)
FROZEN_METHODS = ("L2", "max", "OCSVM", "IF")
FEATURE_GROUPS = ("R3", "CPU-only")

FEATURE_VALUE_COLUMNS = (
    "cpu_cores",
    "memory_usage_ratio",
    "process_count",
    "memory_usage_bytes",
    "memory_working_set_bytes",
    "memory_limit_bytes",
)
FEATURE_STATUS_ENUM = (
    "OBSERVED",
    "UNOBSERVED",
    "NO_DENOMINATOR",
    "COUNTER_RESET",
    "GAP",
    "IDENTITY_AMBIGUOUS",
    "TRANSITION",
    "UNSUPPORTED",
    "CONFLICTING_SAMPLE",
)
STATE_CLASS_ENUM = (
    "RESOURCE_CHANGE",
    "EXECUTION_FAILURE",
    "SERVICE_UNAVAILABLE",
    "INSTANCE_TERMINATION",
    "NO_OBSERVED_CHANGE",
    "UNKNOWN",
)
EVIDENCE_GRADE_ENUM = (
    "INDEPENDENT_STATE",
    "CONTROLLED_EVENT",
    "OBSERVATION_DERIVED",
    "UNKNOWN",
)
EVIDENCE_UNIT_ENUM = ("EVENT", "WINDOW", "INSTANCE")
SCORE_STATUS_ENUM = ("Scored", "Unknown", "Untrained")
TERMINATION_FACT_TYPES = frozenset({"INSTANCE_TERMINATION", "KILLING", "POD_KILL"})

KEY_COLUMNS = (
    "case_id",
    "canonical_service_id",
    "sample_id",
    "window_start_ms",
    "window_end_ms",
    "score_time_lower_bound_ms",
    "schema_version",
)
DIAGNOSTIC_COLUMNS = (
    "observed_instance_count",
    "identity_ambiguity",
    "metric_last_query_time_ms",
    "metric_receipt_time_ms",
    "resource_valid",
    "quality_reasons",
)
# 身份/阶段/事件字段绝不允许以列名（含前缀变体）混进模型主输入
FORBIDDEN_COLUMN_TOKENS = ("phase", "target", "inject")

INDEX_COLUMNS = (
    "case_id",
    "canonical_service_id",
    "sample_id",
    "window_start_ms",
    "window_end_ms",
    "partition",
    "phase",
    "paired_resource_eligible",
    "exclusion_reasons",
    "proxy_exposure_label",
    "evidence_ids",
)
EVIDENCE_COLUMNS = (
    "evidence_id",
    "case_id",
    "canonical_service_id",
    "unit",
    "start_ms",
    "end_ms",
    "state_class",
    "evidence_grade",
    "fact_type",
    "source_path",
    "source_locator",
    "source_sha256",
    "review_status",
    "review_note",
    "heldout_from_model",
    "depends_on_input_sources",
    "independent_label",
    "proxy_label",
)
SCORE_COLUMNS = (
    "sample_id",
    "case_id",
    "canonical_service_id",
    "window_start_ms",
    "window_end_ms",
    "partition",
    "feature_group",
    "method",
    "anomaly_score",
    "calibrated_score",
    "score_status",
    "quality_reasons",
    "protocol_id",
    "model_id",
    "score_time_lower_bound_ms",
)

# 运行时字段仿 configs/base.yaml 先例，与协议同文件但不参与协议严格校验
RUNTIME_FIELDS = (
    "command",
    "out",
    "data_dir",
    "run_dir",
    "evaluation_dir",
    "dataset",
)

_META_REQUIRED = ("case_id", "window_start_ms", "window_end_ms", "is_anomaly_case")


@dataclass(frozen=True)
class ServiceProtocol:
    config: dict[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_resolved(raw: dict[str, Any]) -> dict[str, Any]:
    """逐字段校验 §6.1 冻结参数；任何偏差立即失败，不做静默补默认值。"""

    def req(key: str, expected: object) -> None:
        _require(key in raw, f"service protocol 缺少字段: {key}")
        _require(
            raw[key] == expected,
            f"service protocol 字段 {key} 必须冻结为 {expected!r}，实际 {raw[key]!r}",
        )

    req("protocol_id", PROTOCOL_ID)
    req("window_ms", WINDOW_MS)
    req("guard_ms", 30000)
    req("seed", 42)
    req("min_fit_rows", 30)
    req("min_calibration_rows", 10)
    req("max_counter_gap_s", 30)
    req("transition_ms", 30000)
    req("event_min_coverage", 0.8)
    req("event_top_fraction", 0.2)
    req("methods", list(FROZEN_METHODS))

    split = raw.get("split")
    _require(
        isinstance(split, dict) and split.get("fit_frac") == 0.6 and split.get("val_frac") == 0.2,
        "split 必须冻结为 fit_frac=0.6/val_frac=0.2",
    )

    feature_groups = raw.get("feature_groups")
    _require(
        isinstance(feature_groups, dict)
        and list(feature_groups.get("R3", []))
        == ["cpu_cores", "memory_usage_ratio", "process_count"]
        and list(feature_groups.get("CPU-only", [])) == ["cpu_cores"],
        "feature_groups 的 R3/CPU-only 成员与列序被冻结，不允许更改",
    )

    cases = raw.get("cases")
    _require(
        isinstance(cases, list) and [c.get("case_id") for c in cases] == list(FROZEN_CASE_IDS),
        f"cases 必须恰好是冻结的 4 个且顺序固定: {FROZEN_CASE_IDS}",
    )
    for case in cases:
        unknown_case_keys = set(case) - CASE_KEYS
        _require(
            not unknown_case_keys,
            f"case {case.get('case_id')!r} 出现未声明字段: {sorted(unknown_case_keys)}",
        )
        for key in ("case_id", "role", "window_start_ms", "window_end_ms", "is_anomaly_case"):
            _require(key in case, f"case {case.get('case_id')!r} 缺少 {key}")
        _require(
            case["role"] in CASE_ROLES,
            f"case {case['case_id']}.role 必须是 {CASE_ROLES} 之一，实际 {case['role']!r}",
        )
        _require(
            isinstance(case["is_anomaly_case"], bool),
            f"case {case['case_id']}.is_anomaly_case 必须是 bool",
        )
        _require(_is_int(case["window_start_ms"]), f"{case['case_id']}.window_start_ms 必须是 int")
        _require(_is_int(case["window_end_ms"]), f"{case['case_id']}.window_end_ms 必须是 int")
        # role 与 is_anomaly_case 必须互相一致，防止两条身份通道分叉
        _require(
            case["is_anomaly_case"] == (case["role"] != "normal"),
            f"case {case['case_id']} 的 role={case['role']} 与 is_anomaly_case 不一致",
        )
        _require(
            (case["role"], case["is_anomaly_case"], case.get("target_service"))
            == FROZEN_CASE_IDENTITIES[case["case_id"]],
            f"case {case['case_id']} 的 role/is_anomaly_case/target_service 必须冻结",
        )
        _require(case["window_start_ms"] < case["window_end_ms"], "window 起止必须有序")
        if not case["is_anomaly_case"]:
            _require(
                all(
                    case.get(k) is None
                    for k in ("inject_start_ms", "inject_end_ms", "target_service")
                ),
                "Normal 注入时间与目标必须 null",
            )
        if case["is_anomaly_case"]:
            for key in ("inject_start_ms", "inject_end_ms", "target_service"):
                _require(
                    key in case and case[key] is not None, f"故障 case {case['case_id']} 缺少 {key}"
                )
            _require(
                _is_int(case["inject_start_ms"]) and _is_int(case["inject_end_ms"]),
                f"故障 case {case['case_id']} 的 inject 时段必须是 int",
            )
            _require(
                case["target_service"] in FROZEN_SERVICES,
                f"{case['case_id']} target_service 不在候选集合",
            )

            _require(
                case["window_start_ms"] <= case["inject_start_ms"] < case["window_end_ms"]
                and case["inject_start_ms"] < case["inject_end_ms"],
                "inject 起止必须有序且起点位于 capture window",
            )

    services = raw.get("services")
    _require(
        isinstance(services, list) and tuple(services) == FROZEN_SERVICES,
        "services 必须是字典序冻结的 25 个候选服务",
    )

    identity = raw.get("identity")
    _require(isinstance(identity, dict), "缺少 identity 段")
    registry = identity.get("registry", {})
    aliases = identity.get("statefulset_aliases", {})
    _require(isinstance(registry, dict) and registry, "registry 必须是非空映射")
    for pod, svc in registry.items():
        _require(svc in FROZEN_SERVICES, f"registry 目标 {svc!r} 不在候选集合")
        _require(
            pod == svc or pod.startswith(svc + "-"),
            f"registry 键 {pod!r} 与其规范化服务 {svc!r} 前缀不一致",
        )
    _require(isinstance(aliases, dict), "statefulset_aliases 必须是映射")
    for pod, svc in aliases.items():
        _require(
            svc not in FROZEN_SERVICES,
            f"StatefulSet 别名目标 {svc} 是基础设施组件，禁止加入 v1 候选",
        )
    # 同一 Pod 同时出现在 registry 与 aliases 时合并视图会静默覆盖——冲突即报错
    overlap = set(registry) & set(aliases)
    _require(not overlap, f"registry 与 statefulset_aliases 存在键冲突: {sorted(overlap)}")
    container_map = identity.get("application_container_mapping")
    _require(
        isinstance(container_map, dict)
        and tuple(sorted(container_map)) == FROZEN_SERVICES
        and all(container_map[s] == s for s in FROZEN_SERVICES),
        "application_container_mapping 必须覆盖全部 25 候选且 container 名与 service 同名",
    )
    _require(
        identity.get("extra_container_aliases") == {},
        "extra_container_aliases: v1 不允许登记额外 container 别名"
        "（异常 container/sidecar 不自动混入）",
    )
    _require(
        identity.get("allowed_namespaces") == ["default"],
        "allowed_namespaces 冻结为 [default]",
    )
    _require(
        identity.get("allowed_jobs") == ["kubelet-cadvisor"],
        "allowed_jobs 冻结为 [kubelet-cadvisor]",
    )
    _require(
        identity.get("cpu_label_preference") == ["total", ""],
        "cpu_label_preference 冻结为 [total, '']（不累加每核）",
    )
    _require(
        identity.get("default_namespace") == "default"
        and identity.get("default_job") == "kubelet-cadvisor",
        "default_namespace/default_job 冻结为 default/kubelet-cadvisor",
    )
    _require(
        identity.get("raw_series_keys")
        == [
            "case_id",
            "namespace",
            "pod",
            "container",
            "id",
            "instance",
            "job",
            "metric_name",
            "cpu",
        ],
        "raw_series_keys 与冻结的原始序列键不符",
    )

    schema = raw.get("schema")
    _require(
        isinstance(schema, dict) and schema.get("version") == PROTOCOL_ID, "schema.version 不匹配"
    )
    _require(
        tuple(schema.get("feature_value_columns", [])) == FEATURE_VALUE_COLUMNS,
        "feature_value_columns 与冻结列不符",
    )
    for key, enum in (
        ("feature_status_enum", FEATURE_STATUS_ENUM),
        ("state_class_enum", STATE_CLASS_ENUM),
        ("evidence_grade_enum", EVIDENCE_GRADE_ENUM),
        ("evidence_unit_enum", EVIDENCE_UNIT_ENUM),
        ("review_status_enum", REVIEW_STATUS_ENUM),
        ("score_status_enum", SCORE_STATUS_ENUM),
    ):
        _require(tuple(schema.get(key, ())) == enum, f"schema.{key} 与冻结枚举不符")
    _require(
        tuple(schema.get("termination_fact_types", ())) == tuple(sorted(TERMINATION_FACT_TYPES)),
        "schema.termination_fact_types 与冻结集合不符",
    )

    freeze = raw.get("freeze")
    _require(
        isinstance(freeze, dict)
        and all(
            freeze.get(k) is False
            for k in (
                "fault_specific_adaptation",
                "hyperparameter_search",
                "score_dependent_selection",
            )
        ),
        "freeze 段三项必须全部为 false（禁止 Fault 适配/超参搜索/分数依赖选择）",
    )

    unknown = (
        set(raw)
        - {
            "protocol_id",
            "window_ms",
            "guard_ms",
            "seed",
            "split",
            "min_fit_rows",
            "min_calibration_rows",
            "max_counter_gap_s",
            "transition_ms",
            "event_min_coverage",
            "event_top_fraction",
            "methods",
            "feature_groups",
            "cases",
            "services",
            "identity",
            "schema",
            "freeze",
        }
        - set(RUNTIME_FIELDS)
    )
    _require(not unknown, f"service protocol 出现未声明字段: {sorted(unknown)}")
    return raw


def load_protocol(config_path: Path, dataset_path: Path) -> ServiceProtocol:
    config_path = Path(config_path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"protocol config {config_path} 不是合法 YAML mapping")
    resolved = _validate_resolved(dict(raw))

    # 顶层 registry 为"Deployment Pod + StatefulSet 显式别名"的合并解析视图，
    # canonical_service(pod, registry) 只需这一个参数（§3.2 签名冻结）
    resolved["registry"] = {
        **resolved["identity"]["registry"],
        **resolved["identity"]["statefulset_aliases"],
    }

    dataset_cfg = load_dataset_config(dataset_path)
    resolved["dataset_resolved"] = {
        "path": str(Path(dataset_path)),
        "name": dataset_cfg.name,
        "roots": [str(p) for p in dataset_cfg.roots],
        "normal_source": str(dataset_cfg.normal_source),
        "fused_window": dataset_cfg.fused_window,
    }
    LOG.info(
        "已加载冻结协议 %s（%d services, %d cases）",
        PROTOCOL_ID,
        len(resolved["services"]),
        len(resolved["cases"]),
    )
    return ServiceProtocol(config=resolved)


def load_case_meta(case_dir: Path) -> dict[str, object]:
    """读取 case_dir/trace_data/case_metadata.json；缺文件或缺字段直接失败。

    不允许把缺 metadata 的目录猜成 Normal——Normal 身份只来自真实元数据。
    """
    meta_path = Path(case_dir) / "trace_data" / "case_metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"case metadata 不存在: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if not isinstance(meta, dict):
        raise ValueError(f"case metadata 不是 JSON object: {meta_path}")
    for key in _META_REQUIRED:
        if key not in meta:
            raise ValueError(f"case metadata {meta_path} 缺少必填字段: {key}")
    if not isinstance(meta["is_anomaly_case"], bool):
        # 字符串 "false" 在分支判断里 truthy，会把 Normal 静默识别成故障 case
        raise ValueError(
            f"case metadata {meta_path} 的 is_anomaly_case 必须是 bool，"
            f"实际 {type(meta['is_anomaly_case']).__name__}"
        )
    for key in ("window_start_ms", "window_end_ms"):
        if not _is_int(meta[key]):
            raise ValueError(f"case metadata {meta_path} 的 {key} 必须是 int")
    # inject_* 对 Normal 显式为 null；键必须存在，禁止用缺键表达"无注入"
    for key in ("inject_start_ms", "inject_end_ms"):
        if key not in meta or (meta[key] is not None and not _is_int(meta[key])):
            raise ValueError(f"case metadata {meta_path} 的 {key} 必须是 int 或 null")
    if meta["window_end_ms"] <= meta["window_start_ms"]:
        raise ValueError(f"case metadata {meta_path} 时间窗起止非法")
    return meta


def canonical_service(pod: str, registry: dict[str, str]) -> str | None:
    """Pod 名 → canonical service；无法严格证明归属时返回 None。

    顺序：冻结 registry 精确命中 → StatefulSet 显式别名 → 完整服务名 →
    严格 Deployment 后缀模式（<service>-<rs-hash>-<pod-hash>，service 必须已知）。
    """
    if not isinstance(pod, str) or not pod:
        return None
    if pod in registry:
        return registry[pod]

    # 完整候选服务名原样返回；别名目标（tsdb-mysql 等基础设施）不在此列
    if pod in FROZEN_SERVICES:
        return pod

    # rsplit 严格三段 + ReplicaSet/Pod hash 形状校验，再验候选白名单：
    # 朴素"截掉最后两段"会把 ts-order-service 错成 ts-order，
    # 只看非空后缀则会放过来路不明的 Pod
    parts = pod.rsplit("-", 2)
    if (
        len(parts) == 3
        and re.fullmatch(r"[a-z0-9]{9,10}", parts[1]) is not None
        and re.fullmatch(r"[a-z0-9]{5}", parts[2]) is not None
        and parts[0] in FROZEN_SERVICES
    ):
        return parts[0]
    return None


def _sample_id(case_id: str, service: str, start_ms: int) -> str:
    # UTF-8 JSON 数组序列化：无哈希碰撞、无分隔符歧义
    return json.dumps([case_id, service, int(start_ms)], ensure_ascii=False)


def make_grid(meta: dict[str, object], services: list[str]) -> pd.DataFrame:
    """生成 (case × service × 完整窗) 笛卡尔网格。

    首窗 ceil 对齐 WINDOW_MS（与 yaml window_ms 镜像，load_protocol 已校验一致），
    只保留 end <= window_end_ms 的完整窗。
    """
    window_ms = WINDOW_MS
    start0 = -(-int(meta["window_start_ms"]) // window_ms) * window_ms
    end_limit = int(meta["window_end_ms"])
    starts = list(range(start0, end_limit - window_ms + 1, window_ms))
    case_id = str(meta["case_id"])

    rows = [
        {
            "case_id": case_id,
            "canonical_service_id": svc,
            "window_start_ms": start,
            "window_end_ms": start + window_ms,
            "sample_id": _sample_id(case_id, svc, start),
        }
        for start in starts
        for svc in services
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "case_id",
            "canonical_service_id",
            "window_start_ms",
            "window_end_ms",
            "sample_id",
        ],
    )


def assign_phase(
    start_ms: int,
    end_ms: int,
    meta: dict[str, object],
    guard_ms: int = 30000,
) -> str:
    """按全窗包含规则赋阶段；guard 外的过渡整窗为 boundary。"""
    if not meta.get("is_anomaly_case"):
        return "normal"
    inj_start = meta.get("inject_start_ms")
    inj_end = meta.get("inject_end_ms")
    if inj_start is None or inj_end is None:
        # 故障 case 缺注入时段是方向性元数据错误：返回 normal 会静默污染切分与标签
        raise ValueError(
            f"故障 case {meta.get('case_id')!r} 缺少 inject_start_ms/inject_end_ms，"
            "无法判定 phase"
        )
    if end_ms <= inj_start - guard_ms:
        return "baseline"
    if start_ms >= inj_start + guard_ms and end_ms <= inj_end - guard_ms:
        return "inject_core"
    if start_ms >= inj_end + guard_ms:
        return "recover"
    return "boundary"


def make_indices(features: pd.DataFrame, protocol: ServiceProtocol) -> pd.DataFrame:
    """网格 → index：phase 由 metadata 计算，Normal 按时间窗 60/20/20 共享切分。

    phase 只落在 index，绝不回写 features（§4.4）。
    """
    cfg = protocol.config
    window_ms = int(cfg["window_ms"])
    guard_ms = int(cfg["guard_ms"])
    case_meta = {c["case_id"]: c for c in cfg["cases"]}

    frames: list[pd.DataFrame] = []
    for case_id, group in features.groupby("case_id", sort=False):
        meta = case_meta.get(case_id)
        _require(meta is not None, f"case {case_id} 不在冻结协议内，拒绝生成 index")
        g = group.copy()
        g["phase"] = [
            assign_phase(int(s), int(s) + window_ms, meta, guard_ms) for s in g["window_start_ms"]
        ]
        if meta["is_anomaly_case"]:
            g["partition"] = "fault_evaluation"
        else:
            g["partition"] = _normal_partitions(g, cfg)
        g["paired_resource_eligible"] = False
        g["exclusion_reasons"] = [[] for _ in range(len(g))]
        g["evidence_ids"] = [[] for _ in range(len(g))]
        g["proxy_exposure_label"] = _proxy_labels(g, meta)
        frames.append(g)

    index = pd.concat(frames, ignore_index=True)
    return index[list(INDEX_COLUMNS)].sort_values(
        ["case_id", "window_start_ms", "canonical_service_id"], ignore_index=True
    )


def _normal_partitions(normal_grid: pd.DataFrame, cfg: dict[str, object]) -> pd.Series:
    # 按"时间窗"切（同窗 25 服务不拆散）；直接复用契约层时序切分
    split = split_normal_rows_temporal(
        normal_grid,
        time_col="window_start_ms",
        fit_frac=float(cfg["split"]["fit_frac"]),
        val_frac=float(cfg["split"]["val_frac"]),
    )
    # split 返回帧的索引经 ignore_index 重建，不能按 index 回贴；
    # 切分单位本来就是时间窗，按窗口集合映射回行
    window_part = {
        w: label
        for part, label in (
            ("train_fit", "normal_fit"),
            ("train_val", "normal_calibration"),
            ("eval_normal_holdout", "normal_tail"),
        )
        for w in split[part]["window_start_ms"].unique()
    }
    return normal_grid["window_start_ms"].map(window_part).astype("object")


def _proxy_labels(group: pd.DataFrame, meta: dict[str, object]) -> pd.Series:
    """§8.3 CPU 受控暴露弱对照：仅 CPU 事件 case 的目标服务，baseline=0/inject_core=1。"""
    labels = pd.array([pd.NA] * len(group), dtype="Int8")
    if meta.get("role") != "cpu":
        return labels
    is_target = group["canonical_service_id"] == meta["target_service"]
    labels[np.where(is_target & (group["phase"] == "baseline"))[0]] = 0
    labels[np.where(is_target & (group["phase"] == "inject_core"))[0]] = 1
    return labels


def validate_features(frame: pd.DataFrame, protocol: ServiceProtocol) -> None:
    missing = [c for c in KEY_COLUMNS if c not in frame.columns]
    _require(not missing, f"features 表缺少键/metadata 列: {missing}")
    _require(
        (frame["schema_version"] == PROTOCOL_ID).all(),
        f"schema_version 必须全部为 {PROTOCOL_ID}",
    )
    # 冻结成员校验：表中不允许出现 4 case / 25 service 之外的实体
    _require(
        set(frame["case_id"]) <= set(FROZEN_CASE_IDS),
        f"features 含冻结协议外 case_id: {sorted(set(frame['case_id']) - set(FROZEN_CASE_IDS))}",
    )
    _require(
        set(frame["canonical_service_id"]) <= set(FROZEN_SERVICES),
        "features 含冻结协议外 canonical_service_id: "
        f"{sorted(set(frame['canonical_service_id']) - set(FROZEN_SERVICES))}",
    )

    for col in FEATURE_VALUE_COLUMNS:
        _require(col in frame.columns, f"features 表缺少数值特征列: {col}")
        status_col = f"{col}__status"
        _require(status_col in frame.columns, f"features 表缺少状态列: {status_col}")
        values = frame[col]
        _require(
            pd.api.types.is_float_dtype(values),
            f"{col} 必须是 nullable float64，实际 {values.dtype}",
        )
        finite = values.dropna()
        _require(
            np.isfinite(finite.to_numpy()).all(),
            f"{col} 不允许 inf，只允许有限值或 NaN",
        )
        _require(frame[status_col].notna().all(), f"{status_col} 不允许 null")
        bad_status = set(frame[status_col].dropna().unique()) - set(FEATURE_STATUS_ENUM)
        _require(not bad_status, f"{status_col} 出现非法枚举: {sorted(bad_status)}")

        observed = frame[status_col].eq("OBSERVED")
        _require(
            observed.eq(values.notna()).all(), f"{col} OBSERVED 必须对应有限值，其余状态必须 NaN"
        )
        _require(values.dropna().ge(0).all(), f"{col} 不允许负值")
        if col == "process_count":
            _require(values.dropna().mod(1).eq(0).all(), "process_count 必须是整数计数")

    for col in DIAGNOSTIC_COLUMNS:
        _require(col in frame.columns, f"features 表缺少诊断列: {col}")
    _require(
        pd.api.types.is_bool_dtype(frame["identity_ambiguity"])
        and pd.api.types.is_bool_dtype(frame["resource_valid"]),
        "identity_ambiguity/resource_valid 必须是 bool",
    )
    for col in ("observed_instance_count", "metric_last_query_time_ms", "metric_receipt_time_ms"):
        _require(
            pd.api.types.is_integer_dtype(frame[col]),
            f"{col} 必须是 nullable 整数列",
        )

    for col in ("identity_ambiguity", "resource_valid"):
        _require(frame[col].notna().all(), f"{col} 不允许 null")
    counts = frame["observed_instance_count"]
    _require(counts.notna().all() and counts.ge(0).all(), "observed_instance_count 必须非空非负")
    expected_valid = counts.eq(1) & ~frame["identity_ambiguity"]
    for col in protocol.config["feature_groups"]["R3"]:
        expected_valid &= frame[col + "__status"].eq("OBSERVED") & frame[col].notna()
    _require(
        frame["resource_valid"].eq(expected_valid).all(), "resource_valid 与 R3/身份状态不一致"
    )

    allowed = set(KEY_COLUMNS) | set(DIAGNOSTIC_COLUMNS)
    allowed |= set(FEATURE_VALUE_COLUMNS) | {f"{c}__status" for c in FEATURE_VALUE_COLUMNS}
    extra = set(frame.columns) - allowed
    _require(
        not extra, f"features 表出现协议外列（防止身份/事件字段泄漏或静默加特征）: {sorted(extra)}"
    )
    leaked = [c for c in frame.columns if any(tok in c.lower() for tok in FORBIDDEN_COLUMN_TOKENS)]
    _require(not leaked, f"features 表禁止出现 phase/target/inject 类字段: {leaked}")

    # 主键唯一且 sample_id 与三元组一致
    _require(
        not frame.duplicated(["case_id", "canonical_service_id", "window_start_ms"]).any(),
        "features 主键 (case_id, canonical_service_id, window_start_ms) 存在重复",
    )
    expected = [
        _sample_id(c, s, int(t))
        for c, s, t in zip(
            frame["case_id"], frame["canonical_service_id"], frame["window_start_ms"]
        )
    ]
    _require(frame["sample_id"].tolist() == expected, "sample_id 与主键三元组序列化不一致")
    _require(
        (frame["window_end_ms"] == frame["window_start_ms"] + WINDOW_MS).all(),
        f"window_end_ms 必须 = start + {WINDOW_MS}",
    )
    _require(
        (frame["score_time_lower_bound_ms"] == frame["window_end_ms"]).all(),
        "score_time_lower_bound_ms 必须 = window_end_ms（无未来信息泄漏）",
    )


def validate_scores(frame: pd.DataFrame) -> None:
    missing = [c for c in SCORE_COLUMNS if c not in frame.columns]
    _require(not missing, f"scores 表缺少列: {missing}")
    extra = set(frame.columns) - set(SCORE_COLUMNS)
    _require(not extra, f"scores 表出现协议外列: {sorted(extra)}")

    bad_group = set(frame["feature_group"]) - set(FEATURE_GROUPS)
    _require(not bad_group, f"非法 feature_group: {sorted(bad_group)}")
    bad_method = set(frame["method"]) - set(FROZEN_METHODS)
    _require(not bad_method, f"非法 method: {sorted(bad_method)}")
    bad_status = set(frame["score_status"]) - set(SCORE_STATUS_ENUM)
    _require(not bad_status, f"非法 score_status: {sorted(bad_status)}")
    _require(frame["protocol_id"].eq(PROTOCOL_ID).all(), "scores.protocol_id 不匹配")

    dup = frame.duplicated(["sample_id", "feature_group", "method"])
    _require(not dup.any(), "scores 唯一键 (sample_id, feature_group, method) 存在重复")

    combos = {(g, m) for g in FEATURE_GROUPS for m in FROZEN_METHODS}
    for sample_id, grp in frame.groupby("sample_id"):
        seen = set(zip(grp["feature_group"], grp["method"]))
        _require(seen == combos, f"sample {sample_id} 必须恰好有 8 行（2 组 × 4 方法）")

    scored = frame["score_status"] == "Scored"
    for col in ("anomaly_score", "calibrated_score"):
        _require(
            pd.api.types.is_float_dtype(frame[col]),
            f"{col} 必须是 nullable float64",
        )
        _require(frame.loc[scored, col].notna().all(), f"Scored 行的 {col} 不允许 NaN")
        _require(
            np.isfinite(frame.loc[scored, col].to_numpy()).all(),
            f"Scored 行的 {col} 不允许 inf",
        )
        nonscored = ~scored
        _require(
            frame.loc[nonscored, col].isna().all(),
            f"Unknown/Untrained 行的 {col} 必须为 NaN",
        )


def _nullable_label_values(series: pd.Series, field: str) -> None:
    vals = pd.Series(series).dropna().unique()
    bad = [v for v in vals if v not in (0, 1, np.int8(0), np.int8(1))]
    _require(not bad, f"{field} 只允许 null/0/1，实际出现 {bad}")


def validate_evidence(frame: pd.DataFrame) -> None:
    missing = [c for c in EVIDENCE_COLUMNS if c not in frame.columns]
    _require(not missing, f"evidence 表缺少列: {missing}")
    extra = set(frame.columns) - set(EVIDENCE_COLUMNS)
    _require(not extra, f"evidence 表出现协议外列: {sorted(extra)}")

    # 非空 string 枚举列：先 notna 再校验枚举，NaN 不允许被 dropna 静默放行
    required_string_cols = (
        "evidence_id",
        "case_id",
        "canonical_service_id",
        "fact_type",
        "source_path",
        "source_locator",
    )
    for col in required_string_cols:
        _require(
            frame[col].map(lambda v: isinstance(v, str) and v != "").all(),
            f"evidence.{col} 必须是非空 string（不允许 NaN/空串）",
        )
    # review_note 允许空串但必须是 str（缺列已在上面拦截，None 不允许）
    _require(
        frame["review_note"].map(lambda v: isinstance(v, str)).all(),
        "evidence.review_note 必须是 string（允许空串，不允许 NaN）",
    )
    for col, enum in (
        ("unit", EVIDENCE_UNIT_ENUM),
        ("state_class", STATE_CLASS_ENUM),
        ("evidence_grade", EVIDENCE_GRADE_ENUM),
        ("review_status", REVIEW_STATUS_ENUM),
    ):
        _require(frame[col].notna().all(), f"evidence.{col} 不允许 NaN，必须取枚举值 {enum}")
        bad = set(frame[col]) - set(enum)
        _require(not bad, f"evidence.{col} 非法枚举: {sorted(map(str, bad))}")
    _require(
        set(frame["case_id"]) <= set(FROZEN_CASE_IDS),
        f"evidence 含冻结协议外 case_id: {sorted(set(frame['case_id']) - set(FROZEN_CASE_IDS))}",
    )
    _require(
        set(frame["canonical_service_id"]) <= set(FROZEN_SERVICES),
        "evidence 含冻结协议外 canonical_service_id: "
        f"{sorted(set(frame['canonical_service_id']) - set(FROZEN_SERVICES))}",
    )
    for col in ("start_ms", "end_ms"):
        _require(pd.api.types.is_integer_dtype(frame[col]), f"evidence.{col} 必须是 int64")
        _require(frame[col].notna().all(), f"evidence.{col} 不允许 NaN")
    _require(
        (frame["start_ms"] <= frame["end_ms"]).all(),
        "evidence start_ms 必须 <= end_ms",
    )
    _require(
        frame["heldout_from_model"].map(lambda v: isinstance(v, (bool, np.bool_))).all(),
        "heldout_from_model 必须是 bool",
    )
    _require(
        frame["depends_on_input_sources"].map(lambda v: isinstance(v, list)).all(),
        "depends_on_input_sources 必须是 list",
    )
    for col in ("independent_label", "proxy_label"):
        _nullable_label_values(frame[col], col)
    _require(not frame["evidence_id"].duplicated().any(), "evidence_id 必须唯一")

    for _, row in frame.iterrows():
        deps = row["depends_on_input_sources"]
        if deps and row["evidence_grade"] == "INDEPENDENT_STATE":
            raise ValueError(
                f"evidence {row['evidence_id']}: depends_on_input_sources 非空时"
                "不得标 INDEPENDENT_STATE（模型可消费来源不构成独立状态金标）"
            )
        if row["evidence_grade"] == "INDEPENDENT_STATE" and not row["heldout_from_model"]:
            raise ValueError(
                f"evidence {row['evidence_id']}: INDEPENDENT_STATE 声称独立于模型输入来源，"
                "heldout_from_model=false 与之矛盾，必须对模型留一"
            )
        if pd.notna(row["independent_label"]):
            _require(
                row["evidence_grade"] == "INDEPENDENT_STATE"
                and row["heldout_from_model"]
                and not deps
                and row["review_status"] == "confirmed",
                "independent_label 必须具有 confirmed/heldout/无输入依赖的 INDEPENDENT_STATE 证据",
            )
        if (
            row["fact_type"] in TERMINATION_FACT_TYPES
            and row["state_class"] == "SERVICE_UNAVAILABLE"
            and _label_is_one(row["independent_label"])
        ):
            raise ValueError(
                f"evidence {row['evidence_id']}: Killing/实例终止事实不得自动产生"
                "SERVICE_UNAVAILABLE 的 independent_label=1 金标"
            )


def _label_is_one(value: object) -> bool:
    return value is not None and not pd.isna(value) and int(value) == 1


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def reserve_output(path: Path, protected_roots: list[Path]) -> Path:
    """输出路径必须是不存在的新目录，且解析 symlink 后不落在任何受保护根内。

    无 force 开关：已存在即拒绝，防止覆盖或把输出写进数据/既有产物目录。
    """
    raw = Path(path)
    resolved = raw.resolve()
    if resolved.exists():
        raise ValueError(f"输出路径已存在 (already exists)，拒绝覆盖（无 force 开关）: {resolved}")
    for root in protected_roots:
        root_resolved = Path(root).resolve()
        if _is_relative_to(resolved, root_resolved) or resolved == root_resolved:
            raise ValueError(f"输出路径位于受保护目录 {root_resolved} 内，拒绝: {resolved}")
    resolved.mkdir(parents=True)
    LOG.info("已保留新输出目录 %s", resolved)
    return resolved


def write_manifest(out: Path, inputs: list[Path], protocol: ServiceProtocol) -> None:
    out = Path(out)
    excluded = {
        "manifest.json": "self-reference excluded",
        "_SUCCESS": "completion marker written after manifest",
        "_FAILURE": "failure marker is not an immutable data artifact",
    }
    outputs = {}
    for path in sorted(out.iterdir(), key=lambda p: p.name):
        if not path.is_file() or path.name in excluded:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        outputs[path.name] = {"size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    manifest = {
        "output_hash_scope": "direct artifact files present before completion; SHA256 full content",
        "outputs": outputs,
        "excluded_outputs": excluded,
        "protocol_id": PROTOCOL_ID,
        "inputs": [str(Path(p)) for p in inputs],
        "config": protocol.config,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
