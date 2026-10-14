"""T0: service_minimal_v1 冻结协议测试。

约束行为而非实现：实体解析、共享 Normal 切分、证据不自动成金标、输出保护。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.service import protocol as sp

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/service/minimal_v1.yaml"
DATASET = REPO / "configs/data/new_merge.yaml"

NORMAL_CASE = "Normal_20260728T161044Z"
CPU_CASE = "Lv_P_CPU_preserve_20260728T164711Z_em"
KILL_ORDER_CASE = "Lv_S_KILLPOD_order_20260728T203705Z_em"
KILL_PRESERVE_CASE = "Lv_S_KILLPOD_preserve_20260728T200124Z_em"

FEATURE_COLS = [
    "cpu_cores",
    "memory_usage_ratio",
    "process_count",
    "memory_usage_bytes",
    "memory_working_set_bytes",
    "memory_limit_bytes",
]
DIAG_COLS = [
    "observed_instance_count",
    "identity_ambiguity",
    "metric_last_query_time_ms",
    "metric_receipt_time_ms",
    "resource_valid",
    "quality_reasons",
]
KEY_COLS = [
    "case_id",
    "canonical_service_id",
    "sample_id",
    "window_start_ms",
    "window_end_ms",
    "score_time_lower_bound_ms",
    "schema_version",
]
EVIDENCE_COLS = [
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
]


@pytest.fixture(scope="module")
def protocol() -> sp.ServiceProtocol:
    return sp.load_protocol(CONFIG, DATASET)


def test_registry_and_statefulset(protocol: sp.ServiceProtocol) -> None:
    registry = protocol.config["registry"]

    # 合法 Deployment Pod 名（<service>-<rs-hash>-<pod-hash>）映射回完整服务名
    assert sp.canonical_service("ts-order-service-6c44d949d9-kmd96", registry) == "ts-order-service"
    # 完整 Service 名原样返回，不允许朴素地截掉最后两段（ts-order-service → ts-order）
    assert sp.canonical_service("ts-order-service", registry) == "ts-order-service"
    assert (
        sp.canonical_service("ts-preserve-other-service", registry) == "ts-preserve-other-service"
    )
    # 冻结 registry 中 Normal 实测 Pod 直接命中
    assert (
        sp.canonical_service("ts-preserve-service-5b54899475-hbpqd", registry)
        == "ts-preserve-service"
    )
    # StatefulSet 只能走显式别名，不能靠后缀猜测（tsdb-mysql-0 不得被截成 tsdb）
    assert sp.canonical_service("tsdb-mysql-0", registry) == "tsdb-mysql"
    assert sp.canonical_service("nacos-0", registry) == "nacos"
    # 未注册 / 白名单外的 Pod 返回 None
    assert sp.canonical_service("ts-unknown-service-6c44d949d9-kmd96", registry) is None
    assert sp.canonical_service("random-pod", registry) is None
    # 数据库别名组件不得混入候选
    assert "tsdb-mysql" not in protocol.config["services"]


def test_grid_and_shared_normal_split(protocol: sp.ServiceProtocol) -> None:
    services = protocol.config["services"]
    assert services == sorted(services)
    assert len(services) == 25

    # 合成 meta：ceil(1000/15000)*15000=15000；end<=61000 → 3 个完整窗
    meta = {
        "case_id": NORMAL_CASE,
        "window_start_ms": 1000,
        "window_end_ms": 61000,
        "is_anomaly_case": False,
    }
    grid = sp.make_grid(meta, services)
    starts = sorted(grid["window_start_ms"].unique().tolist())
    assert starts == [15000, 30000, 45000]
    assert (grid["window_end_ms"] == grid["window_start_ms"] + 15000).all()
    assert len(grid) == 3 * 25
    assert set(grid["canonical_service_id"]) == set(services)

    index = sp.make_indices(grid, protocol)
    # 同一时间窗的 25 个服务必须落在同一 partition，不允许按服务拆散
    parts_by_window = index.groupby("window_start_ms")["partition"].nunique()
    assert (parts_by_window == 1).all()
    fit = set(index.loc[index["partition"] == "normal_fit", "window_start_ms"])
    cal = set(index.loc[index["partition"] == "normal_calibration", "window_start_ms"])
    tail = set(index.loc[index["partition"] == "normal_tail", "window_start_ms"])
    assert fit.isdisjoint(tail)
    assert cal.isdisjoint(tail)
    assert fit.isdisjoint(cal)

    # 真实 Normal 元数据：120 个完整窗，60/20/20 按全窗切 = 72/24/24 窗
    case_dir = REPO / "data/new_merge" / NORMAL_CASE
    real_meta = sp.load_case_meta(case_dir)
    real_grid = sp.make_grid(real_meta, services)
    assert sorted(real_grid["window_start_ms"].unique().tolist())[0] == 1785255060000
    assert real_grid["window_start_ms"].nunique() == 120
    real_index = sp.make_indices(real_grid, protocol)
    counts = real_index.groupby("partition")["window_start_ms"].nunique().to_dict()
    assert counts == {
        "normal_fit": 72,
        "normal_calibration": 24,
        "normal_tail": 24,
    }
    # Normal 全部 phase=normal，且 phase 只存在于 index，不进 features
    assert set(real_index["phase"]) == {"normal"}
    assert "phase" not in real_grid.columns

    # CPU case：proxy_exposure_label 只在目标服务的 baseline=0/inject_core=1，其余 null
    cpu_meta = next(
        c
        for c in protocol.config["cases"]
        if c["case_id"] == "Lv_P_CPU_preserve_20260728T164711Z_em"
    )
    cpu_index = sp.make_indices(sp.make_grid(cpu_meta, services), protocol)
    assert set(cpu_index["partition"]) == {"fault_evaluation"}
    tgt = cpu_index[cpu_index["canonical_service_id"] == "ts-preserve-service"]
    assert set(tgt.loc[tgt["phase"] == "baseline", "proxy_exposure_label"].dropna()) == {0}
    assert set(tgt.loc[tgt["phase"] == "inject_core", "proxy_exposure_label"].dropna()) == {1}
    others = cpu_index[
        (cpu_index["canonical_service_id"] != "ts-preserve-service")
        | (~cpu_index["phase"].isin(["baseline", "inject_core"]))
    ]
    assert others["proxy_exposure_label"].isna().all()

    # 两个 KILL case：inject_end+guard 已超过 window_end，公式自然不产生 recover 窗
    for cid in (KILL_ORDER_CASE, KILL_PRESERVE_CASE):
        kill_meta = next(c for c in protocol.config["cases"] if c["case_id"] == cid)
        kill_index = sp.make_indices(sp.make_grid(kill_meta, services), protocol)
        assert "recover" not in set(kill_index["phase"])

    # 冻结声明：无 Fault 适配 / 超参搜索 / 分数依赖选择
    assert protocol.config["freeze"] == {
        "fault_specific_adaptation": False,
        "hyperparameter_search": False,
        "score_dependent_selection": False,
    }


def _make_feature_frame(index: pd.DataFrame) -> pd.DataFrame:
    frame = index[["case_id", "canonical_service_id", "sample_id", "window_start_ms"]].copy()
    frame["window_end_ms"] = frame["window_start_ms"] + 15000
    frame["score_time_lower_bound_ms"] = frame["window_end_ms"]
    frame["schema_version"] = "service_minimal_v1"
    rng = np.random.default_rng(0)
    for col in FEATURE_COLS:
        frame[col] = rng.random(len(frame)).astype(np.float64)
        frame[f"{col}__status"] = "OBSERVED"
    frame["process_count"] = 1.0
    frame["observed_instance_count"] = pd.array([1] * len(frame), dtype="Int64")
    frame["identity_ambiguity"] = False
    frame["metric_last_query_time_ms"] = pd.array(frame["window_end_ms"], dtype="Int64")
    frame["metric_receipt_time_ms"] = pd.array([pd.NA] * len(frame), dtype="Int64")
    frame["resource_valid"] = True
    frame["quality_reasons"] = [[] for _ in range(len(frame))]
    return frame


def _evidence_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "evidence_id": "ev-x",
        "case_id": CPU_CASE,
        "canonical_service_id": "ts-preserve-service",
        "unit": "EVENT",
        "start_ms": 1785258442613,
        "end_ms": 1785259047896,
        "state_class": "RESOURCE_CHANGE",
        "evidence_grade": "OBSERVATION_DERIVED",
        "fact_type": "CPU_RESOURCE_CHANGE",
        "source_path": "metric_data/x.csv",
        "source_locator": json.dumps({"metric": "container_cpu_usage_seconds_total"}),
        "source_sha256": None,
        "review_status": "pending_human_review",
        "review_note": "",
        "heldout_from_model": True,
        "depends_on_input_sources": ["metric"],
        "independent_label": None,
        "proxy_label": 1,
    }
    row.update(overrides)
    return row


def test_evidence_not_auto_gold(protocol: sp.ServiceProtocol) -> None:
    services = protocol.config["services"]
    meta = next(c for c in protocol.config["cases"] if c["case_id"] == CPU_CASE)
    grid = sp.make_grid(meta, services)
    index = sp.make_indices(grid, protocol)
    features = _make_feature_frame(index)
    sp.validate_features(features, protocol)

    # features 表出现 phase/target/inject 等身份或事件字段必须被拒绝
    leaked = features.copy()
    leaked["phase"] = "inject_core"
    with pytest.raises(ValueError, match="phase"):
        sp.validate_features(leaked, protocol)
    leaked2 = features.copy()
    leaked2["is_target_endpoint"] = 0
    with pytest.raises(ValueError):
        sp.validate_features(leaked2, protocol)
    # inf 不允许
    bad = features.copy()
    bad.loc[0, "cpu_cores"] = np.inf
    with pytest.raises(ValueError):
        sp.validate_features(bad, protocol)

    # CPU 变化来自模型也输入的 metric：OBSERVATION_DERIVED 合法
    cpu_row = _evidence_row()
    # 人工阅读 Log 增强语义，但 Log/Trace 属于模型可消费来源 → 仍只能 OBSERVATION_DERIVED
    log_row = _evidence_row(
        evidence_id="ev-log",
        unit="WINDOW",
        state_class="EXECUTION_FAILURE",
        fact_type="OBSERVED_LOG_ERROR",
        source_path="log_data/x.log",
        source_locator=json.dumps({"line_marker": "placeholder"}),
        depends_on_input_sources=["log"],
        proxy_label=None,
    )
    # Killing 事实：只能是 INSTANCE_TERMINATION，不自动变成 SERVICE_UNAVAILABLE 独立正窗
    kill_row = _evidence_row(
        evidence_id="ev-kill",
        case_id=KILL_ORDER_CASE,
        unit="INSTANCE",
        start_ms=1785272238000,
        end_ms=1785272838000,
        state_class="INSTANCE_TERMINATION",
        evidence_grade="INDEPENDENT_STATE",
        fact_type="INSTANCE_TERMINATION",
        source_path="log_data/kubernetes_events.json",
        source_locator=json.dumps({"reason": "Killing", "first_ms": 1785272238000}),
        depends_on_input_sources=[],
        review_status="confirmed",
        independent_label=1,
        proxy_label=None,
    )
    good = pd.DataFrame([cpu_row, log_row, kill_row], columns=EVIDENCE_COLS)
    sp.validate_evidence(good)

    # 依赖输入来源的行冒充 INDEPENDENT_STATE → 拒绝
    fake_gold = pd.DataFrame(
        [_evidence_row(evidence_id="ev-bad", evidence_grade="INDEPENDENT_STATE")],
        columns=EVIDENCE_COLS,
    )
    with pytest.raises(ValueError, match="INDEPENDENT_STATE"):
        sp.validate_evidence(fake_gold)

    # Killing 行不允许产生 SERVICE_UNAVAILABLE + independent_label=1 的独立金标
    auto_unavail = pd.DataFrame(
        [
            _evidence_row(
                evidence_id="ev-bad2",
                case_id=KILL_ORDER_CASE,
                state_class="SERVICE_UNAVAILABLE",
                fact_type="INSTANCE_TERMINATION",
                evidence_grade="INDEPENDENT_STATE",
                depends_on_input_sources=[],
                review_status="confirmed",
                independent_label=1,
            )
        ],
        columns=EVIDENCE_COLS,
    )
    with pytest.raises(ValueError, match="SERVICE_UNAVAILABLE"):
        sp.validate_evidence(auto_unavail)

    # scores：Scored 不允许 NaN，Unknown/Untrained 必须 NaN
    sample = index.iloc[0]
    base_score = {
        "sample_id": sample["sample_id"],
        "case_id": sample["case_id"],
        "canonical_service_id": sample["canonical_service_id"],
        "window_start_ms": sample["window_start_ms"],
        "window_end_ms": sample["window_start_ms"] + 15000,
        "partition": sample["partition"],
        "protocol_id": "service_minimal_v1",
        "model_id": "m1",
        "score_time_lower_bound_ms": sample["window_start_ms"] + 15000,
        "quality_reasons": [],
    }
    rows = []
    for group in ["CPU-only", "R3"]:
        for method in ["L2", "max", "OCSVM", "IF"]:
            rows.append(
                {
                    **base_score,
                    "feature_group": group,
                    "method": method,
                    "anomaly_score": 0.1,
                    "calibrated_score": 0.2,
                    "score_status": "Scored",
                }
            )
    scores = pd.DataFrame(rows)
    sp.validate_scores(scores)

    bad_scores = scores.copy()
    bad_scores.loc[0, "anomaly_score"] = np.nan
    with pytest.raises(ValueError, match="Scored"):
        sp.validate_scores(bad_scores)


def test_output_protection(tmp_path: Path, protocol: sp.ServiceProtocol) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "raw.csv").write_text("x")
    existing = tmp_path / "exists"
    existing.mkdir()

    with pytest.raises(ValueError, match="exist"):
        sp.reserve_output(existing, [data_root])
    with pytest.raises(ValueError, match="protect"):
        sp.reserve_output(data_root / "run_out", [data_root])

    # 指向受保护目录的 symlink 解析后同样必须拒绝
    link = tmp_path / "link_to_data"
    link.symlink_to(data_root / "run_out", target_is_directory=True)
    with pytest.raises(ValueError, match="protect"):
        sp.reserve_output(link, [data_root])

    fresh = tmp_path / "runs" / "new_out"
    resolved = sp.reserve_output(fresh, [data_root])
    assert resolved == fresh.resolve()
    assert fresh.is_dir()
    # 成功保留后不得再写/覆盖旧路径
    with pytest.raises(ValueError, match="exist"):
        sp.reserve_output(fresh, [data_root])

    sp.write_manifest(resolved, [CONFIG, DATASET], protocol)
    manifest = json.loads((resolved / "manifest.json").read_text())
    assert manifest["protocol_id"] == "service_minimal_v1"

    # 缺 metadata 的 case 必须报错，不允许猜成 Normal
    bogus = tmp_path / "case_x"
    (bogus / "trace_data").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        sp.load_case_meta(bogus)
    (bogus / "trace_data" / "case_metadata.json").write_text(json.dumps({"case_id": "case_x"}))
    with pytest.raises(ValueError, match="window_start_ms"):
        sp.load_case_meta(bogus)


EVIDENCE_JSON = REPO / "configs/service/evidence_minimal_v1.json"


def test_committed_evidence_file_passes_schema(protocol: sp.ServiceProtocol) -> None:
    """落盘证据文件必须被 validate_evidence 接受，且顶层标识与协议一致。"""
    doc = json.loads(EVIDENCE_JSON.read_text())
    assert doc["schema_version"] == "service_minimal_v1"
    assert doc["protocol_id"] == protocol.config["protocol_id"]
    frame = pd.DataFrame(doc["evidence"], columns=sp.EVIDENCE_COLUMNS)
    sp.validate_evidence(frame)


def test_features_reject_nonfrozen_members(protocol: sp.ServiceProtocol) -> None:
    services = protocol.config["services"]
    meta = next(c for c in protocol.config["cases"] if c["case_id"] == CPU_CASE)
    features = _make_feature_frame(sp.make_indices(sp.make_grid(meta, services), protocol))
    sp.validate_features(features, protocol)

    bad_case = features.copy()
    bad_case.loc[0, "case_id"] = "Lv_NOT_A_FROZEN_CASE"
    bad_case.loc[0, "sample_id"] = json.dumps(
        [
            "Lv_NOT_A_FROZEN_CASE",
            bad_case.loc[0, "canonical_service_id"],
            int(bad_case.loc[0, "window_start_ms"]),
        ],
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match="case_id"):
        sp.validate_features(bad_case, protocol)

    bad_svc = features.copy()
    bad_svc.loc[0, "canonical_service_id"] = "tsdb-mysql"
    bad_svc.loc[0, "sample_id"] = json.dumps(
        [bad_svc.loc[0, "case_id"], "tsdb-mysql", int(bad_svc.loc[0, "window_start_ms"])],
        ensure_ascii=False,
    )
    with pytest.raises(ValueError, match="canonical_service_id"):
        sp.validate_features(bad_svc, protocol)


def test_evidence_schema_gaps_rejected() -> None:
    base = _evidence_row()

    # 非空 string 枚举列为 None/NaN 必须报错（dropna 静默放行是 bug）
    for col in ("unit", "state_class", "evidence_grade"):
        row = _evidence_row(evidence_id=f"ev-nan-{col}")
        row[col] = None
        with pytest.raises(ValueError, match=col):
            sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))

    # review_status 必须在枚举内（yaml 的 review_status_enum 不是死配置）
    row = _evidence_row(evidence_id="ev-review", review_status="approved_by_boss")
    with pytest.raises(ValueError, match="review_status"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))

    # case/service 必须属于冻结集合
    row = _evidence_row(evidence_id="ev-case", case_id="not-a-frozen-case")
    with pytest.raises(ValueError, match="case_id"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))
    row = _evidence_row(evidence_id="ev-svc", canonical_service_id="tsdb-mysql")
    with pytest.raises(ValueError, match="canonical_service_id"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))

    # start_ms > end_ms 非法
    row = _evidence_row(evidence_id="ev-time", start_ms=2000, end_ms=1000)
    with pytest.raises(ValueError, match="start_ms"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))

    # INDEPENDENT_STATE 声称独立于输入来源，heldout_from_model=false 自相矛盾
    row = _evidence_row(
        evidence_id="ev-heldout",
        case_id=KILL_ORDER_CASE,
        unit="INSTANCE",
        state_class="INSTANCE_TERMINATION",
        evidence_grade="INDEPENDENT_STATE",
        fact_type="INSTANCE_TERMINATION",
        depends_on_input_sources=[],
        heldout_from_model=False,
        review_status="confirmed",
        independent_label=1,
        proxy_label=None,
    )
    with pytest.raises(ValueError, match="heldout_from_model"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))
    # 对照组：heldout=true 合法
    row["heldout_from_model"] = True
    sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))


def test_load_case_meta_rejects_nonbool_flag(tmp_path: Path) -> None:
    trace = tmp_path / "trace_data"
    trace.mkdir()
    meta_path = trace / "case_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "case_id": "c",
                "window_start_ms": 0,
                "window_end_ms": 1000,
                # 字符串 "false" 在 if 判断里 truthy，必须在加载阶段拒绝
                "is_anomaly_case": "false",
                "inject_start_ms": None,
                "inject_end_ms": None,
            }
        )
    )
    with pytest.raises(ValueError, match="is_anomaly_case"):
        sp.load_case_meta(tmp_path)


def test_meta_bool_type_and_fault_without_inject_window_raises() -> None:
    # is_anomaly_case 必须是 bool：字符串 "false" 在分支判断里 truthy，必须在加载时拒绝
    good = {
        "case_id": "c",
        "window_start_ms": 0,
        "window_end_ms": 200000,
        "is_anomaly_case": True,
        "inject_start_ms": 100000,
        "inject_end_ms": 150000,
    }
    assert sp.assign_phase(0, 50, good) == "baseline"
    bad_meta = dict(good, inject_start_ms=None, inject_end_ms=None)
    with pytest.raises(ValueError, match="inject"):
        sp.assign_phase(0, 50, bad_meta)


def _load_raw_config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG.read_text())


def _swap_first_two_services(raw: dict) -> None:
    raw["services"][0], raw["services"][1] = raw["services"][1], raw["services"][0]


@pytest.mark.parametrize(
    "mutate,message",
    [
        (_swap_first_two_services, "services"),
        (lambda r: r["identity"]["registry"].update({"evil-pod": "not-a-service"}), "registry"),
        (
            lambda r: r["identity"]["statefulset_aliases"].update({"evil-0": "ts-order-service"}),
            "别名",
        ),
        (
            lambda r: r["identity"]["application_container_mapping"].pop("ts-user-service"),
            "application_container_mapping",
        ),
        (
            lambda r: r["identity"]["extra_container_aliases"].update(
                {"ts-user-service": "sidecar"}
            ),
            "extra_container_aliases",
        ),
        (lambda r: r.update({"bogus_top_key": 1}), "未声明字段"),
        (lambda r: r.update(window_ms=16000), "window_ms"),
        (lambda r: r.update(guard_ms=999), "guard_ms"),
        (lambda r: r.update(methods=["L2"]), "methods"),
        (lambda r: r["cases"][1].update(role="cPu"), "role"),
        (lambda r: r["cases"][0].update(unknown_per_case_key=1), "case"),
        (
            lambda r: r["identity"]["statefulset_aliases"].update(
                {"ts-order-service-6c44d949d9-kmd96": "tsdb-mysql"}
            ),
            "冲突",
        ),
    ],
)
def test_resolved_config_rejects_mutations(mutate: object, message: str) -> None:
    raw = _load_raw_config()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        # 直接驱动 resolved 校验，避免每个参数化用例落一个临时 YAML
        sp._validate_resolved(raw)


@pytest.mark.parametrize("change", ["role", "target"])
def test_frozen_case_identity_guard(protocol, change):
    import copy

    config = copy.deepcopy(protocol.config)
    for key in ("registry", "dataset_resolved"):
        config.pop(key, None)
    if change == "role":
        config["cases"][1].update(
            role="normal",
            is_anomaly_case=False,
            inject_start_ms=None,
            inject_end_ms=None,
            target_service=None,
        )
    else:
        config["cases"][1]["target_service"] = "ts-order-service"
    with pytest.raises(ValueError):
        sp._validate_resolved(config)


@pytest.mark.parametrize(
    "change", ["null_status", "observed_nan", "negative", "count", "valid", "null_bool"]
)
def test_resource_status_consistency_guard(protocol, change):
    frame = _make_feature_frame(
        sp.make_grid(protocol.config["cases"][0], protocol.config["services"])
    )
    if change == "null_status":
        frame.loc[0, "cpu_cores__status"] = None
    elif change == "observed_nan":
        frame.loc[0, "cpu_cores"] = np.nan
    elif change == "negative":
        frame.loc[0, "cpu_cores"] = -1.0
    elif change == "count":
        frame.loc[0, "observed_instance_count"] = -1
    elif change == "valid":
        frame.loc[0, "resource_valid"] = False
    else:
        frame["resource_valid"] = frame["resource_valid"].astype("boolean")
        frame.loc[0, "resource_valid"] = pd.NA
    with pytest.raises(ValueError):
        sp.validate_features(frame, protocol)


@pytest.mark.parametrize("grade", ["CONTROLLED_EVENT", "OBSERVATION_DERIVED", "UNKNOWN"])
def test_weak_evidence_cannot_carry_independent_label(grade):
    row = _evidence_row(evidence_grade=grade, independent_label=1)
    with pytest.raises(ValueError, match="independent_label"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))


@pytest.mark.parametrize("service", ["security", "execute", "travel", "user"])
def test_changed_nine_character_rs_hash(protocol, service):
    name = f"ts-{service}-service"
    assert sp.canonical_service(f"{name}-64996ff68-abcde", protocol.config["registry"]) == name


@pytest.mark.parametrize("review", ["pending_human_review", "rejected"])
def test_independent_label_requires_confirmed_review(review):
    row = _evidence_row(
        evidence_grade="INDEPENDENT_STATE",
        independent_label=1,
        depends_on_input_sources=[],
        review_status=review,
    )
    with pytest.raises(ValueError, match="independent_label"):
        sp.validate_evidence(pd.DataFrame([row], columns=EVIDENCE_COLS))


def test_legitimate_unknown_and_zero_denominator(protocol):
    frame = _make_feature_frame(
        sp.make_grid(protocol.config["cases"][0], protocol.config["services"])
    )
    frame.loc[0, "memory_limit_bytes"] = 0.0
    frame.loc[0, "memory_usage_ratio"] = np.nan
    frame.loc[0, "memory_usage_ratio__status"] = "NO_DENOMINATOR"
    frame.loc[0, "resource_valid"] = False
    frame.loc[1, "process_count"] = 0.0
    sp.validate_features(frame, protocol)
