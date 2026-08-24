"""per-endpoint 精确标签接入验证：label_granularity / is_endpoint_anomaly 两列。

对应 docs/superpowers/specs/2026-07-08-per-endpoint-label-eval-design.md。
"""

import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.build_contract import _attach_label_columns

REPO_ROOT = Path(__file__).parents[1]
CONFIG = REPO_ROOT / "configs/contract/v0.yaml"


def _build(out_dir: Path, dataset_yaml: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(CONFIG),
            "--dataset",
            str(dataset_yaml),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_precise_case_label_granularity_and_is_endpoint_anomaly(tmp_path):
    """带 target_endpoint 的 case：目标 endpoint 只在 inject 窗口 is_endpoint_anomaly=True，
    非目标 endpoint 全程 False；两者 label_granularity 都是 'endpoint'。"""
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/merged_mini.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case = eval_all[eval_all["case_id"] == "Lv_E_HTTPABORT_assurance_mini"]
    assert not case.empty, "fixture 未产出该 case，检查 Task 2 的 fixture 是否正确落地"

    target = case[case["endpoint_key"] == "GET:/api/v1/assuranceservice/assurances/types"]
    assert set(target["label_granularity"]) == {"endpoint"}
    target_by_phase = target.set_index("phase")["is_endpoint_anomaly"]
    assert target_by_phase["baseline"] == False  # noqa: E712
    assert target_by_phase["inject"] == True  # noqa: E712
    assert target_by_phase["recover"] == False  # noqa: E712

    non_target = case[case["endpoint_key"] == "POST:/api/v1/users/login"]
    assert set(non_target["label_granularity"]) == {"endpoint"}
    assert not non_target["is_endpoint_anomaly"].any()


def test_service_level_case_narrows_positives_to_target_service(tmp_path):
    """只有 target_service、没有 target_endpoint 的 case（如 anomod_v1 的 Lv_P/Lv_S/Lv_D）：
    label_granularity == 'service'，正样本收窄到 target_service 的行。

    这条测试取代了 entry 025 之前的 test_fallback_case_is_endpoint_anomaly_matches_is_anomaly
    ——那条断言 is_endpoint_anomaly 严格等于 is_anomaly，正是被修掉的 bug 本身：case 级
    fallback 对整个 case 内所有 service 一视同仁标正，把已存在于 case_meta 的
    target_service 丢掉，导致 service 级特征判别力被无关 service 的行稀释（entry 024
    实测 cpu_usage_rate AUROC 0.534 → 0.906）。

    fixture 里 Lv_D_CASELVL_travel 的 target_service=ts-travel-service，每个时间窗都含
    两个分属不同 service 的 endpoint（travel / travel2），构成 service 级 fan-out——
    修复前两者的 inject 行都是正样本（各 5 行），修复后只有 travel 侧是。
    用 nontarget_split_mini 而不是 mini_dataset：后者的故障 case 经 v0 endpoint 白名单
    过滤后只剩单个 endpoint，非目标 service 候选恒为空，验证不了收窄。
    """
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/nontarget_split_mini.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case = eval_all[eval_all["case_id"] == "Lv_D_CASELVL_travel"]
    assert not case.empty

    assert set(case["label_granularity"]) == {"service"}
    assert set(case["target_service"]) == {"ts-travel-service"}

    positives = case[case["is_endpoint_anomaly"]]
    assert not positives.empty, "target_service 命中的 inject 行应产出正样本"
    assert (positives["phase"] == "inject").all()
    assert (positives["service_name"] == "ts-travel-service").all()

    # 关键回归断言：inject 阶段内非 target_service 的行不再是正样本（旧实现下它们是）
    inject_rows = case[case["phase"] == "inject"]
    non_target_inject = inject_rows[inject_rows["service_name"] != "ts-travel-service"]
    assert not non_target_inject.empty, "fixture 前提变了：inject 阶段应含非目标 service 行"
    assert not non_target_inject["is_endpoint_anomaly"].any()

    # 正样本必须严格窄于 is_anomaly，否则说明收窄没生效（5 vs 10）
    assert int(case["is_endpoint_anomaly"].sum()) < int(case["is_anomaly"].sum())


def test_normal_case_stays_case_granularity(tmp_path):
    """两个 target 字段都没有的 case（Normal）仍走 'case' 档，且 label_target_observable
    必须为 True——该列语义是"是否已知不可观测"，Normal 既无 target 也无 inject 行，
    不属于不可观测。若误标 False，它会被 _write_v1 的吸收闸门莫名排除。"""
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/nontarget_split_mini.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    normal = eval_all[eval_all["anomaly_type"] == "Normal"]
    assert not normal.empty
    assert set(normal["label_granularity"]) == {"case"}
    assert not normal["is_endpoint_anomaly"].any()
    assert normal["label_target_observable"].all()


def test_unobservable_target_service_yields_zero_positives_and_warns(caplog):
    """target_service 落在 endpoint→service 映射覆盖范围外（mysql/gateway 类基础设施
    组件）时：正样本为 0、label_target_observable=False、且必须告警。

    选择放行 n_pos=0 而不是扩展映射表，是 entry 025 的显式决策——mysql/gateway 本就
    不对应任何客户端 endpoint，硬塞进映射表会造出假 endpoint 行，比 n_pos=0 更坏。
    产出 null AUROC 与 entry 016/023 对 Lv_S_KILLPOD_gateway 的处理惯例一致。
    """
    ep_df = pd.DataFrame(
        {
            "timestamp_window_ms": [0, 1000, 2000, 3000],
            "service_name": ["ts-order-service"] * 4,
        }
    )
    case_meta = {
        "inject_start_ms": 1000,
        "inject_end_ms": 2000,
        "target_service": "tsdb-mysql",
    }

    with caplog.at_level(logging.WARNING):
        _attach_label_columns(ep_df, case_meta)

    assert set(ep_df["label_granularity"]) == {"service"}
    assert not ep_df["is_endpoint_anomaly"].any(), "tsdb-mysql 不在 service_name 值域内"
    assert not ep_df["label_target_observable"].any()
    assert any("不参与训练池吸收" in r.message for r in caplog.records)


def test_observable_target_service_sets_flag_true_without_warning(caplog):
    """正向对照：target_service 能匹配上时 label_target_observable=True 且不告警，
    避免上面的守卫误报健康 case。"""
    ep_df = pd.DataFrame(
        {
            "timestamp_window_ms": [0, 1000, 2000, 3000],
            "service_name": ["ts-order-service"] * 4,
        }
    )
    case_meta = {
        "inject_start_ms": 1000,
        "inject_end_ms": 2000,
        "target_service": "ts-order-service",
    }

    with caplog.at_level(logging.WARNING):
        _attach_label_columns(ep_df, case_meta)

    assert ep_df["label_target_observable"].all()
    assert int(ep_df["is_endpoint_anomaly"].sum()) == 2  # 两个 inject 窗
    assert not any("不参与训练池吸收" in r.message for r in caplog.records)


def test_target_endpoint_takes_precedence_over_target_service():
    """两个 target 字段同时存在时（真实 Lv_E 数据就是如此）必须走 endpoint 档，
    正样本按 is_target_endpoint 收窄，不能因为 target_service 也在就退成 service 档。
    """
    ep_df = pd.DataFrame(
        {
            "timestamp_window_ms": [1000, 1000, 2000, 2000],
            "service_name": ["ts-travel-service", "ts-order-service"] * 2,
            "is_target_endpoint": [True, False, True, False],
        }
    )
    case_meta = {
        "inject_start_ms": 1000,
        "inject_end_ms": 2000,
        "target_service": "ts-travel-service",
        "target_endpoint": "POST:/api/v1/travelservice/trips/left",
    }

    _attach_label_columns(ep_df, case_meta)

    assert set(ep_df["label_granularity"]) == {"endpoint"}
    assert ep_df["is_endpoint_anomaly"].tolist() == [True, False, True, False]


def test_attach_label_columns_warns_when_target_endpoint_never_matches(caplog):
    """target_endpoint 已声明为 endpoint 级精确标签，但 is_target_endpoint 全为 False
    （比如目标 endpoint 被 v0 白名单过滤掉、或数据对齐出错）时，inject 阶段全部行会
    静默退化为 is_endpoint_anomaly=False。这种退化必须记录 warning，而不是无声发生。"""
    ep_df = pd.DataFrame(
        {
            "timestamp_window_ms": [0, 1000, 2000, 3000],
            "is_target_endpoint": [False, False, False, False],
        }
    )
    case_meta = {
        "inject_start_ms": 1000,
        "inject_end_ms": 2000,
        "target_endpoint": "GET:/api/v1/some/endpoint",
    }

    with caplog.at_level(logging.WARNING):
        _attach_label_columns(ep_df, case_meta)

    assert not ep_df["is_endpoint_anomaly"].any()
    assert any("is_endpoint_anomaly 将全为 False" in record.message for record in caplog.records)


def test_attach_label_columns_no_warning_when_target_endpoint_matches(caplog):
    """正常路径（is_target_endpoint 与 inject 窗口有交集）不应触发 warning，
    避免上面的守卫误报健康 case。"""
    ep_df = pd.DataFrame(
        {
            "timestamp_window_ms": [0, 1000, 2000, 3000],
            "is_target_endpoint": [False, True, True, False],
        }
    )
    case_meta = {
        "inject_start_ms": 1000,
        "inject_end_ms": 2000,
        "target_endpoint": "GET:/api/v1/some/endpoint",
    }

    with caplog.at_level(logging.WARNING):
        _attach_label_columns(ep_df, case_meta)

    assert ep_df["is_endpoint_anomaly"].any()
    assert not any(
        "is_endpoint_anomaly 将全为 False" in record.message for record in caplog.records
    )
