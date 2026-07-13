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


def test_fallback_case_is_endpoint_anomaly_matches_is_anomaly(tmp_path):
    """没有 target_endpoint 的 case（如 Lv_P_DISKIO_preserve）：
    label_granularity 全部为 'case'，is_endpoint_anomaly 严格等于 is_anomaly。"""
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/mini_dataset.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case = eval_all[eval_all["case_id"] == "Lv_P_DISKIO_preserve"]
    assert not case.empty

    assert set(case["label_granularity"]) == {"case"}
    pd.testing.assert_series_equal(
        case["is_endpoint_anomaly"].astype(bool).reset_index(drop=True),
        case["is_anomaly"].astype(bool).reset_index(drop=True),
        check_names=False,
    )


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
