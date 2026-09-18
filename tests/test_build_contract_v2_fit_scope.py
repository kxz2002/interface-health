"""Task 12 泄漏红线（设计文档 D2，本 PR 最重要的测试）：

contract v2 把时序切分提到归一化之前后，Normalizer.fit 的行集必须**恰好**是
「Normal 的 train_fit ∪ 各故障 case baseline 前 fault_baseline_train_fraction
窗口」——eval_all 里剩下的 80% baseline 负样本、Normal holdout、inject/recover
行（含被训练池吸收的非目标行）一律不参与任何归一化统计量。

用 monkeypatch 截获 Normalizer.fit 的实参做行集断言，因此必须 in-process 调用
build_contract.main（既有 build_contract 测试都走 subprocess，spy 无法跨进程
生效）。fixture（tests/fixtures/v2_fit_scope_mini，生成器在同目录 _generate.py）
刻意做成单 endpoint、每 case 10 个 baseline 窗、无 inject/recover 窗，使预期
fit 行集可以按窗序号直接手算。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from scripts import build_contract as bc
from src.data.normalization import Normalizer

REPO_ROOT = Path(__file__).parents[1]
DATASET_CONFIG = REPO_ROOT / "tests/fixtures/v2_fit_scope_mini.yaml"

_EP = "POST:/api/v1/travelservice/trips/left"
_WINDOW_MS = 15_000

# fixture 时间基（与 tests/fixtures/v2_fit_scope_mini/_generate.py 一致）。
# 手算预期行集时直接引用，不回头从 CSV 推断——那样断言会退化成"实现与实现一致"。
_NORMAL_BASE_TS = 1_796_000_000_000
_FAULT_CASES = (
    ("Lv_A_FAULTONE_travel", 1_796_100_000_000),
    ("Lv_B_FAULTTWO_travel", 1_796_200_000_000),
)


def _sample_ids(case_id: str, base_ts: int, w_indices: range) -> set[str]:
    return {f"{case_id}__{_EP}__{base_ts + w * _WINDOW_MS}" for w in w_indices}


# 预期 fit 行集（单 endpoint、每窗一行）：
# - Normal 10 窗经 split_normal_rows_temporal（fit_frac=0.6）→ 最早 6 窗
# - 每个故障 case 10 个 baseline 窗 × fraction 0.2 → int(10*0.2)=2 个最早窗
_EXPECTED_FIT_IDS = _sample_ids("Normal", _NORMAL_BASE_TS, range(6)) | set().union(
    *(_sample_ids(cid, base, range(2)) for cid, base in _FAULT_CASES)
)
# 明确不应进 fit 的故障行：两个 case 各剩下的 8 个 baseline 窗（eval 侧负样本）
_EXPECTED_EVAL_BASELINE_IDS = set().union(
    *(_sample_ids(cid, base, range(2, 10)) for cid, base in _FAULT_CASES)
)


def _write_v2_config(tmp_path: Path) -> str:
    # 只声明非 rate 列：Task 13 才放宽 rate [0,1] 校验，z-score 的负值/>1 值在那之前
    # 过不了 validate_contract_df。两套 per-case scope 都声明（process_count 在本
    # fixture 全 NaN，顺带覆盖 z-score 回退链在空数据上不崩）。
    cfg = {
        "contract_version": "v2",
        "window_size_s": 15,
        "expand_train_pool": True,
        "fit_endpoint_baseline_stats": False,
        "fault_baseline_train_fraction": 0.2,
        # fixture 无 inject/recover 窗，这两个 fraction 对行集无影响，沿用 v2 规格
        # 0.2/1.0/1.0 的后两个值钉死配置形态
        "fault_inject_nontarget_train_fraction": 1.0,
        "fault_recover_nontarget_train_fraction": 1.0,
        "modalities": {
            "endpoint_red": {
                "preprocessor": "TracePreprocessor",
                "preprocessor_version": "v0",
                "features": [
                    "trace_request_count",
                    "trace_latency_p50",
                    "trace_latency_p95",
                    "client_request_count",
                    "client_latency_p95",
                    "latency_divergence",
                ],
                "normalization": "per_case_endpoint_z_score",
                "candidates_pool": [],
            },
            "service_metric": {
                "preprocessor": "MetricPreprocessor",
                "preprocessor_version": "v0",
                "features": ["process_count"],
                "normalization": "per_case_service_z_score",
                "candidates_pool": [],
            },
        },
    }
    cfg_path = tmp_path / "v2_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return str(cfg_path)


def _run_build_in_process(tmp_path: Path, monkeypatch, captured: dict) -> Path:
    out_dir = tmp_path / "contract_v2"
    cfg_path = _write_v2_config(tmp_path)

    orig_fit = Normalizer.fit

    def spy(self, df):
        # 截获行**身份**（sample_id）而非数值：fit 发生在归一化之前，这里是原始尺度。
        captured["sample_ids"] = set(df["sample_id"])
        captured["n_rows"] = len(df)
        return orig_fit(self, df)

    monkeypatch.setattr(Normalizer, "fit", spy)
    # dataset config 里的 roots 是相对路径，既有 subprocess 测试靠 cwd=REPO_ROOT 解析
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_contract.py",
            "--config",
            cfg_path,
            "--dataset",
            str(DATASET_CONFIG),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
    )
    bc.main()
    return out_dir


def test_v2_normalizer_fit_rows_exactly_train_pool(tmp_path, monkeypatch):
    """最强断言：fit 行集恰好 = Normal train_fit（6 行）∪ 两个故障 case 各最早
    2 个 baseline 窗（4 行），共 10 行——不多（不能含 eval baseline/holdout）
    也不少。"""
    captured: dict = {}
    out_dir = _run_build_in_process(tmp_path, monkeypatch, captured)

    assert "sample_ids" in captured, "Normalizer.fit 未被调用，spy 失效"
    assert captured["n_rows"] == 10
    assert captured["sample_ids"] == _EXPECTED_FIT_IDS


def test_v2_normalizer_fit_rows_subset_of_train_and_disjoint_from_eval(tmp_path, monkeypatch):
    """D2 红线的关系形式：fit ⊆ train.parquet，且与 eval_all.parquet 零交集。"""
    captured: dict = {}
    out_dir = _run_build_in_process(tmp_path, monkeypatch, captured)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    fit_ids = captured["sample_ids"]
    train_ids = set(train["sample_id"])
    eval_ids = set(eval_all["sample_id"])

    assert fit_ids <= train_ids
    assert not (fit_ids & eval_ids)
    assert train_ids & eval_ids == set()

    # 泄漏的正面刻画：留在 eval 的 16 个故障 baseline 窗（每 case 8 窗 × 1 行）
    # 必须一个都没进 fit——这正是 v1 实现（fit=纯 Normal）与 v2 实现之间最容易
    # 被悄悄破坏的边界。
    assert _EXPECTED_EVAL_BASELINE_IDS <= eval_ids
    assert not (_EXPECTED_EVAL_BASELINE_IDS & fit_ids)

    # Normal holdout（w=8,9）属 eval 侧，同样不进 fit
    normal_holdout_ids = _sample_ids("Normal", _NORMAL_BASE_TS, range(8, 10))
    assert normal_holdout_ids <= eval_ids
    assert not (normal_holdout_ids & fit_ids)


def test_v2_fault_baseline_rows_split_2_to_train_8_to_eval(tmp_path, monkeypatch):
    """正向对照，防止上面的精确断言在切分整体不发生时 vacuously 通过：训练池
    必须真实吸收了每个故障 case 最早 2 个 baseline 窗且带 source_phase 标签。"""
    captured: dict = {}
    out_dir = _run_build_in_process(tmp_path, monkeypatch, captured)

    train = pd.read_parquet(out_dir / "train.parquet")
    fault_rows = train[train["anomaly_type"] != "Normal"]
    assert len(fault_rows) == 4
    assert (fault_rows["phase"] == "baseline").all()
    assert set(fault_rows["case_id"]) == {cid for cid, _ in _FAULT_CASES}
    assert (fault_rows["source_phase"] == "fault_baseline").all()
    for cid, _ in _FAULT_CASES:
        assert fault_rows[fault_rows["case_id"] == cid]["timestamp_window_ms"].nunique() == 2


def test_v2_does_not_emit_endpoint_baseline_stats_sidecar(tmp_path, monkeypatch):
    """D6：v2 不 fit、不落盘 endpoint_baseline_stats.json。"""
    captured: dict = {}
    out_dir = _run_build_in_process(tmp_path, monkeypatch, captured)

    assert (out_dir / "normalization_stats.json").exists()
    assert not (out_dir / "endpoint_baseline_stats.json").exists()
    train = pd.read_parquet(out_dir / "train.parquet")
    assert "endpoint_id" not in train.columns
