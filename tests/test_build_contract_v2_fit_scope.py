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


# 本文件测试用到的 endpoint_red 非 rate 特征（Task 13 才放宽 rate [0,1] 校验，
# z-score 的负值/>1 值在那之前过不了 validate_contract_df，故不声明任何 rate 列）。
_NON_RATE_ENDPOINT_FEATURES = [
    "trace_request_count",
    "trace_latency_p50",
    "trace_latency_p95",
    "client_request_count",
    "client_latency_p95",
    "latency_divergence",
]


def _write_config(tmp_path: Path, cfg: dict, name: str = "v2_test.yaml") -> str:
    cfg_path = tmp_path / name
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return str(cfg_path)


def _v2_config(**overrides) -> dict:
    # process_count 在两个 mini fixture 上都全 NaN（无 metric CSV），顺带覆盖
    # z-score 回退链在整列空数据上不崩。
    cfg = {
        "contract_version": "v2",
        "window_size_s": 15,
        "expand_train_pool": True,
        "fit_endpoint_baseline_stats": False,
        "fault_baseline_train_fraction": 0.2,
        "fault_inject_nontarget_train_fraction": 1.0,
        "fault_recover_nontarget_train_fraction": 1.0,
        "modalities": {
            "endpoint_red": {
                "preprocessor": "TracePreprocessor",
                "preprocessor_version": "v0",
                "features": list(_NON_RATE_ENDPOINT_FEATURES),
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
    cfg.update(overrides)
    return cfg


def _run_build_in_process(
    tmp_path: Path,
    monkeypatch,
    captured: dict,
    cfg: dict | None = None,
    dataset_config: Path | None = None,
) -> Path:
    out_dir = tmp_path / "contract_v2"
    cfg_path = _write_config(tmp_path, cfg if cfg is not None else _v2_config())

    orig_fit = Normalizer.fit

    def spy(self, df):
        # 截获行**身份**（sample_id）而非数值：fit 发生在归一化之前，这里是原始尺度。
        # 同时保留 normalizer 实例与 fit 帧副本，供"落盘尺度"测试重算归一化值。
        captured["sample_ids"] = set(df["sample_id"])
        captured["n_rows"] = len(df)
        captured["normalizer"] = self
        captured["fit_df"] = df.copy()
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
            str(dataset_config or DATASET_CONFIG),
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


def test_v2_train_parquet_written_in_normalized_scale_not_raw(tmp_path, monkeypatch):
    """落盘尺度红线：train.parquet 里的特征必须是归一化**之后**的 z-score 值，
    不是切分时预算帧携带的原始量纲。

    预算切分发生在归一化之前，_write_v1 若直接落盘预算帧（漏掉
    _reselect_rows_by_sample_id 取回），parquet 里会是 20000+ 毫秒的原始 latency，
    且行集断言全部存活——必须用这条测试单独承载"尺度正确性"。

    两层断言：
    1. 量级：fixture 原始 trace_latency_p50 是 20000+ 毫秒，落盘值必须是 z-score
       量级（|x| 小个位数）——raw 泄漏会产生 4 个数量级的反差；
    2. 逐值：用 spy 截获的已 fit Normalizer 对 fit_df 现场 transform，重算结果
       必须与 train.parquet 中同 sample_id 的值逐值相等（覆盖 Normal train_fit
       与被吸收的 fault_baseline 两类行）。
    """
    captured: dict = {}
    out_dir = _run_build_in_process(tmp_path, monkeypatch, captured)
    train = pd.read_parquet(out_dir / "train.parquet")

    feature_cols = [f"endpoint_red__{f}" for f in _NON_RATE_ENDPOINT_FEATURES]

    # 1) 量级断言：被吸收的 fault_baseline 行
    fault_rows = train[train["source_phase"] == "fault_baseline"]
    assert len(fault_rows) == 4
    lat = fault_rows["endpoint_red__trace_latency_p50"]
    # 原始量纲 20000+ ms；z-score 后必然是个位数。100 与两侧都隔着 2~4 个数量级。
    assert (lat.abs() < 100).all(), lat.tolist()
    # 同一批行的原始值确实是万级（钉死 fixture 前提，避免断言 vacuous）
    raw_lat = (
        captured["fit_df"]
        .set_index("sample_id")
        .loc[fault_rows["sample_id"], "endpoint_red__trace_latency_p50"]
    )
    assert (raw_lat > 10_000).all(), raw_lat.tolist()

    # 2) 逐值重算：fit_df 是归一化之前的帧，用截获的 normalizer 现场 transform，
    #    得到的就是 parquet 里应落的归一化值（normalizer transform 需要 group 列）。
    normalizer = captured["normalizer"]
    fit_df = captured["fit_df"]
    group_cols = ["endpoint_key", "service_name", "case_id"]
    recomputed = normalizer.transform(fit_df[feature_cols + group_cols])
    recomputed = recomputed.assign(sample_id=fit_df["sample_id"].to_numpy()).set_index("sample_id")
    written = train.set_index("sample_id")
    fit_ids = captured["sample_ids"]
    assert set(written.index) & fit_ids == fit_ids
    pd.testing.assert_frame_equal(
        written.loc[sorted(fit_ids), feature_cols].sort_index(),
        recomputed.loc[sorted(fit_ids), feature_cols].sort_index(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    # Normal train_fit 单独点一遍（它走另一条预算帧 normal_parts，不经 fault
    # baseline 的 reselect 分支）：train_fit.parquet 同样必须是归一化后尺度。
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert train_fit["endpoint_red__trace_latency_p50"].abs().max() < 100


_NONTARGET_DATASET = REPO_ROOT / "tests/fixtures/nontarget_split_mini.yaml"


def test_v2_inject_recover_rows_never_enter_fit_even_when_absorbed(tmp_path, monkeypatch):
    """D2 承诺此前零测试承载：inject/recover 行（含被训练池**吸收**的非目标行）
    一律不参与 Normalizer.fit。

    v2_fit_scope_mini 结构性无法违反这条（没有 inject/recover 窗），故复用
    nontarget_split_mini：15 窗/故障 case = 5 baseline + 5 inject + 5 recover、
    2 endpoint fan-out，两个 nontarget fraction 推到 1.0 后 train.parquet 实际
    吸收 10 行 inject 非目标 + 5 行 recover 非目标——先证明吸收真实发生（否则
    下面的不相交断言 vacuous），再断言这些行一行都没进 fit，同时 fit ⊆ train、
    fit ∩ eval = ∅ 在含吸收行的真实形态下依旧成立。
    """
    captured: dict = {}
    out_dir = _run_build_in_process(
        tmp_path,
        monkeypatch,
        captured,
        cfg=_v2_config(
            fault_baseline_train_fraction=0.2,
            fault_inject_nontarget_train_fraction=1.0,
            fault_recover_nontarget_train_fraction=1.0,
        ),
        dataset_config=_NONTARGET_DATASET,
    )

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    fit_ids = captured["sample_ids"]
    train_ids = set(train["sample_id"])
    eval_ids = set(eval_all["sample_id"])

    # 正向对照：吸收必须真实发生，否则本测试没有区分力
    absorbed_inject = train[train["source_phase"] == "fault_inject_nontarget"]
    absorbed_recover = train[train["source_phase"] == "fault_recover_nontarget"]
    assert len(absorbed_inject) == 10
    assert len(absorbed_recover) == 5

    # 全数据里所有 inject/recover 窗的行身份（无论最终落 train 还是 eval）
    all_rows = pd.concat([train, eval_all], ignore_index=True)
    inject_or_recover_ids = set(
        all_rows.loc[all_rows["phase"].isin(["inject", "recover"]), "sample_id"]
    )
    assert inject_or_recover_ids, "fixture 前提变了：没有 inject/recover 行"

    # 核心红线：fit 与所有 inject/recover 行（含被吸收进 train 的那 15 行）零交集
    assert not (
        fit_ids & inject_or_recover_ids
    ), f"{len(fit_ids & inject_or_recover_ids)} 行 inject/recover 行污染了 fit 统计量"
    # 被吸收的非目标行进了 train，但绝不能进 fit
    assert not (fit_ids & set(absorbed_inject["sample_id"]))
    assert not (fit_ids & set(absorbed_recover["sample_id"]))

    # 关系形式的红线在真实吸收形态下仍成立
    assert fit_ids <= train_ids
    assert not (fit_ids & eval_ids)

    # fit 的精确构成：仅 Normal train_fit（6 行）+ 三个故障 case 各最早 1 个
    # baseline 窗 × 2 endpoint（6 行）= 12 行，一行不多
    assert captured["n_rows"] == 12
    fit_rows = all_rows.set_index("sample_id").loc[sorted(fit_ids)]
    assert set(fit_rows["phase"]) == {"normal", "baseline"}
    assert not fit_rows["is_anomaly"].any()
