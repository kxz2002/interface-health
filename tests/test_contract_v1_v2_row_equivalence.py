"""v1↔v2 行等价红线（PR-4 Task 14）。

v2 与 v1_new_merge 的唯一差异是归一化参照系（per-case z-score + 三级 shrinkage
替代单一 Normal case 全局 min-max），split/标签/吸收闸门逻辑逐行一致（v2 把时序
切分提到归一化之前，归一化不参与任何行去向决策）。因此两个产物目录的
train/eval_all 必须：

1. 行集（sample_id 多重集）完全相同；
2. 全部标签/归属列逐行相等；
3. 特征值确实变了（反向断言，防 v2 原样复制 v1 的 vacuous 通过）。

第 1+2 条把"归一化改动"与"划分改动"彻底隔离——v1↔v2 的 AUROC 差异只可能来自
特征值本身。任何一条失败都意味着 v2 意外改动了划分或标签逻辑，优先级高于一切
AUROC 数字。

两份产物都是本地 gitignore 的大 parquet（由 scripts/build_contract.py 构建，
见 configs/contract/v1_new_merge.yaml / v2_new_merge.yaml 头部的 dvc/手动命令），
产物缺失时 skip 而非 fail——CI 新 clone 没有、也不应被要求持有这批数据。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parents[1]
V1_DIR = REPO_ROOT / "artifacts" / "contract_new_merge_expanded"
V2_DIR = REPO_ROOT / "artifacts" / "contract_new_merge_v2"

# 标签/归属列白名单：不含 endpoint_id——v1 开 fit_endpoint_baseline_stats=true
# 会产 endpoint_id，v2（D6）不产 sidecar，该列本就与归一化版本绑定，不属于
# "划分/标签逻辑"。is_target_endpoint 是 float64 含 NaN，语义已由
# is_endpoint_anomaly/label_granularity/label_target_observable 覆盖，不列入
# （NaN 列的精确比较另需 fillna 约定，无额外回归价值）。
LABEL_COLS = [
    "sample_id",
    "case_id",
    "endpoint_key",
    "phase",
    "is_anomaly",
    "is_endpoint_anomaly",
    "label_granularity",
    "label_target_observable",
]

FEATURE_PREFIXES = ("endpoint_red__", "service_metric__", "service_log__")


def _load_pair(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_path = V1_DIR / name
    v2_path = V2_DIR / name
    if not (v1_path.exists() and v2_path.exists()):
        pytest.skip(f"缺少 {v1_path.name}（v1/v2 产物之一未构建），跳过行等价断言")
    return pd.read_parquet(v1_path), pd.read_parquet(v2_path)


def _sorted_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df[LABEL_COLS].sort_values("sample_id").reset_index(drop=True)


def test_v1_v2_eval_all_rows_and_labels_identical():
    """eval_all 的行集与全部标签列必须逐行相等。"""
    v1, v2 = _load_pair("eval_all.parquet")
    assert len(v1) == len(v2) == 14186
    assert set(v1["sample_id"]) == set(v2["sample_id"])
    pd.testing.assert_frame_equal(_sorted_labels(v1), _sorted_labels(v2))


def test_v1_v2_train_rows_and_labels_identical():
    """train（训练池实际消费的帧）同样逐行相等：三路吸收（baseline 0.2 /
    inject 非目标 1.0 / recover 非目标 1.0）与 Normal 切分在 v2 路径上不能有
    任何行去向差异。"""
    v1, v2 = _load_pair("train.parquet")
    assert len(v1) == len(v2) == 9513
    assert set(v1["sample_id"]) == set(v2["sample_id"])
    pd.testing.assert_frame_equal(_sorted_labels(v1), _sorted_labels(v2))
    # One-Class 训练池不含"目标异常"正样本：吸收闸门的核心不变量是
    # is_endpoint_anomaly（评估标签）恒 0。注意 is_anomaly 不恒 0——
    # fault_inject_nontarget_fraction=1.0 吸收了 inject 阶段的非目标 fan-out
    # 行（new_merge 上 5380 行），它们 phase=="inject" 故 is_anomaly=True，
    # 但 is_endpoint_anomaly=False，按设计就是 One-Class 可用的"未受冲击"行
    # （见 CLAUDE.md expand_train_pool gotcha / history/entries/022,025）。
    assert int(v2["is_endpoint_anomaly"].sum()) == 0
    assert int(v1["is_endpoint_anomaly"].sum()) == 0


def test_v1_v2_train_eval_partition_structure_matches():
    """行守恒的外部交叉检查：v2 三个 Normal 切分帧 + 行数结构与 v1 一致。
    v2 不产 endpoint_baseline_stats.json（D6），但 parquet 切分帧齐全。"""
    for name, expected in [
        ("train_fit.parquet", 478),
        ("train_val.parquet", 169),
        ("eval_normal_holdout.parquet", 170),
    ]:
        v1, v2 = _load_pair(name)
        assert len(v2) == len(v1) == expected, name
        assert set(v1["sample_id"]) == set(v2["sample_id"]), name
        pd.testing.assert_frame_equal(_sorted_labels(v1), _sorted_labels(v2), obj=name)
    assert not (V2_DIR / "endpoint_baseline_stats.json").exists()


def test_v2_feature_values_actually_changed():
    """反向断言：v2 不能是 v1 的原样复制。

    min-max 与 per-case z-score 是不同参照系，**信息量列**（v1 侧自身存在
    取值变化、非近常数）在绝大多数行上都应取不同值；恒零/稀疏 0-1/近常数列
    （如 net_*_error_rate 全 0、trace_5xx_rate 是仅出现在 preserve endpoint
    上的稀疏 0/1）经归一化后仍可能逐行相等，与参照系无关，不纳入分母——
    否则断言测的是"数据集里有没有恒值列"这个无关性质。另有全体
    (行×特征) 单元改变占比兜底，防"每列只有零星几行不同"。
    """
    v1, v2 = _load_pair("eval_all.parquet")
    feat_cols = [c for c in v1.columns if c.startswith(FEATURE_PREFIXES) and c in v2.columns]
    assert feat_cols  # schema 防呆

    a = v1[["sample_id", *feat_cols]].sort_values("sample_id").reset_index(drop=True)
    b = v2[["sample_id", *feat_cols]].sort_values("sample_id").reset_index(drop=True)

    changed_cell_total = 0
    comparable_cell_total = 0
    informative: list[str] = []
    changed_informative = 0
    per_col_frac: dict[str, float] = {}
    for col in feat_cols:
        x, y = a[col], b[col]
        both_known = x.notna() & y.notna()
        n_cmp = int(both_known.sum())
        comparable_cell_total += n_cmp
        if n_cmp == 0:
            continue  # 全 NaN 列，与参照系无关
        n_changed = int(((x[both_known] - y[both_known]).abs() > 1e-6).sum())
        changed_cell_total += n_changed
        frac = n_changed / n_cmp
        per_col_frac[col] = frac
        # v1 侧自身非近常数（众数占比 <95%）才算信息量列；恒零/近常数列在
        # 任何归一化下都不变，不能要求两版本在该列上不同。
        xv = x[both_known]
        mode_frac = float((xv == xv.mode().iloc[0]).mean()) if len(xv) else 1.0
        if mode_frac < 0.95:
            informative.append(col)
            if frac > 0.30:
                changed_informative += 1

    assert informative, "没有任何信息量特征列，反向断言失去意义（检查 schema）"
    # 信息量列里至少 75% 要在 30% 以上的行上取不同值
    assert changed_informative >= 0.75 * len(informative), (
        f"信息量列 {len(informative)} 个中仅 {changed_informative} 个发生实质变化，"
        f"逐列改变行占比={per_col_frac}"
    )
    # 全体可比单元中至少 25% 改变——防"每列只有零星几行不同"
    overall = changed_cell_total / max(comparable_cell_total, 1)
    assert overall > 0.25, f"全体特征单元改变占比仅 {overall:.3f}"
