#!/usr/bin/env python3
"""平凡基线打分器：产出契约合规的 scores.parquet，走现有 train↔eval 接口墙。

entry 027 实测：不含任何学习过程的平凡基线在 transductive 口径下打败了全部
六种融合机制（诊断数字见 entry 027；本脚本产出 lean 口径，常驻数字见 entry 029）。
把它们固化成常驻 dvc stage，使以后每个实验的 metrics.json 旁边都有参照线
——否则"有没有真的超过平凡规则"这个判断会被遗忘（P0 首要动机）。

**lean 口径**：z-score 的统计量只取 train.parquet，与学习模型共享同一训练池。
注意该池按 is_endpoint_anomaly==False 吸收 inject 非目标行，池的构成本身携带
标签信息，因此 lean 是"与模型信息量对等"，不是"未使用标签"。entry 027 诊断脚本
里的 transductive 版（用 eval_all 中留下的 80% baseline 行）留在那个一次性脚本里
作附注，不进本 stage——拿一个偷看过 eval 的参照来审判模型，赢输都说不清。

**y_true 口径**：本脚本输出的 y_true 取 `is_endpoint_anomaly`（与
eval_baseline_v0.py 实际消费的标签列一致），而 train_baseline_v0.py 的 y_true
取 `is_anomaly`（phase == "inject"）——两个 scores 生产者的 y_true 口径不同。
eval_baseline_v0.py 不读 y_true 列（只用 is_endpoint_anomaly），当前影响为零；
但未来直接按 scores_v0 契约读 y_true 的消费方需注意此差异（entry 029 记录）。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts.scores_v0 import validate_scores_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# eval_baseline_v0.py 的分层指标硬依赖这些诊断列（is_endpoint_anomaly 缺失会
# 显式报错；case_id 支撑 per-case macro；anomaly_type 无守卫直接 groupby），
# 与 train_baseline_v0.py 的落盘列保持同款 10 列。
DIAGNOSTIC_COLUMNS = [
    "case_id",
    "endpoint_key",
    "phase",
    "anomaly_type",
    "anomaly_level",
    "label_granularity",
    "is_endpoint_anomaly",
]

_BASELINE_CHOICES = ("rel_pos", "zscore_l2", "zscore_max")


def compute_rel_pos_scores(df: pd.DataFrame) -> pd.DataFrame:
    """窗口在 case 内的相对时序位置：唯一窗口的 dense 序号 ÷ (窗口数 − 1)。

    eval_all 是 endpoint×window 粒度，同一时间窗有多行（每 endpoint 一行），
    故按 timestamp 的 dense 排名取序号，让同窗各行同分。per-case 宏平均对任何
    时间单调变换不变，dense-rank 与 entry 027 诊断脚本的时间戳 min-max 两版在
    per-case 口径恒等（与采集是否等距无关）；只在 pooled 口径上分歧——当前
    eval_all 已存在 30s/75s/285s 空窗（非等距），pooled 0.9303 vs 0.9340 即
    来源于此。单窗 case 置 0.0，避免除零。
    """
    pos = df.groupby("case_id")["timestamp_window_ms"].rank(method="dense").astype(int) - 1
    n_windows_minus_one = pos.groupby(df["case_id"]).transform("max")
    score = np.where(n_windows_minus_one > 0, pos / n_windows_minus_one, 0.0)
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "score": score.astype(float),
            "y_true": df["is_endpoint_anomaly"].astype(int),
        }
    )


def compute_zscore_scores(
    df: pd.DataFrame, fit_df: pd.DataFrame, feature_cols: list[str], agg: str
) -> pd.DataFrame:
    """逐特征 z-score 后聚合：agg="l2" → sqrt(Σz²)；agg="max" → max|z|。

    统计量只来自 fit_df（`main()` 传 train.parquet；函数本身接受任意 fit_df），
    按 (case_id, endpoint_key) 分组——per-case 自适应参照系，采集漂移在分子里
    抵消。退化规则：

    - 组存在但零方差：std 置 **1.0 哨兵**（保留原始量纲）。绝不能用 1e-9 兜底：
      eval 侧任何非零差值会被放大 1e9 倍（history 013 的爆值与 entry 027 的 RG
      sigmoid 饱和同根因）。注意哨兵只兜 std==0/NaN：`0<std<1e-3` 的单元格仍可
      产生 |z|~1e4（实测最大 13731）——AUROC 只看排序不受影响，但不可对该分数
      做绝对阈值。
    - 组只有 1 行：std 为 NaN（ddof=1）同样落 1.0 哨兵，但 mu 有限——
      z = x − 单点值按原始量纲计入，不归零（实际有 3 个 Lv_D
      `POST:/api/v1/users/login` 组属于此情形）。
    - 组在 fit_df 里不存在（如故障 case 的 baseline 行整窗未被训练池吸收）：
      没有参照就不判异常，该组 z 为 NaN，最终按 0 偏离处理。
    - 特征本身 NaN（缺测）同样按 0 偏离——缺测不等于异常，与 contract_dataloader
      的均值填补精神一致。
    """
    if agg not in ("l2", "max"):
        raise ValueError(f"agg must be 'l2' or 'max', got {agg!r}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from scored frame: {missing}")
    missing_fit = [c for c in feature_cols if c not in fit_df.columns]
    if missing_fit:
        raise ValueError(f"feature columns missing from fit frame: {missing_fit}")

    keys = ["case_id", "endpoint_key"]
    means = fit_df.groupby(keys, sort=False)[feature_cols].mean().reset_index()
    stds = fit_df.groupby(keys, sort=False)[feature_cols].std().reset_index()

    mu = df[keys].merge(means, on=keys, how="left", validate="many_to_one")[feature_cols]
    sd = df[keys].merge(stds, on=keys, how="left", validate="many_to_one")[feature_cols]
    mu_arr = mu.to_numpy(dtype=float)
    sd_arr = sd.to_numpy(dtype=float)
    # 可观测性日志（只记数，不改行为）：组缺席行会静默按 0 偏离处理、std 退化
    # 单元格会静默落 1.0 哨兵——两类退化都曾以"指标照出、stage 照绿"的形式漏过。
    absent = (
        df[keys].merge(means[keys], on=keys, how="left", indicator=True)["_merge"].eq("left_only")
    )
    n_absent = int(absent.sum())
    if n_absent:
        logger.info(
            "z-score：%d 行的 (case_id, endpoint_key) 组在 fit_df 中缺席，按 0 偏离处理",
            n_absent,
        )
    sentinel = (~np.isfinite(sd_arr)) | (sd_arr == 0.0)
    if sentinel.any():
        per_col = pd.Series(sentinel.sum(axis=0), index=feature_cols)
        per_col = per_col[per_col > 0]
        logger.info("z-score：std 退化落 1.0 哨兵的单元格数（按列）：%s", per_col.to_dict())
    # NaN（组不存在）也落到 1.0；此时 mu 同为 NaN，z 仍是 NaN，下游统一归零。
    # std 为 NaN 还有第二来源——fit 组只有 1 行（ddof=1）：此时 mu 有限，
    # z = x − 单点值按原始量纲计入，不归零（见 docstring 退化规则）。
    sd_arr = np.where((~np.isfinite(sd_arr)) | (sd_arr == 0.0), 1.0, sd_arr)

    x = df[feature_cols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        z = (x - mu_arr) / sd_arr
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    if agg == "l2":
        score = np.sqrt(np.sum(z**2, axis=1))
    else:
        score = np.max(np.abs(z), axis=1)

    return pd.DataFrame(
        {
            "sample_id": df["sample_id"].astype(str),
            "score": score.astype(float),
            "y_true": df["is_endpoint_anomaly"].astype(int),
        }
    )


def _feature_cols_from_schema(schema: dict) -> list[str]:
    """从 schema.json 的 feature_groups 保序派生特征列。

    不按列名前缀扫描——build_contract.py 曾因前缀扫描产生 schema 维度错位
    （PR #25），以契约声明为唯一准绳。
    """
    cols: list[str] = []
    for group in schema["feature_groups"].values():
        cols.extend(group["columns"])
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, choices=_BASELINE_CHOICES)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    contract_dir: Path = args.contract_dir
    eval_df = pd.read_parquet(contract_dir / "eval_all.parquet")
    logger.info("Loaded %d eval rows from %s", len(eval_df), contract_dir / "eval_all.parquet")

    if args.baseline == "rel_pos":
        scores = compute_rel_pos_scores(eval_df)
    else:
        train_df = pd.read_parquet(contract_dir / "train.parquet")
        missing_cases = set(eval_df["case_id"].unique()) - set(train_df["case_id"].unique())
        if missing_cases:
            raise ValueError(
                f"train.parquet 缺少 {len(missing_cases)} 个 eval case：{sorted(missing_cases)}。"
                "这通常说明 --contract-dir 指到了 expand_train_pool=false 的 contract"
                "（训练池只有 Normal case）——z-score 基线在缺席 case 上会静默全 0 分、"
                "宏平均恰为 0.5 且 stage 仍显示成功。"
            )
        schema = json.loads((contract_dir / "schema.json").read_text())
        feature_cols = _feature_cols_from_schema(schema)
        logger.info(
            "z-score lean 口径：%d 个特征，统计量仅取 train.parquet %d 行",
            len(feature_cols),
            len(train_df),
        )
        agg = "l2" if args.baseline == "zscore_l2" else "max"
        scores = compute_zscore_scores(eval_df, train_df, feature_cols, agg)

    # 3 列函数输出 + 7 个诊断列在落盘层装配成 10 列，供 eval 原样消费
    # （接口墙不打洞：eval 仍只读 scores.parquet）。
    diag = eval_df[["sample_id", *DIAGNOSTIC_COLUMNS]].copy()
    out_df = diag.merge(scores[["sample_id", "score"]], on="sample_id", validate="one_to_one")
    # inner merge 必须保行：每个 eval 行都该拿到一个分数，少了说明 sample_id 对不上
    # （不用 assert：`python -O` 会把 assert 剥掉，静默放行缺行产物）
    if len(out_df) != len(eval_df):
        raise ValueError(
            f"scores 行数 {len(out_df)} != eval_all 行数 {len(eval_df)}：有 eval 行未匹配到分数"
        )
    out_df["y_true"] = out_df["is_endpoint_anomaly"].astype(int)
    out_df = out_df[
        [
            "sample_id",
            "score",
            "y_true",
            "case_id",
            "endpoint_key",
            "phase",
            "anomaly_type",
            "anomaly_level",
            "is_endpoint_anomaly",
            "label_granularity",
        ]
    ]

    validate_scores_df(out_df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    logger.info("写出 %s scores：%d 行 → %s", args.baseline, len(out_df), args.out)


if __name__ == "__main__":
    main()
