"""Scores Contract v0：模型推断输出表的格式约定。

## 必需列（v0 强制校验）

| 列名        | dtype  | 约束                                          |
|-------------|--------|-----------------------------------------------|
| sample_id   | str    | 全表唯一，无空值                              |
| score       | float  | 无 NaN/inf，越大越异常（higher_is_more_anomalous=True） |
| y_true      | int    | 取值严格 ∈ {0, 1}（0=正常，1=异常）           |

## 可选列（不校验，允许存在）

诊断/分组用，例如：`case_id`, `endpoint_key`, `phase`, `timestamp_window`, `target_service`。
contract 不感知这些列，但 eval 未来可用它们做分组指标（按 case 的 AUROC 等）。

## 约定

- score 方向：固定 `higher_is_more_anomalous=True`。如果模型内部输出反向分数
  （如重构误差的负值），必须在生成 scores.parquet 之前由训练代码翻转。
- NaN 由训练阶段处理：缺特征/空窗口等情况必须在写 scores 前完成填充或丢弃，
  contract 不允许 NaN，以避免下游 dropna 带来的样本数不一致。
- sample_id 拼接惯例：`f"{case_id}__{endpoint_key}__{timestamp_window}"`。
  contract 只校验唯一性，不校验拼接格式——具体规则由训练代码维护。
"""

from __future__ import annotations

import math
from typing import NewType

import pandas as pd

ScoresV0 = NewType("ScoresV0", pd.DataFrame)

REQUIRED_COLUMNS: dict[str, str] = {
    "sample_id": "str",
    "score": "float",
    "y_true": "int",
}


class ScoresContractError(ValueError):
    """Scores 表违反 contract 时抛出。错误信息会列出所有发现的问题。"""


def validate_scores_df(df: pd.DataFrame) -> ScoresV0:
    """对 scores DataFrame 做 contract v0 校验。

    校验失败时抛 ScoresContractError，错误信息一次性列出所有问题
    （而非发现第一个就 raise），方便调用方一次定位多个错误。

    校验通过返回 ScoresV0（运行时等价于原 df，仅类型层面标记"已校验"）。
    """
    if not isinstance(df, pd.DataFrame):
        raise ScoresContractError(f"expected pandas.DataFrame, got {type(df).__name__}")

    errors: list[str] = []

    if len(df) == 0:
        errors.append("scores df is empty (need at least 1 row)")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")
        # 缺列后续校验无意义，直接报告并返回
        raise ScoresContractError("scores contract v0 violations: " + "; ".join(errors))

    # sample_id：字符串 + 唯一 + 无空
    sid = df["sample_id"]
    if not (pd.api.types.is_string_dtype(sid) or sid.dtype == object):
        errors.append(f"sample_id dtype must be string-like, got {sid.dtype}")
    if sid.isna().any():
        errors.append(f"sample_id contains {int(sid.isna().sum())} NaN values")
    n_dup = int(sid.duplicated().sum())
    if n_dup > 0:
        errors.append(f"sample_id has {n_dup} duplicate values (must be unique)")

    # score：可转 float，无 NaN/inf
    score = df["score"]
    try:
        score_f = score.astype(float)
    except (TypeError, ValueError) as exc:
        errors.append(f"score column cannot be cast to float: {exc}")
    else:
        n_nan = int(score_f.isna().sum())
        if n_nan > 0:
            errors.append(f"score has {n_nan} NaN values (NaN disallowed by v0)")
        is_inf = score_f.apply(lambda x: math.isinf(x) if pd.notna(x) else False)
        n_inf = int(is_inf.sum())
        if n_inf > 0:
            errors.append(f"score has {n_inf} inf values (disallowed)")

    # y_true：值严格 ∈ {0, 1}
    y = df["y_true"]
    if y.isna().any():
        errors.append(f"y_true contains {int(y.isna().sum())} NaN values")
    elif not (
        pd.api.types.is_integer_dtype(y)
        or pd.api.types.is_float_dtype(y)
        or pd.api.types.is_bool_dtype(y)
    ):
        errors.append(f"y_true dtype must be numeric (int/float/bool), got {y.dtype}")
    else:
        try:
            y_int = y.astype(int)
        except (TypeError, ValueError) as exc:
            errors.append(f"y_true cannot be cast to int: {exc}")
        else:
            # 浮点型若存在小数部分则非法（如 0.5）
            if pd.api.types.is_float_dtype(y) and not (y == y_int).all():
                errors.append("y_true is float-typed but contains non-integer values")
            bad = set(y_int.unique()) - {0, 1}
            if bad:
                errors.append(f"y_true contains values outside {{0, 1}}: {sorted(bad)}")

    if errors:
        raise ScoresContractError("scores contract v0 violations: " + "; ".join(errors))
    return ScoresV0(df)
