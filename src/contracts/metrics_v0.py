"""Metrics Contract v0：评测结果 metrics.json 的格式约定。

## 必需字段

| 字段                        | 类型           | 约束                                  |
|-----------------------------|----------------|---------------------------------------|
| protocol_version            | str            | 固定 "v0"                             |
| higher_is_more_anomalous    | bool           | v0 固定 True                          |
| n_samples                   | int            | > 0                                   |
| has_labels                  | bool           | scores 是否含 y_true（v0 内恒为 True；False 分支预留 v1） |
| auroc                       | float or None  | has_labels=True 时 ∈ [0,1]，否则 None |
| auprc                       | float or None  | has_labels=True 时 ∈ [0,1]，否则 None |

## 一致性约束

- has_labels=True  ⇒ auroc 与 auprc 必须是 [0,1] 范围内的数值
- has_labels=False ⇒ auroc 与 auprc 必须为 None

## 可选字段

允许任何额外字段（如 git_commit, timestamp, score 分布统计、分组指标等），
contract 不校验。这是为了让 eval 输出诊断信息时不被 contract 拖累。
"""

from __future__ import annotations

REQUIRED_FIELDS: tuple[str, ...] = (
    "protocol_version",
    "higher_is_more_anomalous",
    "n_samples",
    "has_labels",
    "auroc",
    "auprc",
)


class MetricsContractError(ValueError):
    """Metrics dict 违反 contract 时抛出。"""


def _is_valid_score(x: object) -> bool:
    """auroc/auprc 是否是 [0,1] 内的合法数值（拒绝 bool）。"""
    if isinstance(x, bool):
        return False
    return isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0


def validate_metrics_dict(d: dict) -> None:
    """对 metrics dict 做 contract v0 校验。

    校验失败抛 MetricsContractError，一次性列出所有问题。
    通过则静默返回。
    """
    if not isinstance(d, dict):
        raise MetricsContractError(f"expected dict, got {type(d).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in d]
    if missing:
        raise MetricsContractError(
            f"metrics contract v0 violations: missing required fields: {missing}"
        )

    errors: list[str] = []

    if d["protocol_version"] != "v0":
        errors.append(f'protocol_version must be "v0", got {d["protocol_version"]!r}')

    if not isinstance(d["higher_is_more_anomalous"], bool):
        errors.append(
            f"higher_is_more_anomalous must be bool, got {type(d['higher_is_more_anomalous']).__name__}"
        )
    elif d["higher_is_more_anomalous"] is not True:
        errors.append("higher_is_more_anomalous must be True in v0")

    # n_samples：正整数（拒绝 bool，因为 bool 是 int 的子类）
    n = d["n_samples"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        errors.append(f"n_samples must be a positive int, got {n!r}")

    if not isinstance(d["has_labels"], bool):
        errors.append(f"has_labels must be bool, got {type(d['has_labels']).__name__}")

    # auroc/auprc 与 has_labels 的一致性
    has_labels = d.get("has_labels")
    for metric in ("auroc", "auprc"):
        val = d[metric]
        if has_labels is True:
            if not _is_valid_score(val):
                errors.append(
                    f"{metric} must be a float in [0,1] when has_labels=True, got {val!r}"
                )
        elif has_labels is False:
            if val is not None:
                errors.append(f"{metric} must be None when has_labels=False, got {val!r}")

    if errors:
        raise MetricsContractError("metrics contract v0 violations: " + "; ".join(errors))
