from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

Scope = Literal["per_endpoint", "per_service", "global"]
Method = Literal["min_max"]

_GROUP_COL: dict[Scope, str | None] = {
    "per_endpoint": "endpoint_key",
    "per_service": "service_name",
    "global": None,
}

# fit 集合该 group 只有单一取值（或更少）时 hi-lo 为 0，min-max 无法提供有效 scale。
# 用这个容差判"零方差退化"而不是严格 == 0，同时也覆盖浮点误差下的近零 gap。
_DEGENERATE_GAP_EPS = 1e-9


def _is_degenerate(lo: float, hi: float) -> bool:
    """lo/hi 为 NaN（fit 集合全 NaN）或 hi-lo≈0（fit 集合零方差）时，min-max 无法
    提供有效 scale，两者都应跳过归一化、保留原值——否则 (value-lo)/(hi-lo) 在零方差
    情形下会被除数下限（历史实现里的 1e-9 地板）放大出 1e9~1e13 量级的离谱数值。"""
    return pd.isna(lo) or pd.isna(hi) or (hi - lo) < _DEGENERATE_GAP_EPS


@dataclass
class _Stats:
    scope: Scope
    method: Method
    # key=group_value, value=[min, max]；scope=global 时 key 为 "__global__"
    by_group: dict[str, list[float]]


class Normalizer:
    def __init__(self, rules: dict[str, tuple[Scope, Method]]):
        self.rules = rules
        self._stats: dict[str, _Stats] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col, (scope, method) in self.rules.items():
            group_col = _GROUP_COL[scope]
            by_group: dict[str, list[float]] = {}
            if group_col is None:
                by_group["__global__"] = [float(df[col].min()), float(df[col].max())]
            else:
                for g, sub in df.groupby(group_col):
                    by_group[str(g)] = [float(sub[col].min()), float(sub[col].max())]
            self._stats[col] = _Stats(scope=scope, method=method, by_group=by_group)

    def skipped_groups(self) -> dict[str, list[str]]:
        """返回 fit 阶段统计量退化（全 NaN 或零方差）、transform 时被跳过归一化的
        {列名: [group...]}。

        调用方（如 build_contract 的 RATE_COLUMNS clip 逻辑）需要知道哪些列在哪些
        group 上其实是未归一化的原始量纲，避免对原始量纲值做 [0,1] 语义的裁剪。
        """
        result: dict[str, list[str]] = {}
        for col, stats in self._stats.items():
            groups = [g for g, (lo, hi) in stats.by_group.items() if _is_degenerate(lo, hi)]
            if groups:
                result[col] = groups
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, stats in self._stats.items():
            if col not in out.columns:
                continue
            group_col = _GROUP_COL[stats.scope]
            if group_col is None:
                lo, hi = stats.by_group["__global__"]
                if _is_degenerate(lo, hi):
                    # fit 集合该列全 NaN（如某模态在 Normal 上采集缺失）或零方差（fit
                    # 集合里只出现过单一取值）：前者 lo/hi 本身是 NaN，(value-lo) 天然
                    # 就是 NaN；后者 hi-lo=0，若不跳过会走到下面的除法，被除数下限
                    # （1e-9）放大成 1e9~1e13 量级的离谱数值——两种情形都是"min-max
                    # 无法提供有效 scale"，统一跳过归一化、保留原值。
                    continue
                out[col] = (out[col] - lo) / (hi - lo)
            else:
                # 归一化结果是 float，整列先转 float 避免对 int 列做 mask 赋值触发 dtype 警告
                out[col] = out[col].astype(float)
                # 未知组（fit 时未见过的 group_col 值）保持原值
                for g, (lo, hi) in stats.by_group.items():
                    if _is_degenerate(lo, hi):
                        # 同上：该 group 在 fit 集合里退化（全 NaN 或零方差），跳过归
                        # 一化保留原值，不让退化统计量通过减法/除法把其他 case 里的
                        # 真实数值放大或抹成 NaN
                        continue
                    mask = out[group_col] == g
                    out.loc[mask, col] = (out.loc[mask, col] - lo) / (hi - lo)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    col: {"scope": s.scope, "method": s.method, "by_group": s.by_group}
                    for col, s in self._stats.items()
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        raw = json.loads(Path(path).read_text())
        rules = {col: (spec["scope"], spec["method"]) for col, spec in raw.items()}
        norm = cls(rules)
        norm._stats = {
            col: _Stats(
                scope=spec["scope"],
                method=spec["method"],
                by_group={k: list(v) for k, v in spec["by_group"].items()},
            )
            for col, spec in raw.items()
        }
        return norm
