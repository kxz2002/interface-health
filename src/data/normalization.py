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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, stats in self._stats.items():
            if col not in out.columns:
                continue
            group_col = _GROUP_COL[stats.scope]
            if group_col is None:
                lo, hi = stats.by_group["__global__"]
                out[col] = (out[col] - lo) / max(hi - lo, 1e-9)
            else:
                # 归一化结果是 float，整列先转 float 避免对 int 列做 mask 赋值触发 dtype 警告
                out[col] = out[col].astype(float)
                # 未知组（fit 时未见过的 group_col 值）保持原值
                for g, (lo, hi) in stats.by_group.items():
                    mask = out[group_col] == g
                    out.loc[mask, col] = (out.loc[mask, col] - lo) / max(hi - lo, 1e-9)
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
