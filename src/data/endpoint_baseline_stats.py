"""每 endpoint 一份的 RED/SVC 双分支 normal-only mean/std 统计量。

供 ReliabilityGatedFusion 计算偏离量（z-score）使用。只在 train_fit 上 fit，
拟合后固定、不参与 backprop——与 Normalizer 的防泄漏纪律一致（build_contract.py
的 fit_df 收窄到 train_fit 约定，见该文件 L369-381 的注释）。

退化处理与 src/data/normalization.py::_is_degenerate 同一哲学但落点不同：
Normalizer 遇到零方差退化时"跳过归一化保留原值"；这里偏离量计算不允许 NaN
分量（会破坏 dev 向量固定维度、传播进 softmax），故退化列改为直接退化到
global std（等价于 shrinkage k→inf 的极限），而非跳过。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

Branch = Literal["ep", "svc"]

_DEGENERATE_STD_EPS = 1e-9


@dataclass
class _EndpointBranchStats:
    mean: np.ndarray
    std: np.ndarray


class EndpointBaselineStats:
    def __init__(
        self,
        red_cols: list[str],
        svc_cols: list[str],
        shrinkage_k: float = 10.0,
        use_shrinkage: bool = True,
    ):
        self._red_cols = list(red_cols)
        self._svc_cols = list(svc_cols)
        self._shrinkage_k = shrinkage_k
        self._use_shrinkage = use_shrinkage
        self._stats: dict[str, dict[Branch, _EndpointBranchStats]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        branch_cols: dict[Branch, list[str]] = {"ep": self._red_cols, "svc": self._svc_cols}
        # ddof=1（pandas 默认，样本标准差）而非 ddof=0——与测试期望的 df[...].std()
        # 口径一致，且样本量小的稀疏 endpoint 用有偏估计（ddof=0）会系统性低估方差。
        global_stats = {
            branch: (df[cols].mean().to_numpy(), df[cols].std().to_numpy())
            for branch, cols in branch_cols.items()
        }

        self._stats = {}
        for ep, group in df.groupby("endpoint_key"):
            n = len(group)
            per_branch: dict[Branch, _EndpointBranchStats] = {}
            for branch, cols in branch_cols.items():
                local_mean = group[cols].mean().to_numpy()
                local_std = group[cols].std().to_numpy()  # ddof=1，同上
                g_mean, g_std = global_stats[branch]

                degenerate = np.isnan(local_std) | (local_std < _DEGENERATE_STD_EPS)
                std = local_std.copy()
                mean = local_mean.copy()
                # 退化列（全 NaN / 零方差）直接退化到 global 统计量，
                # 不参与 shrinkage 混合（shrinkage 对一个未定义的局部估计值取
                # 加权平均没有意义），也不保留 NaN（下游 z-score 计算会传染 NaN）。
                std[degenerate] = g_std[degenerate]
                mean[degenerate] = np.where(
                    np.isnan(mean[degenerate]), g_mean[degenerate], mean[degenerate]
                )

                if self._use_shrinkage:
                    w = n / (n + self._shrinkage_k)
                    non_degenerate = ~degenerate
                    std[non_degenerate] = w * std[non_degenerate] + (1 - w) * g_std[non_degenerate]
                    mean[non_degenerate] = (
                        w * mean[non_degenerate] + (1 - w) * g_mean[non_degenerate]
                    )

                per_branch[branch] = _EndpointBranchStats(mean=mean, std=std)
            self._stats[str(ep)] = per_branch

    def branch_stats(self, endpoint_key: str, branch: Branch) -> tuple[np.ndarray, np.ndarray]:
        if endpoint_key not in self._stats:
            raise KeyError(
                f"endpoint_key={endpoint_key!r} 未出现在 fit 集合（train_fit）中，"
                "无法计算偏离量——检查上游 endpoint_id 映射是否与 fit 时一致"
            )
        s = self._stats[endpoint_key][branch]
        return s.mean, s.std

    def save(self, path: str | Path) -> None:
        payload = {
            "red_cols": self._red_cols,
            "svc_cols": self._svc_cols,
            "shrinkage_k": self._shrinkage_k,
            "use_shrinkage": self._use_shrinkage,
            "stats": {
                ep: {
                    branch: {"mean": s.mean.tolist(), "std": s.std.tolist()}
                    for branch, s in per_branch.items()
                }
                for ep, per_branch in self._stats.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EndpointBaselineStats":
        raw = json.loads(Path(path).read_text())
        obj = cls(
            red_cols=raw["red_cols"],
            svc_cols=raw["svc_cols"],
            shrinkage_k=raw["shrinkage_k"],
            use_shrinkage=raw["use_shrinkage"],
        )
        obj._stats = {
            ep: {
                branch: _EndpointBranchStats(mean=np.array(s["mean"]), std=np.array(s["std"]))
                for branch, s in per_branch.items()
            }
            for ep, per_branch in raw["stats"].items()
        }
        return obj
