"""参数量对齐工具：二分查找使模型参数量最接近目标的 hidden_dim。

用于消融实验把不同容量的 baseline（如 L0/L1）参数量对齐到 L2/L3，
排除"参数量差异"这个混淆变量。纯函数，不依赖任何具体模型类。
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def solve_hidden_dim_for_param_budget(
    build_fn: Callable[[int], nn.Module],
    target_params: int,
    lo: int = 1,
    hi: int = 1024,
) -> int:
    """二分查找整数 hidden_dim ∈ [lo, hi]，使 build_fn(h) 的参数量最接近 target_params。

    前提：参数量随 hidden_dim 单调不减（纯 Linear 堆叠满足）。若 target_params 落在
    [count(build_fn(lo)), count(build_fn(hi))] 之外，说明 lo/hi 边界给错了，显式抛
    ValueError——而非静默返回边界值，那样会让调用方拿到一个"最接近但其实差很远"的错误结果。
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) 必须 <= hi ({hi})")

    lo_params = _count_params(build_fn(lo))
    hi_params = _count_params(build_fn(hi))
    if not (lo_params <= target_params <= hi_params):
        raise ValueError(
            f"target_params={target_params} 超出可达范围 [{lo_params}, {hi_params}]"
            f"（hidden_dim ∈ [{lo}, {hi}]），调整 lo/hi 边界"
        )

    # 标准整数二分：收敛到使 count(build_fn(h)) 不小于 target 的最小 h，
    # 再在 h 与 h-1 之间取参数量更接近 target 的那个。
    best = lo
    best_diff = abs(lo_params - target_params)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = _count_params(build_fn(mid))
        diff = abs(p - target_params)
        if diff < best_diff:
            best, best_diff = mid, diff
        if p == target_params:
            return mid
        if p < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return best
