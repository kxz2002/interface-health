import pytest
import torch.nn as nn

from src.utils.param_budget import solve_hidden_dim_for_param_budget


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _two_layer(h: int) -> nn.Module:
    # 输入 20 → hidden h → 输出 10，参数量随 h 单调递增
    return nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 10))


def test_converges_close_to_target_when_no_exact_hit_exists():
    """target 落在两个相邻可达值之间，没有精确解——验证二分查找返回更接近的那个，
    而不是恰好命中的边界情况（原测试用 target=count(128) 是精确命中，没测到这条路径）。"""
    lo_count = _count(_two_layer(64))
    hi_count = _count(_two_layer(65))
    target = lo_count + (hi_count - lo_count) // 3  # 落在区间内，偏向 64 侧，但不等于任何一侧
    h = solve_hidden_dim_for_param_budget(_two_layer, target, lo=1, hi=1024)
    expected = 64 if abs(lo_count - target) <= abs(hi_count - target) else 65
    assert h == expected


def test_exact_hit_returns_that_dim():
    target = _count(_two_layer(64))
    h = solve_hidden_dim_for_param_budget(_two_layer, target, lo=1, hi=1024)
    assert _count(_two_layer(h)) == target


def test_target_below_range_raises():
    below = _count(_two_layer(1)) - 1
    with pytest.raises(ValueError):
        solve_hidden_dim_for_param_budget(_two_layer, below, lo=1, hi=1024)


def test_target_above_range_raises():
    above = _count(_two_layer(1024)) + 1
    with pytest.raises(ValueError):
        solve_hidden_dim_for_param_budget(_two_layer, above, lo=1, hi=1024)


def test_lo_equals_hi_returns_that_point():
    h = solve_hidden_dim_for_param_budget(_two_layer, _count(_two_layer(50)), lo=50, hi=50)
    assert h == 50
