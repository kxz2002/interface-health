import pytest
import torch.nn as nn

from src.utils.param_budget import solve_hidden_dim_for_param_budget


def _count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def _two_layer(h: int) -> nn.Module:
    # 输入 20 → hidden h → 输出 10，参数量随 h 单调递增
    return nn.Sequential(nn.Linear(20, h), nn.ReLU(), nn.Linear(h, 10))


def test_converges_close_to_target():
    target = _count(_two_layer(128))
    h = solve_hidden_dim_for_param_budget(_two_layer, target, lo=1, hi=1024)
    # 二分找到的 h 构造出的参数量应贴近 target（±该结构单步 hidden 的参数量增量）
    assert abs(_count(_two_layer(h)) - target) <= _count(_two_layer(2)) - _count(_two_layer(1)) + 1


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
