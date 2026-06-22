import torch

from src.utils.seed import set_seed


def test_set_seed_reproducible():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.equal(a, b)
