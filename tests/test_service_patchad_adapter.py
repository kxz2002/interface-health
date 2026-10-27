import pytest
import torch

from src.service.modeling.patchad_adapter import patchad_contrastive_loss, patchad_last_window_score


def distributions():
    generator = torch.Generator().manual_seed(42)
    return [
        torch.softmax(torch.randn(2, 4, 6, generator=generator), dim=-1).requires_grad_(),
        torch.softmax(torch.randn(2, 3, 6, generator=generator), dim=-1).requires_grad_(),
    ]


def test_patchad_loss_and_last_point_score_are_finite_and_differentiable():
    output = (distributions(), distributions(), distributions(), distributions(), None)
    loss = patchad_contrastive_loss(output, sequence_length=12, patch_weight=0.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None for group in output[:4] for value in group)
    scores = patchad_last_window_score(output, sequence_length=12)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()


def test_patchad_rejects_nondivisible_patch_lengths():
    output = (distributions(), distributions(), distributions(), distributions(), None)
    with pytest.raises(ValueError, match="divide"):
        patchad_last_window_score(output, sequence_length=10)


def test_patchad_score_uses_upstream_patch_expansion_order():
    first = torch.tensor([[[0.9, 0.1], [0.2, 0.8]]])
    second = torch.tensor([[[0.4, 0.6], [0.7, 0.3]]])
    expanded_first = first[:, [0, 1, 0, 1]]
    expanded_second = second[:, [0, 0, 1, 1]]
    forward = expanded_first * (
        torch.log(expanded_first + 1e-7) - torch.log(expanded_second + 1e-7)
    )
    reverse = expanded_second * (
        torch.log(expanded_second + 1e-7) - torch.log(expanded_first + 1e-7)
    )
    expected = torch.softmax(-(forward + reverse).sum(-1), dim=-1)[:, -1]
    output = ([first], [second], [], [], None)
    torch.testing.assert_close(patchad_last_window_score(output, 4), expected)
