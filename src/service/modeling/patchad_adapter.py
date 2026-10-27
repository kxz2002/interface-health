"""PatchAD contrastive objective and causal last-window scoring adapter."""

import torch


def _symmetric_divergences(first, second, sequence_length):
    if len(first) != len(second) or not first:
        raise ValueError("PatchAD requires matching nonempty patch distributions")
    first_loss = None
    second_loss = None
    for first_distribution, second_distribution in zip(first, second):
        first_length = first_distribution.shape[1]
        second_length = second_distribution.shape[1]
        if sequence_length % first_length or sequence_length % second_length:
            raise ValueError("Patch distributions must divide the sequence length")
        first_distribution = first_distribution.repeat(1, sequence_length // first_length, 1)
        second_distribution = second_distribution.repeat_interleave(
            sequence_length // second_length, dim=1
        )
        first_distribution = first_distribution / first_distribution.sum(dim=-1, keepdim=True)
        second_distribution = second_distribution / second_distribution.sum(dim=-1, keepdim=True)

        def divergence(left, right):
            return (left * (torch.log(left + 1e-7) - torch.log(right + 1e-7))).sum(dim=-1)

        first_term = divergence(first_distribution, second_distribution.detach()) + divergence(
            second_distribution.detach(), first_distribution
        )
        second_term = divergence(first_distribution.detach(), second_distribution) + divergence(
            second_distribution, first_distribution.detach()
        )
        first_loss = first_term if first_loss is None else first_loss + first_term
        second_loss = second_term if second_loss is None else second_loss + second_term
    return first_loss, second_loss


def patchad_contrastive_loss(output, sequence_length, patch_weight):
    if not 0 <= patch_weight <= 1:
        raise ValueError("Patch weight must be between zero and one")
    patch_number, patch_size, projected_number, projected_size, _ = output
    first, second = _symmetric_divergences(patch_number, projected_size, sequence_length)
    third, fourth = _symmetric_divergences(projected_number, patch_size, sequence_length)
    fifth, sixth = _symmetric_divergences(patch_number, patch_size, sequence_length)
    return -patch_weight * (first.mean() - second.mean() + third.mean() - fourth.mean()) - (
        1 - patch_weight
    ) * (fifth.mean() - sixth.mean()) / len(patch_number)


def patchad_last_window_score(output, sequence_length):
    patch_number, patch_size, _, _, _ = output
    divergence, _ = _symmetric_divergences(patch_number, patch_size, sequence_length)
    return torch.softmax(-divergence / len(patch_number), dim=-1)[:, -1]
