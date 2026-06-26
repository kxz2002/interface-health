from __future__ import annotations

import torch

from src.fusion.base import FusionModule


class EarlyConcatFusion(FusionModule):
    """Early concatenation：直接拼接所有 modality 特征。顺序固定为 MODALITY_ORDER。"""

    MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")

    def __init__(self, modality_dims: dict[str, int]):
        super().__init__()
        self._dims = modality_dims

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([modality_dict[m] for m in self.MODALITY_ORDER], dim=-1)

    @property
    def output_dim(self) -> int:
        return sum(self._dims[m] for m in self.MODALITY_ORDER)
