from __future__ import annotations

import torch

from src.fusion.base import FusionModule


class EarlyConcatFusion(FusionModule):
    """Early concatenation：直接拼接所有 modality 特征。顺序固定为 MODALITY_ORDER。"""

    MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")

    def __init__(self, modality_dims: dict[str, int]):
        super().__init__()
        missing = set(self.MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(self.MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {self.MODALITY_ORDER}"
            )
        self._dims = modality_dims

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([modality_dict[m] for m in self.MODALITY_ORDER], dim=-1)

    @property
    def output_dim(self) -> int:
        return sum(self._dims[m] for m in self.MODALITY_ORDER)
