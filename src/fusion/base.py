from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    @abstractmethod
    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
