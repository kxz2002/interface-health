from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# 三个模态的固定顺序，所有融合机制共享（不绑定任何具体子类）。
# 训练脚本据此拼 batch/tensor，顺序错位会导致特征维度错位但不报错——务必保持全局唯一来源。
MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...
