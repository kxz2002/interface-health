from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

# 三个模态的固定顺序，所有融合机制共享（不绑定任何具体子类）。
# 训练脚本据此拼 batch/tensor，顺序错位会导致特征维度错位但不报错——务必保持全局唯一来源。
MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")


class FusionModule(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @classmethod
    def from_contract(
        cls, cfg: DictConfig, *, contract_dir: Path, modality_dims: dict[str, int]
    ) -> "FusionModule":
        """从 contract 产物构造融合机制。默认实现等价于
        hydra.utils.instantiate(cfg, modality_dims=...)，即 L0/L1/L2 的现有行为——
        contract_dir 在默认实现里被忽略。需要读取 contract 侧产物（如 per-endpoint
        基线统计量）的融合机制覆写此方法，把加载逻辑收进自己模块内，训练脚本因此
        无需再按 _target_ 字符串分支为特定机制注入运行时 kwargs。"""
        return hydra.utils.instantiate(cfg, modality_dims=modality_dims)
