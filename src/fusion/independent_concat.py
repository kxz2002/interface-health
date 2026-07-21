from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import MODALITY_ORDER, FusionModule


class IndependentConcatFusion(FusionModule):
    """L1：endpoint/service 两路各过独立单层编码器后拼接，无门控交互。

    用于把 L0(裸拼接零参数)→L2(门控) 的性能提升，与"是否引入任意非线性容量"
    这一混淆变量分离——L1 提供有非线性容量但无门控设计的对照组。
    """

    def __init__(self, modality_dims: dict[str, int], branch_dim: int = 16):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._branch_dim = branch_dim
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        # bias=False：与 DeepSVDD.encoder 的约定一致，避免 bias 吸收偏移。
        self.ep_encoder = nn.Sequential(nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU())
        self.svc_encoder = nn.Sequential(nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU())

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 接受即忽略(见 EarlyConcatFusion.forward 注释)。
        ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        svc = self.svc_encoder(svc_in)
        return torch.cat([ep, svc], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self._branch_dim
