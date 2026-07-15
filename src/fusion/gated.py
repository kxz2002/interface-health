from __future__ import annotations

import torch
import torch.nn as nn

from src.fusion.base import MODALITY_ORDER, FusionModule


class GatedFusion(FusionModule):
    """L2：门控条件融合。

    g = sigmoid(gate([e_ep; e_svc]))
    z = e_ep + g ⊙ value(e_svc)

    e_ep 用 endpoint 级信息动态调制被 left-join 广播复制的 service 级
    特征（e_svc）该贡献多少，而非无条件全量拼接（对比 L1）。gate/value
    是门控/调制层，不是表征编码器，bias 保留默认 True
    （区别于 ep_encoder/svc_encoder 的 bias=False 约定）。
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

        self.ep_encoder = nn.Sequential(nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU())
        self.svc_encoder = nn.Sequential(nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU())
        self.gate = nn.Linear(2 * branch_dim, branch_dim)
        self.value = nn.Linear(branch_dim, branch_dim)

    def forward(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        e_ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        e_svc = self.svc_encoder(svc_in)
        g = torch.sigmoid(self.gate(torch.cat([e_ep, e_svc], dim=-1)))
        return e_ep + g * self.value(e_svc)

    @property
    def output_dim(self) -> int:
        return self._branch_dim
