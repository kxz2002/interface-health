"""Deviation-Conditioned Reliability Gate Fusion（reliability-aware 模态路由门控）。

与 L2 GatedFusion 的核心区别：门控输入是"两分支各自相对自身 normal 基线的
偏离摘要"（dev_ep/dev_svc，活在异常度量空间），而非原始特征拼接（L2 活在
特征空间）；归一化用 softmax（竞争性二选一）而非逐维 sigmoid（独立开关）；
融合形式是对称加权 w_ep*value_ep + w_svc*value_svc，而非 L2 的非对称
e_ep + gate*value(e_svc)。详见 spec §2.2 对比表。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule

# |z|>此阈值算作"该特征显著偏离"，用于 dev 向量第3维（偏离特征占比）。
# 2.0 对应约 95% 正态分位数，无强理论依据，是消融维度。
_DEVIATION_THRESHOLD = 2.0


class ReliabilityGatedFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        endpoint_baseline_stats: EndpointBaselineStats,
        id_to_endpoint_key: dict[int, str],
        branch_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._branch_dim = branch_dim
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        self.ep_encoder = nn.Sequential(nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU())
        self.svc_encoder = nn.Sequential(nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU())
        self.value_ep = nn.Linear(branch_dim, branch_dim)
        self.value_svc = nn.Linear(branch_dim, branch_dim)
        # dev_ep(3) + dev_svc(3) = 6 维输入，输出 2 维 logits 供 softmax。
        self.gate_mlp = nn.Sequential(
            nn.Linear(6, branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 2),
        )

    def _deviation_summary(
        self, raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        z = (raw - mean) / std
        norm = z.norm(dim=-1)
        max_abs = z.abs().max(dim=-1).values
        frac_exceed = (z.abs() > _DEVIATION_THRESHOLD).float().mean(dim=-1)
        return torch.stack([norm, max_abs, frac_exceed], dim=-1)

    def gate_weights(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = modality_dict["endpoint_red"].shape[0]
        if endpoint_id is None:
            half = torch.full((batch_size,), 0.5, device=modality_dict["endpoint_red"].device)
            return half, half

        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        dev_eps, dev_svcs = [], []
        for i in range(batch_size):
            eid = int(endpoint_id[i].item())
            if eid not in self._id_to_key:
                raise KeyError(
                    f"endpoint_id={eid} 不在 id_to_endpoint_key 映射中，"
                    "检查上游 endpoint_id 编码是否与 fusion 构造时传入的映射一致"
                )
            key = self._id_to_key[eid]
            ep_mean, ep_std = self._baseline.branch_stats(key, "ep")
            svc_mean, svc_std = self._baseline.branch_stats(key, "svc")
            device = modality_dict["endpoint_red"].device
            dev_eps.append(
                self._deviation_summary(
                    modality_dict["endpoint_red"][i],
                    torch.as_tensor(ep_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(ep_std, dtype=torch.float32, device=device),
                )
            )
            dev_svcs.append(
                self._deviation_summary(
                    svc_in[i],
                    torch.as_tensor(svc_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(svc_std, dtype=torch.float32, device=device),
                )
            )
        dev = torch.cat([torch.stack(dev_eps), torch.stack(dev_svcs)], dim=-1)
        w = torch.softmax(self.gate_mlp(dev), dim=-1)
        return w[:, 0], w[:, 1]

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        w_ep, w_svc = self.gate_weights(modality_dict, endpoint_id)
        e_ep = self.ep_encoder(modality_dict["endpoint_red"])
        svc_in = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)
        e_svc = self.svc_encoder(svc_in)
        return w_ep.unsqueeze(-1) * self.value_ep(e_ep) + w_svc.unsqueeze(-1) * self.value_svc(
            e_svc
        )

    @property
    def output_dim(self) -> int:
        return self._branch_dim
