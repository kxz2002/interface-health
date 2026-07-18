"""Deviation-Conditioned Reliability Gate Fusion（reliability-aware 模态路由门控）。

与 L2 GatedFusion 的核心区别：门控输入是"两分支各自相对自身 normal 基线的
偏离摘要"（dev_ep/dev_svc，活在异常度量空间），而非原始特征拼接（L2 活在
特征空间）；归一化用 softmax（竞争性二选一）而非逐维 sigmoid（独立开关）；
融合形式是对称加权 w_ep*value_ep + w_svc*value_svc，而非 L2 的非对称
e_ep + gate*value(e_svc)。详见 spec §2.2 对比表。
"""

from __future__ import annotations

from typing import Literal

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
        deviation_mode: Literal["branch_aware", "scalar"] = "branch_aware",
        gate_normalization: Literal["softmax", "independent_sigmoid", "fixed_uniform"] = "softmax",
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        # Literal[...] 只是类型标注，Python 运行时不校验——typo（如
        # "branch_awer"）会静默落入 deviation_mode == "scalar" 判断的 else
        # 分支，被当成合法的 "branch_aware" 处理，没有任何提示，必须手动兜底。
        valid_deviation_modes = ("branch_aware", "scalar")
        if deviation_mode not in valid_deviation_modes:
            raise ValueError(
                f"deviation_mode must be one of {valid_deviation_modes}, got {deviation_mode!r}"
            )
        valid_gate_normalizations = ("softmax", "independent_sigmoid", "fixed_uniform")
        if gate_normalization not in valid_gate_normalizations:
            raise ValueError(
                f"gate_normalization must be one of {valid_gate_normalizations}, "
                f"got {gate_normalization!r}"
            )
        self._branch_dim = branch_dim
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        self._deviation_mode = deviation_mode
        self._gate_normalization = gate_normalization
        ep_dim = modality_dims["endpoint_red"]
        svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        self.ep_encoder = nn.Sequential(nn.Linear(ep_dim, branch_dim, bias=False), nn.ReLU())
        self.svc_encoder = nn.Sequential(nn.Linear(svc_dim, branch_dim, bias=False), nn.ReLU())
        self.value_ep = nn.Linear(branch_dim, branch_dim)
        self.value_svc = nn.Linear(branch_dim, branch_dim)
        # branch_aware: dev_ep(3) + dev_svc(3) = 6 维输入；scalar 消融变体去掉
        # 分支内部的 3 维细分（norm/max_abs/frac_exceed），每分支只留 1 个总偏离
        # 范数标量，dev_ep(1) + dev_svc(1) = 2 维——验证"该信哪个分支"这个能力
        # 被移除后确实反映在 gate_mlp 的输入维度上，不是只加了个没用的开关。
        # gate_mlp 在 fixed_uniform 消融变体下不参与 forward（见 gate_weights），
        # 但仍在此处构造，保持跨消融变体的模块结构统一。
        gate_in_dim = 2 if deviation_mode == "scalar" else 6
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_dim, 2),
        )

    def _deviation_summary(
        self, raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        z = (raw - mean) / std
        norm = z.norm(dim=-1)
        if self._deviation_mode == "scalar":
            # 消融1：只保留总偏离幅度标量，去掉 max_abs/frac_exceed 这两个
            # "具体哪个特征偏离"的细分信息——门控看不到分支内部结构。
            return norm.unsqueeze(-1)
        max_abs = z.abs().max(dim=-1).values
        frac_exceed = (z.abs() > _DEVIATION_THRESHOLD).float().mean(dim=-1)
        return torch.stack([norm, max_abs, frac_exceed], dim=-1)

    def gate_weights(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = modality_dict["endpoint_red"].shape[0]
        # 两种情形都退化为均匀权重、跳过 dev 计算和 gate_mlp 前向，但触发原因不同：
        # endpoint_id is None 是"没有 endpoint 信息可用"（L0/L1/L2 兼容调用路径）；
        # fixed_uniform 是"消融3：故意禁用门控"（下界对照）。两者语义不同但代码
        # 路径相同，合并成一个分支避免两份几乎相同的早退代码日后改一个漏改另一个。
        if endpoint_id is None or self._gate_normalization == "fixed_uniform":
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
        logits = self.gate_mlp(dev)
        if self._gate_normalization == "independent_sigmoid":
            # 消融2：换成独立 sigmoid（对比 docstring 里 softmax 的"竞争性二选一"）——
            # 权重和不再恒为1。
            w = torch.sigmoid(logits)
        else:
            w = torch.softmax(logits, dim=-1)
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
