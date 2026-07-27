"""Deviation-Weighted Fusion（逐特征偏离量加权融合）。

与 `ReliabilityGatedFusion` 的核心区别：本类零可学习参数（无 `nn.Linear`
encoder、无 gate_mlp、无 softmax/sigmoid 分支级门控），权重直接由
`EndpointBaselineStats.branch_stats()` 给出的 mean/std 算出，粒度是**逐特征**
（每一维单独一个 sigmoid(|z|-threshold) 权重）而非 RG 的**分支级**（整个
ep/svc 分支共享一个标量权重）。forward 输出直接是加权后的原始特征拼接
（维度 = ep_dim + svc_dim），不经过任何 encoder/value 投影——设计目标是
验证"逐特征加权"这一最小改动本身能否缓解信号被平均稀释的问题（entry 017），
不引入模型容量变化作为混淆变量，故不参考 RG 的复杂度。
"""

from __future__ import annotations

import torch

from src.data.endpoint_baseline_stats import EndpointBaselineStats
from src.fusion.base import MODALITY_ORDER, FusionModule

# |z|>此阈值算作"该特征显著偏离"，复用 RG 的既有显著性标准（见
# reliability_gate.py::_DEVIATION_THRESHOLD），非分支级门控输入而是
# 直接作用于逐特征权重公式 sigmoid(|z|-threshold)。
_DEFAULT_THRESHOLD = 2.0


class DeviationWeightedFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        endpoint_baseline_stats: EndpointBaselineStats,
        id_to_endpoint_key: dict[int, str],
        red_cols: list[str],
        svc_cols: list[str],
        threshold: float = _DEFAULT_THRESHOLD,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        self._baseline = endpoint_baseline_stats
        self._id_to_key = dict(id_to_endpoint_key)
        self._threshold = threshold
        self._ep_dim = modality_dims["endpoint_red"]
        self._svc_dim = modality_dims["service_metric"] + modality_dims["service_log"]

        # 退化列（EndpointBaselineStats 全局 fallback 到 1e-9 std 的列）的权重必须
        # 固定为1，不能走 sigmoid(|z|-threshold) 公式——否则一个从未真正偏离过的
        # 常数列，只因分母被兜底成 1e-9 而在 z 计算里引入数值噪声，被误判为需要
        # 衰减（见 test_degenerate_column_weight_fixed_to_one 的手算示例）。
        # register_buffer 而非普通属性：需要跟随 .to(device) 迁移，但不是可学习参数。
        ep_degenerate = set(endpoint_baseline_stats.degenerate_columns("ep"))
        svc_degenerate = set(endpoint_baseline_stats.degenerate_columns("svc"))
        self.register_buffer(
            "_ep_degenerate_mask",
            torch.tensor([c in ep_degenerate for c in red_cols], dtype=torch.bool),
        )
        self.register_buffer(
            "_svc_degenerate_mask",
            torch.tensor([c in svc_degenerate for c in svc_cols], dtype=torch.bool),
        )

    def _weighted(
        self,
        raw: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        degenerate_mask: torch.Tensor,
    ) -> torch.Tensor:
        z = (raw - mean) / std
        w = torch.sigmoid(z.abs() - self._threshold)
        w = torch.where(degenerate_mask, torch.ones_like(w), w)
        return raw * w

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        raw_ep = modality_dict["endpoint_red"]
        raw_svc = torch.cat([modality_dict["service_metric"], modality_dict["service_log"]], dim=-1)

        # 没有 endpoint 信息时退化为纯 concat（等价 L0），兼容不携带 endpoint_id
        # 的既有调用路径（spec §2.5）——与 RG 的同款早退语义一致。
        if endpoint_id is None:
            return torch.cat([raw_ep, raw_svc], dim=-1)

        device = raw_ep.device
        weighted_eps, weighted_svcs = [], []
        for i in range(raw_ep.shape[0]):
            eid = int(endpoint_id[i].item())
            if eid not in self._id_to_key:
                raise KeyError(
                    f"endpoint_id={eid} 不在 id_to_endpoint_key 映射中，"
                    "检查上游 endpoint_id 编码是否与 fusion 构造时传入的映射一致"
                )
            key = self._id_to_key[eid]
            ep_mean, ep_std = self._baseline.branch_stats(key, "ep")
            svc_mean, svc_std = self._baseline.branch_stats(key, "svc")
            weighted_eps.append(
                self._weighted(
                    raw_ep[i],
                    torch.as_tensor(ep_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(ep_std, dtype=torch.float32, device=device),
                    self._ep_degenerate_mask,
                )
            )
            weighted_svcs.append(
                self._weighted(
                    raw_svc[i],
                    torch.as_tensor(svc_mean, dtype=torch.float32, device=device),
                    torch.as_tensor(svc_std, dtype=torch.float32, device=device),
                    self._svc_degenerate_mask,
                )
            )
        return torch.cat([torch.stack(weighted_eps), torch.stack(weighted_svcs)], dim=-1)

    @property
    def output_dim(self) -> int:
        return self._ep_dim + self._svc_dim
