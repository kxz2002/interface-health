"""Self-Referential Deviation Fusion（自参照逐特征偏离量加权融合）。

与 `DeviationWeightedFusion`（entry 018）的关系：**公式完全相同**，区别只在
偏离量从哪来。旧类查 EndpointBaselineStats sidecar 拿 per-endpoint 的
normal-only mean/std 算 z；本类直接把输入特征当偏离量——因为 contract v2
的 per-case z-score 归一化已使特征值本身就是「相对自身 baseline 的偏离」。

结论因此可以写成：DWF 的公式一开始就是对的，错的是参照系（entry 027）。

本类同时是 max|z| 平凡基线（entry 027 实测 per-case macro 0.9403）的可微
版本——SVDD 学表征、本类做软 top-k 偏离聚合。零参数、零状态、无 sidecar，
不覆写 from_contract（基类默认的 hydra instantiate 即可），也因此旧 DWF
依赖的 red_cols/svc_cols 列顺序强校验与 forward 逐行查表在此一并消失。
"""

from __future__ import annotations

import torch

from src.fusion.base import MODALITY_ORDER, FusionModule

_DEFAULT_THRESHOLD = 2.0
_DEFAULT_SCALE = 1.0


class SelfReferentialDeviationFusion(FusionModule):
    def __init__(
        self,
        modality_dims: dict[str, int],
        threshold: float = _DEFAULT_THRESHOLD,
        scale: float = _DEFAULT_SCALE,
    ):
        super().__init__()
        missing = set(MODALITY_ORDER) - set(modality_dims)
        extra = set(modality_dims) - set(MODALITY_ORDER)
        if missing or extra:
            raise ValueError(
                f"modality_dims keys {set(modality_dims)} must match MODALITY_ORDER {MODALITY_ORDER}"
            )
        if scale <= 0:
            # sigmoid 温度必须为正；传 0 会除零，传负会反转「偏离越大权重越高」的语义。
            raise ValueError(f"scale must be positive, got {scale}")
        self._dims = dict(modality_dims)
        self._threshold = float(threshold)
        self._scale = float(scale)

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 被接受但恒忽略：参照系是特征自身（v2 per-case z-score），
        # 与 endpoint 身份无关，None 与任意张量结果一致。
        x = torch.cat([modality_dict[m] for m in MODALITY_ORDER], dim=-1)
        w = torch.sigmoid((x.abs() - self._threshold) / self._scale)
        return x * w

    @property
    def output_dim(self) -> int:
        return sum(self._dims[m] for m in MODALITY_ORDER)
