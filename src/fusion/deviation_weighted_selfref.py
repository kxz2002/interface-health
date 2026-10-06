"""Self-Referential Deviation Fusion（自参照逐特征偏离量加权融合）。

与 `DeviationWeightedFusion`（entry 018）的关系：**公式完全相同**（旧类对
退化列强制 w=1，本类无此特例——v2 的 per-case z-score 回退链已吸收退化，
不需要）；区别只在偏离量从哪来。旧类查 EndpointBaselineStats sidecar 拿
per-endpoint 的 normal-only mean/std 算 z；本类直接把输入特征当偏离量——
因为 contract v2 的 per-case z-score 归一化已使特征值本身就是「相对自身
baseline 的偏离」。

结论因此可以写成：DWF 的公式一开始就是对的，错的是参照系（entry 027）。

本类同时是 max|z| 平凡基线（entry 027）的可微版本——SVDD 学表征、本类做
逐特征软阈值加权。零参数、零状态、无 sidecar，不覆写 from_contract（基类
默认的 hydra instantiate 即可），也因此旧 DWF 依赖的 red_cols/svc_cols 列
顺序强校验与 forward 逐行查表在此一并消失。
"""

from __future__ import annotations

import math

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
        # NaN/inf 也要拒绝：scale<=0 的比较对 NaN 为 False，会让 NaN 温度静默
        # 通过（sigmoid 全 NaN、训练无声坏掉）；inf 阈值则让所有权重恒为 0。
        if not math.isfinite(scale) or scale <= 0:
            # sigmoid 温度必须为正；传 0 会除零，传负会反转「偏离越大权重越高」的语义。
            raise ValueError(f"scale must be a positive finite number, got {scale}")
        if not math.isfinite(threshold):
            raise ValueError(f"threshold must be finite, got {threshold}")
        self._dims = dict(modality_dims)
        self._threshold = float(threshold)
        self._scale = float(scale)

    def forward(
        self, modality_dict: dict[str, torch.Tensor], endpoint_id: torch.Tensor | None = None
    ) -> torch.Tensor:
        # endpoint_id 被接受但恒忽略：参照系是特征自身（v2 per-case z-score），
        # 与 endpoint 身份无关，None 与任意张量结果一致。
        # 键集合或维度不符时 torch.cat 会抛难以定位的 RuntimeError（甚至更糟：
        # 键名拼错时静默丢模态），在此换成点名 missing/extra/维度的可读报错。
        missing = set(self._dims) - set(modality_dict)
        extra = set(modality_dict) - set(self._dims)
        if missing or extra:
            raise ValueError(
                f"modality_dict 键集合与构造时 modality_dims 不符：missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        bad_dims = {
            m: (modality_dict[m].shape[-1], self._dims[m])
            for m in MODALITY_ORDER
            if modality_dict[m].shape[-1] != self._dims[m]
        }
        if bad_dims:
            raise ValueError(f"modality_dict 末维与声明不符（实际, 声明）：{bad_dims}")
        x = torch.cat([modality_dict[m] for m in MODALITY_ORDER], dim=-1)
        w = torch.sigmoid((x.abs() - self._threshold) / self._scale)
        return x * w

    @property
    def output_dim(self) -> int:
        return sum(self._dims[m] for m in MODALITY_ORDER)
