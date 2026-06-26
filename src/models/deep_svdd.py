from __future__ import annotations

import torch
import torch.nn as nn


class DeepSVDD(nn.Module):
    """Deep SVDD 异常检测器（Ruff et al., 2018）。

    encoder 无 bias（防超球退化），center 通过 init_center 初始化，
    作为 buffer 随模型保存和设备迁移。
    """

    center: torch.Tensor | None

    def __init__(self, input_dim: int, hidden_dim: int = 128, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim
        # bias=False: Deep SVDD 论文约束，避免 bias 吸收超球心偏移导致超球退化
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim, bias=False),
        )
        self.register_buffer("center", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @torch.no_grad()
    def init_center(self, x: torch.Tensor, eps: float = 0.1) -> None:
        """用正常数据初始化超球心（取编码均值，防止接近 0 导致退化）。"""
        training_state = self.training
        self.eval()
        z = self.encoder(x)
        c = z.mean(dim=0)
        # 防止 center 分量过于接近 0（原论文建议），避免训练塌缩
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c >= 0)] = eps
        self.center = c.detach().clone()
        self.train(training_state)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """返回每个样本的异常分数（距超球心距离的平方）。"""
        if self.center is None:
            raise RuntimeError("先调用 init_center 初始化超球心")
        z = self.encoder(x)
        return ((z - self.center) ** 2).sum(dim=1)

    def svdd_loss(self, x: torch.Tensor) -> torch.Tensor:
        """SVDD 训练 loss = 平均距离²。"""
        return self.score(x).mean()
