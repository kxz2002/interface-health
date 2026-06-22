import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置全局随机种子，确保实验可复现。

    必须同时设置所有随机源——只设其中一个会导致不同模块的随机状态不同步，
    在 DataLoader num_workers > 0 或 CUDA 操作时尤其容易出问题。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 关闭非确定性算法；代价是轻微性能下降，换取完全可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
