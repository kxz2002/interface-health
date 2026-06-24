"""数据契约模块：train → eval 之间的数据接口约定。

contract 的核心目的是把"模型如何产生分数"和"评测如何计算指标"解耦。
任何模型只要产出符合 contract 的 scores 表，就能直接喂给 eval 脚本，
不需要修改下游代码。

当前版本：v0（单类异常检测，二分类标签）
未来若做多分类或回归任务，新建 v1 模块，老版本保留以保证历史结果可复算。
"""

from src.contracts.metrics_v0 import validate_metrics_dict
from src.contracts.scores_v0 import validate_scores_df

__all__ = ["validate_scores_df", "validate_metrics_dict"]
