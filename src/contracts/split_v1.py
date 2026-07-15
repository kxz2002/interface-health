"""Contract v1 三路时序切分：每个 Normal case 内按时间窗切成
train_fit / train_val / eval_normal_holdout 三个互斥子集。

修复 v0 的 train ⊆ eval_all 重叠问题——见 spec 组件 2。
按时间窗（而非 service）切分：每个 endpoint 在三份里都出现（无泛化偏移），
训练窗与评估窗时序分离（评估样本训练时未见过）。seed 参数当前不影响时序切分
（纯按时间排序，确定性），保留是为与 build_contract 的 seed 传参接口一致 + 未来扩展。
"""

from __future__ import annotations

import logging

import pandas as pd

LOG = logging.getLogger(__name__)

_PARTS = ("train_fit", "train_val", "eval_normal_holdout")


def split_normal_rows_temporal(
    df: pd.DataFrame,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
    seed: int = 42,
    fit_frac: float = 0.6,
    val_frac: float = 0.2,
) -> dict[str, pd.DataFrame]:
    """返回 {"train_fit", "train_val", "eval_normal_holdout"} 三个互斥 DataFrame。

    每个 case 独立按 time_col 排序，前 fit_frac 的时间窗 → train_fit，
    接着 val_frac → train_val，其余 → eval_normal_holdout。切分单位是"时间窗"
    （同一窗的所有 endpoint 行不拆散），保证 endpoint 在各份都出现、训练窗早于评估窗。

    窗数不足退化：某 case 时间窗数 < 3 时按可用窗数尽力分配（如 1 窗全进 train_fit，
    2 窗 train_fit+holdout），不足的份对该 case 贡献 0 行。空份仍保留 schema。
    """
    empty = df.iloc[0:0]
    buckets: dict[str, list[pd.DataFrame]] = {p: [] for p in _PARTS}

    for case_id, g in df.groupby(case_col, sort=False):
        windows = sorted(g[time_col].unique())
        n = len(windows)
        n_fit = int(n * fit_frac)
        n_val = int(n * val_frac)
        # 边界处理：至少各 1 窗给 fit；holdout 拿走剩余。窗数极少时 val/holdout 可能为空。
        fit_w = set(windows[:n_fit])
        val_w = set(windows[n_fit : n_fit + n_val])
        hold_w = set(windows[n_fit + n_val :])
        if not fit_w:  # n < 2：全部时间窗归 train_fit，另两份空
            fit_w, val_w, hold_w = set(windows), set(), set()
            LOG.warning("case=%s 只有 %d 个时间窗，全部归入 train_fit", case_id, n)
        buckets["train_fit"].append(g[g[time_col].isin(fit_w)])
        buckets["train_val"].append(g[g[time_col].isin(val_w)])
        buckets["eval_normal_holdout"].append(g[g[time_col].isin(hold_w)])

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [f for f in frames if len(f)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else empty.copy()

    return {p: _concat(buckets[p]) for p in _PARTS}
