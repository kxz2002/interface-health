"""Contract v1 训练池扩容专属：故障 case 的 baseline 阶段行两路时序切分。

修复 issue #16——扩容前的实现把故障 baseline 行整段移入训练池、整段移出 eval_all，
导致 eval_all 负样本骤减、正负比从 ~50:50 崩到 ~84:16。这里改为按时间窗时序切分：
每个 case 内最早 fraction 比例的窗口进训练池，其余留在 eval_all。切分单位是时间窗
（同一窗的所有 endpoint 行不拆散），train 窗全部早于 eval 窗（复用与
split_normal_rows_temporal 一致的防泄漏时序纪律）。

与 split_v1.split_normal_rows_temporal 的差异（故意不共用同一函数）：
- 两路而非三路（train / eval，无 val）
- 对象是故障 case 的 baseline 行，不是 Normal case
- 边界退化不做兜底（见下方注释），而 Normal 切分保证 train_fit 至少 1 窗
"""

from __future__ import annotations

import pandas as pd


def split_fault_baseline_temporal(
    df: pd.DataFrame,
    fraction: float,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把故障 baseline 行切成 (train_part, eval_part) 两个互斥子集。

    调用方须保证 df 仅含故障 case 的 baseline 阶段行（phase == 'baseline'）。
    每个 case 独立按 time_col 排序，最早 int(n * fraction) 个时间窗归 train_part，
    其余归 eval_part。切分单位是时间窗，同一窗的所有行整体归属同一侧。
    """
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()

    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []

    for _case_id, g in df.groupby(case_col, sort=False):
        windows = sorted(g[time_col].unique())
        n = len(windows)
        # int() 自然向下取整，不做"至少 1 窗给 train"的兜底——这与
        # split_normal_rows_temporal 的边界处理刻意不同：那里兜底是因为 train_fit
        # 为空会让 Normalizer 无法 fit；这里训练池已有纯 Normal 的 train_fit 打底，
        # 某个 case 贡献 0 行 baseline 完全不影响训练可行性。反过来，若强行保证
        # 至少 1 窗进 train，对窗数极少的 case 会把仅有的窗吃进 train，让该 case 在
        # eval 里的 baseline 覆盖率归零——正是本次修复要避免的失衡。
        n_train = int(n * fraction)
        train_w = set(windows[:n_train])
        train_frames.append(g[g[time_col].isin(train_w)])
        eval_frames.append(g[~g[time_col].isin(train_w)])

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [f for f in frames if len(f)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else df.iloc[0:0].copy()

    return _concat(train_frames), _concat(eval_frames)
