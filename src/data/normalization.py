from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

Scope = Literal[
    "per_endpoint",
    "per_service",
    "global",
    "per_case_endpoint",
    "per_case_service",
]
Method = Literal["min_max", "z_score"]

# scope → 分组列。单元素 tuple 是既有单列分组；两元素 tuple 是 per-case 组合键
# （(case_id, endpoint_key) / (case_id, service_name)）；空 tuple 表示全局。
_GROUP_COLS: dict[Scope, tuple[str, ...]] = {
    "per_endpoint": ("endpoint_key",),
    "per_service": ("service_name",),
    "global": (),
    "per_case_endpoint": ("case_id", "endpoint_key"),
    "per_case_service": ("case_id", "service_name"),
}

# per-case 组合键落盘/内存里的连接符。endpoint_key/service_name/case_id 都不
# 含 "::"，join/split 无损。
_COMPOSITE_SEP = "::"

# 零方差退化判定：与既有 min_max 路径共用一个容差语义（见 _is_degenerate）。
_DEGENERATE_GAP_EPS = 1e-9
_DEGENERATE_STD_EPS = 1e-9

# per-case z-score 收缩强度：组内有效样本数 n 与 k 比较，w = n/(n+k)。
# k=10 对齐 EndpointBaselineStats（entry 015）的既有取值——v1 的 per-endpoint
# 偏离量已在同一收缩强度下校准，v2 不新造超参。实测 per-(case,endpoint) fit
# 子集 median 16 行、26/208 组不足 5 行（设计文档 D1），k=10 让这些小样本组
# 主要信任跨 case 汇总层，避免单组 1~4 行的 std 主导尺度。
_SHRINKAGE_K = 10.0

# 回退链终点的 std 哨兵。**必须是 1.0 而非 1e-9**：1e-9 托底除数会把任何非零
# 差值放大 1e9 倍——这是 history entry 013 的 1e9~1e13 爆值，与 entry 027
# RG independent_sigmoid 退化列 z 爆到 1e9 → logits 1e8 → sigmoid 饱和、
# 梯度消失的**共同根因**。1.0 使 z 在链尾退化为"减均值"（无量纲尺度 1），
# 既不除零也不放大，是单位正态尺度下的中性选择。
_FALLBACK_STD_SENTINEL = 1.0

# 合法 scope×method 配对白名单。z_score 的三级回退链要求分组键至少是
# (case_id, endpoint/service) 两级组合，或无分组（global 单级、链尾即哨兵）；
# per_endpoint/per_service 只有单列分组键，回退链（组 → 跨 case 汇总 → 全局）
# 无从定义——配上 z_score 时旧实现 fit 静默成功、transform 才在 group_cols[1]
# 抛裸 IndexError，故在构造期按此表白名单拒绝。
_VALID_SCOPE_METHOD: frozenset[tuple[Scope, Method]] = frozenset(
    {
        ("global", "min_max"),
        ("per_endpoint", "min_max"),
        ("per_service", "min_max"),
        ("global", "z_score"),
        ("per_case_endpoint", "z_score"),
        ("per_case_service", "z_score"),
    }
)


def _is_degenerate(lo: float, hi: float) -> bool:
    """lo/hi 为 NaN（fit 集合全 NaN）或 hi-lo<eps（fit 集合零方差）时，min-max 无法
    提供有效 scale，两者都应跳过归一化、保留原值——否则零方差组会走到 (value-lo)/(hi-lo)
    的除零（inf/NaN）。历史实现用 max(hi-lo, 1e-9) 托底避免除零，反而把差值放大 1e9 倍
    产出 1e9~1e13 量级离谱数值（history 013），故本次改为跳过而非托底。"""
    return pd.isna(lo) or pd.isna(hi) or (hi - lo) < _DEGENERATE_GAP_EPS


@dataclass
class _Stats:
    scope: Scope
    method: Method
    # min_max：key=group_value（per-case 组合键为 "case::endpoint"），value=[min, max]；
    # scope=global 时 key 为 "__global__"。
    # z_score：value=[mean, std, n]（mean/std 已在 fit 期按收缩/回退链解析为有效值，
    # 见 Normalizer._fit_zscore）。
    by_group: dict[str, list[float]]
    # 仅 z_score：汇总回退层。per_case_endpoint 按 endpoint_key 跨 case 汇总；
    # per_case_service 按 service_name 跨 case 汇总；per-case 以外的 z_score 不用。
    # key=末维分组值，value=[mean, std, n]。
    by_endpoint: dict[str, list[float]] = field(default_factory=dict)
    # 仅 z_score：全局回退层（整列跨全部 case/group），[mean, std, n]。
    global_stat: list[float] | None = None


def _composite_key(parts: tuple[str, ...]) -> str:
    return _COMPOSITE_SEP.join(str(p) for p in parts)


def _stat_mean_std_n(series: pd.Series) -> tuple[float, float, int]:
    """skipna 统计量（NaN 不参与 fit，与既有 minmax 的 .min()/.max() skipna 惯例一致）。
    std 用 ddof=1（pandas 默认样本标准差），与 EndpointBaselineStats 同口径。
    返回的 std 是**原始**值（零方差/NaN 不在这里兜底），交由回退链解析。"""
    n = int(series.count())
    mean = float(series.mean()) if n > 0 else math.nan
    std = float(series.std()) if n > 1 else math.nan
    return mean, std, n


class Normalizer:
    def __init__(self, rules: dict[str, tuple[Scope, Method]]):
        illegal = [
            (col, scope, method)
            for col, (scope, method) in rules.items()
            if (scope, method) not in _VALID_SCOPE_METHOD
        ]
        if illegal:
            detail = ", ".join(f"{c}: ({s}, {m})" for c, s, m in illegal)
            raise ValueError(
                f"不支持的 scope×method 配对: {detail}。合法配对："
                "min_max 仅配 global/per_endpoint/per_service；"
                "z_score 仅配 global/per_case_endpoint/per_case_service"
                "（z_score 的两级回退需要 (case_id, endpoint/service) 组合键，"
                "单列分组的 per_endpoint/per_service 无法定义回退链）。"
            )
        self.rules = rules
        self._stats: dict[str, _Stats] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col, (scope, method) in self.rules.items():
            if method == "z_score":
                self._stats[col] = self._fit_zscore(df, col, scope)
            else:
                self._stats[col] = self._fit_minmax(df, col, scope)

    def _fit_minmax(self, df: pd.DataFrame, col: str, scope: Scope) -> _Stats:
        group_cols = _GROUP_COLS[scope]
        by_group: dict[str, list[float]] = {}
        if not group_cols:
            by_group["__global__"] = [float(df[col].min()), float(df[col].max())]
        else:
            # groupby 默认 dropna=True，与旧实现逐行一致（不新增 "nan" 组）
            for keys, sub in df.groupby(list(group_cols)):
                key = keys if isinstance(keys, tuple) else (keys,)
                by_group[_composite_key(key)] = [
                    float(sub[col].min()),
                    float(sub[col].max()),
                ]
        return _Stats(scope=scope, method="min_max", by_group=by_group)

    def _fit_zscore(self, df: pd.DataFrame, col: str, scope: Scope) -> _Stats:
        group_cols = _GROUP_COLS[scope]
        # 最细粒度组统计量（原始值，不兜底）
        raw_groups: dict[str, tuple[float, float, int]] = {}
        if not group_cols:
            raw_groups["__global__"] = _stat_mean_std_n(df[col])
        else:
            for keys, sub in df.groupby(list(group_cols)):
                key = keys if isinstance(keys, tuple) else (keys,)
                raw_groups[_composite_key(key)] = _stat_mean_std_n(sub[col])

        # 回退层：per-case scope 才有"跨 case 汇总 → 全局"两级；
        # 全局 z_score 没有更高层，链尾直接是哨兵。
        by_endpoint_raw: dict[str, tuple[float, float, int]] = {}
        global_mean, global_std, global_n = _stat_mean_std_n(df[col])
        if scope in ("per_case_endpoint", "per_case_service"):
            rollup_col = group_cols[1]
            for ep, sub in df.groupby(rollup_col):
                by_endpoint_raw[str(ep)] = _stat_mean_std_n(sub[col])

        def resolve_std(std: float, fallback_std: float) -> float:
            """std 回退链：本层有效用本层；本层退化（NaN/≈0）用上层；上层也退化
            则最终落到 1.0 哨兵。fallback_std 已是上层解析后的有效值或哨兵。"""
            if not pd.isna(std) and std >= _DEGENERATE_STD_EPS:
                return std
            return fallback_std

        global_effective_std = resolve_std(global_std, _FALLBACK_STD_SENTINEL)

        by_group: dict[str, list[float]] = {}
        for key, (g_mean, g_std, n) in raw_groups.items():
            if not group_cols:
                # 全局 z_score：没有上层可收缩，退化直接用哨兵
                mean_eff = 0.0 if pd.isna(g_mean) else g_mean
                std_eff = resolve_std(g_std, _FALLBACK_STD_SENTINEL)
            else:
                rollup_key = key.split(_COMPOSITE_SEP, 1)[1] if _COMPOSITE_SEP in key else key
                fb = by_endpoint_raw.get(rollup_key)
                if fb is not None:
                    ep_mean, ep_std_raw, _ = fb
                    ep_std = resolve_std(ep_std_raw, global_effective_std)
                    fb_mean, fb_std = ep_mean, ep_std
                else:
                    fb_mean, fb_std = global_mean, global_effective_std

                # mean 一律按 w=n/(n+k) 向上层收缩，**包括零方差大 n 组**——这是
                # 有意偏离 EndpointBaselineStats（EBS 对退化组保留局部均值原值）：
                # 小 n 组的噪声均值收缩到上层是 shrinkage 的本职；大 n 常数列的局部
                # 均值虽可靠，但 (1-w) 级别的偏移有界且轻微，且恰好给"绝对水平"
                # 保留残余信号（资源型故障的信号在绝对水平上，见设计文档 canary），
                # 完全钉死局部均值反而丢掉这一层。n=0（全 NaN）组无组内估计，
                # 等价 w=0 直接取上层均值；上层也 NaN 时落到 0.0 哨兵。
                w = n / (n + _SHRINKAGE_K) if n > 0 else 0.0
                base_mean = g_mean if not pd.isna(g_mean) else fb_mean
                if pd.isna(base_mean):
                    base_mean = 0.0
                fb_mean_eff = fb_mean if not pd.isna(fb_mean) else 0.0
                mean_eff = w * base_mean + (1.0 - w) * fb_mean_eff

                if not pd.isna(g_std) and g_std >= _DEGENERATE_STD_EPS:
                    # 正常组：std 按同一 w 向上层收缩
                    std_eff = w * g_std + (1.0 - w) * fb_std
                else:
                    # 组内零方差/无法估计 std：局部 std 无定义，不取加权平均
                    # （对未定义量加权没有意义）——这一点与 EndpointBaselineStats
                    # 同哲学；直接退化到上层已解析的 std（其内部可能已退化到全局/
                    # 哨兵）。注意分歧只在 mean：std 退化不加权，mean 恒收缩。
                    std_eff = fb_std

            by_group[key] = [float(mean_eff), float(std_eff), float(n)]

        return _Stats(
            scope=scope,
            method="z_score",
            by_group=by_group,
            by_endpoint={
                k: [float(m), float(resolve_std(s, global_effective_std)), float(n)]
                for k, (m, s, n) in by_endpoint_raw.items()
            },
            global_stat=[
                0.0 if pd.isna(global_mean) else float(global_mean),
                float(global_effective_std),
                float(global_n),
            ],
        )

    def skipped_groups(self) -> dict[str, list[str]]:
        """返回 fit 阶段统计量退化（全 NaN 或零方差）、transform 时被跳过归一化的
        {列名: [group...]}。

        仅对 min_max 有意义（min_max 退化 = 跳过归一化保留原值）。z_score 的退化
        由三级回退链吸收（最差也是 1.0 哨兵），不存在"跳过/透传原始量纲"的组，
        故 z_score 列永远不出现在返回值里——per-case 下透传原始量纲是错误的
        （跨 case 不可比），这正是相对 entry 013 跳过策略的实质分歧。
        """
        result: dict[str, list[str]] = {}
        for col, stats in self._stats.items():
            if stats.method != "min_max":
                continue
            groups = [g for g, (lo, hi) in stats.by_group.items() if _is_degenerate(lo, hi)]
            if groups:
                result[col] = groups
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, stats in self._stats.items():
            if col not in out.columns:
                continue
            if stats.method == "z_score":
                out[col] = out[col].astype(float)
                self._transform_zscore(out, col, stats)
            else:
                self._transform_minmax(out, col, stats)
        return out

    def _transform_minmax(self, out: pd.DataFrame, col: str, stats: _Stats) -> None:
        group_cols = _GROUP_COLS[stats.scope]
        if not group_cols:
            lo, hi = stats.by_group["__global__"]
            if _is_degenerate(lo, hi):
                # fit 集合该列全 NaN（如某模态在 Normal 上采集缺失）或零方差（fit
                # 集合里只出现过单一取值）：前者 lo/hi 本身是 NaN，(value-lo) 天然
                # 就是 NaN；后者 hi-lo=0，若不跳过会走到下面的除零（产出 inf/NaN，
                # 历史托底实现则放大成 1e9~1e13 离谱值）——两种情形都是"min-max 无法
                # 提供有效 scale"，统一跳过归一化、保留原值。
                return
            out[col] = (out[col] - lo) / (hi - lo)
            return

        # 归一化结果是 float，整列先转 float 避免对 int 列做 mask 赋值触发 dtype 警告
        out[col] = out[col].astype(float)
        # 未知组（fit 时未见过的 group_col 值）保持原值
        for g, (lo, hi) in stats.by_group.items():
            if _is_degenerate(lo, hi):
                # 同上：该 group 在 fit 集合里退化（全 NaN 或零方差），跳过归
                # 一化保留原值，不让退化统计量通过减法/除法把其他 case 里的
                # 真实数值抹成 NaN 或（零方差除零时）产出 inf/NaN
                continue
            mask = self._group_mask(out, group_cols, g)
            out.loc[mask, col] = (out.loc[mask, col] - lo) / (hi - lo)

    def _transform_zscore(self, out: pd.DataFrame, col: str, stats: _Stats) -> None:
        group_cols = _GROUP_COLS[stats.scope]
        if not group_cols:
            mean, std, _ = stats.by_group["__global__"]
            out[col] = (out[col] - mean) / std
            return

        global_mean, global_std, _ = stats.global_stat  # type: ignore[misc]
        # 每行的最细组键 → 组统计量；未命中（fit 未见过的 case）留 NaN 后逐级回填。
        composite = out[group_cols[0]].astype(str) + _COMPOSITE_SEP + out[group_cols[1]].astype(str)
        grp_mean = composite.map({k: v[0] for k, v in stats.by_group.items()})
        grp_std = composite.map({k: v[1] for k, v in stats.by_group.items()})

        # 第一级回退：该 endpoint/service 跨 case 汇总
        rollup = out[group_cols[1]].astype(str)
        ep_mean = rollup.map({k: v[0] for k, v in stats.by_endpoint.items()})
        ep_std = rollup.map({k: v[1] for k, v in stats.by_endpoint.items()})
        mean = grp_mean.fillna(ep_mean).fillna(global_mean)
        std = grp_std.fillna(ep_std).fillna(global_std)
        # NaN 输入保持 NaN（与既有 minmax 路径的 NaN 惯例一致：算术运算天然透传
        # NaN，均值填补发生在更下游的 ContractDataset，不在 Normalizer）。
        out[col] = (out[col] - mean) / std

    @staticmethod
    def _group_mask(out: pd.DataFrame, group_cols: tuple[str, ...], key: str) -> pd.Series:
        if len(group_cols) == 1:
            return out[group_cols[0]] == key
        # maxsplit=1 与 fit 侧的 key.split(_COMPOSITE_SEP, 1) 保持同一拆法
        values = key.split(_COMPOSITE_SEP, 1)
        mask = pd.Series(True, index=out.index)
        for c, v in zip(group_cols, values):
            mask &= out[c].astype(str) == v
        return mask

    def save(self, path: str | Path) -> None:
        payload: dict[str, dict] = {}
        for col, s in self._stats.items():
            spec: dict = {"scope": s.scope, "method": s.method, "by_group": s.by_group}
            if s.method == "z_score":
                spec["by_endpoint"] = s.by_endpoint
                spec["global"] = s.global_stat
                # 记录性字段：落盘当时的收缩强度，仅供审计/溯源；load 不读它——
                # by_group 里存的是已解析的有效 mean/std（收缩在 fit 期完成），
                # 重载后 transform 无需也不会重放收缩，故改 k 不影响旧产物的复现。
                spec["shrinkage_k"] = _SHRINKAGE_K
            payload[col] = spec
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        raw = json.loads(Path(path).read_text())
        rules = {col: (spec["scope"], spec["method"]) for col, spec in raw.items()}
        norm = cls(rules)
        norm._stats = {
            col: _Stats(
                scope=spec["scope"],
                method=spec["method"],
                by_group={k: list(v) for k, v in spec["by_group"].items()},
                # 旧格式（min_max，v1 产物）没有这两个键，缺省即空/None——
                # min_max 路径从不读取它们，旧 JSON 无损可读。
                by_endpoint={k: list(v) for k, v in spec.get("by_endpoint", {}).items()},
                global_stat=list(spec["global"]) if spec.get("global") is not None else None,
            )
            for col, spec in raw.items()
        }
        return norm
