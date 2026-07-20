import numpy as np
import pandas as pd
import pytest

from src.data.endpoint_baseline_stats import (
    _FALLBACK_MEAN_SENTINEL,
    _FALLBACK_STD_EPSILON,
    EndpointBaselineStats,
)

RED_COLS = [f"endpoint_red__f{i}" for i in range(3)]
SVC_COLS = [f"service_metric__g{i}" for i in range(2)]


def _synth_df(n_per_ep: dict[str, int], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for ep, n in n_per_ep.items():
        for _ in range(n):
            row = {"endpoint_key": ep}
            for c in RED_COLS + SVC_COLS:
                row[c] = float(rng.normal(loc=10.0, scale=2.0))
            rows.append(row)
    return pd.DataFrame(rows)


def test_fit_produces_mean_std_per_endpoint_per_branch():
    df = _synth_df({"ep1": 50, "ep2": 50})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    for ep in ("ep1", "ep2"):
        ep_mean, ep_std = stats.branch_stats(ep, "ep")
        svc_mean, svc_std = stats.branch_stats(ep, "svc")
        assert ep_mean.shape == (len(RED_COLS),)
        assert ep_std.shape == (len(RED_COLS),)
        assert svc_mean.shape == (len(SVC_COLS),)
        assert svc_std.shape == (len(SVC_COLS),)
        assert (ep_std > 0).all()


def test_degenerate_group_zero_std_falls_back_to_global_std():
    """某 endpoint 某列在 fit 集合里零方差（std=0），复用 Normalizer 同款
    退化哲学：不能直接拿 std=0 去做 z-score（除零），必须退化处理——这里退化
    策略是整列直接用 global std（等价于 shrinkage k→inf 的极限），而非跳过该列
    （偏离量计算不能有 NaN 分量，跳过会破坏 dev 向量的固定维度）。"""
    df = _synth_df({"ep-normal": 50})
    degenerate_col = RED_COLS[0]
    df2 = pd.concat([df, _synth_df({"ep-degenerate": 5}, seed=1)], ignore_index=True)
    df2.loc[df2["endpoint_key"] == "ep-degenerate", degenerate_col] = 7.0  # 常数

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df2)
    ep_mean, ep_std = stats.branch_stats("ep-degenerate", "ep")
    idx = RED_COLS.index(degenerate_col)
    global_std = df2[degenerate_col].std()
    assert ep_std[idx] == pytest.approx(global_std, rel=1e-6)


def test_globally_degenerate_column_falls_back_to_epsilon_not_zero():
    """当某列在整个 fit 集合（跨所有 endpoint）都是零方差时，"退化到 global std"
    这条兜底路径本身失效——global std 也是 0，直接复用会把 0 静默传播到
    branch_stats()，下游 z-score = (raw-mean)/std 除零产出 inf/NaN。

    这正是 history-013（Normalizer 零方差除零放大 bug）在这里的镜像场景：真实
    数据里 trace_5xx_rate / net_rx_error_rate 等列在正常运行期间全局恒为 0，
    但故障期间会跳变——这些恰恰是异常信号最强的列，绝不能因为兜底失效被
    silently 破坏。二级兜底选固定小正数 epsilon（而非跳过或保留 0/NaN），
    因为偏离量计算要求 std 永远非零、非 NaN（固定维度 dev 向量不允许缺失分量），
    且该列本身无真实方差信号可言，epsilon 只是保证除法运算有定义。"""
    df = _synth_df({"ep1": 50, "ep2": 30})
    constant_col = RED_COLS[0]
    df[constant_col] = 5.0  # 全局常数，std=0 覆盖所有 endpoint

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    idx = RED_COLS.index(constant_col)
    for ep in ("ep1", "ep2"):
        _, ep_std = stats.branch_stats(ep, "ep")
        assert ep_std[idx] > 0
        assert not np.isnan(ep_std[idx])


def test_degenerate_columns_reports_globally_degenerate_column():
    """degenerate_columns() 是退化列的可见性入口，类比 Normalizer.skipped_groups()——
    调用方（build_contract.py / ReliabilityGatedFusion）需要知道哪些列的 std 是
    epsilon 兜底而非真实统计量，否则会误以为该列在做正常的 z-score。"""
    df = _synth_df({"ep1": 50, "ep2": 30})
    constant_col = RED_COLS[0]
    df[constant_col] = 5.0

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    assert constant_col in stats.degenerate_columns("ep")
    for col in RED_COLS[1:]:
        assert col not in stats.degenerate_columns("ep")
    assert stats.degenerate_columns("svc") == []


def test_globally_all_nan_column_falls_back_to_sentinel_mean_and_epsilon_std():
    """区别于 test_globally_degenerate_column_falls_back_to_epsilon_not_zero（该测试
    用零方差但非 NaN 的常数列覆盖 std 的退化路径），这里构造一列在整个 fit 集合
    （跨所有 endpoint 的所有行）全 NaN——这是模块 docstring 明确点名的真实生产
    bug 场景（Task 7 e2e 冒烟测试实测命中：某 service 从未命中过 log join，
    对应列全局 NaN，NaN mean 传入 z-score 分子直接产出 NaN loss）。

    全 NaN 列同时触发两条独立的兜底截断：
    - global std 是 NaN → 命中 fit() 里 `global_degenerate` 判定 → g_std 截断为
      _FALLBACK_STD_EPSILON（第 82 行）
    - global mean 是 NaN → 命中 `np.isnan(g_mean)` 判定 → g_mean 截断为
      _FALLBACK_MEAN_SENTINEL（第 85 行），这条截断只看 mean 自己是否 NaN，
      与 std 的退化判定是独立的两条逻辑

    对每个 endpoint，该列局部 std 也必然是 NaN（全 NaN 列的任何子集统计量都是
    NaN）→ 命中局部 `degenerate` 判定（第 96 行）→ std/mean 都直接赋值为已截断
    的 global 值（第 103-106 行），且这一赋值发生在 shrinkage 混合之前，
    shrinkage 只对 `non_degenerate` 索引生效（第 110-114 行）——因此不存在
    "epsilon 被 shrinkage 稀释成别的值"的可能，退化列的最终值必须精确等于
    _FALLBACK_STD_EPSILON 和 _FALLBACK_MEAN_SENTINEL，不是近似。
    """
    df = _synth_df({"ep1": 50, "ep2": 30})
    nan_col = RED_COLS[0]
    df[nan_col] = np.nan  # 全局全 NaN，而非常数

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    idx = RED_COLS.index(nan_col)
    for ep in ("ep1", "ep2"):
        ep_mean, ep_std = stats.branch_stats(ep, "ep")
        assert ep_mean[idx] == pytest.approx(_FALLBACK_MEAN_SENTINEL)
        assert ep_std[idx] == pytest.approx(_FALLBACK_STD_EPSILON)
        assert ep_std[idx] > 0
        assert not np.isnan(ep_std[idx])
        assert not np.isnan(ep_mean[idx])
    assert nan_col in stats.degenerate_columns("ep")


def test_locally_nan_column_falls_back_to_well_defined_global_stats():
    """区别于上一测试（全局也退化，兜底到 epsilon/sentinel），这里构造仅对
    *某一个* endpoint 局部退化（该 endpoint 的列全 NaN），但另一个 endpoint
    在同一列上有正常方差数据，使得 global std/mean 本身不退化——此时应该
    退化到"有效的 global 统计量"，而不是掉到 epsilon/sentinel 兜底值。

    这条路径对应 fit() 第 96/103-106 行：局部 degenerate 判定只看 local_std
    是否 NaN/零方差，不管 global 是否也退化；global_stats 在这里因为
    ep-has-data 贡献了非退化的行，不会触发第 73-85 行的 global 截断，
    所以 g_mean/g_std 是 df 里非 NaN 子集算出的真实统计量。
    """
    df = _synth_df({"ep-nan": 20, "ep-has-data": 50}, seed=4)
    nan_col = RED_COLS[0]
    df.loc[df["endpoint_key"] == "ep-nan", nan_col] = np.nan

    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    idx = RED_COLS.index(nan_col)

    # global 统计量应基于整列（含 ep-nan 全 NaN 行，pandas 默认 skipna）算出，
    # 等价于只用 ep-has-data 的行算出的均值/标准差
    global_mean = df[nan_col].mean()
    global_std = df[nan_col].std()
    assert not np.isnan(global_mean)
    assert not np.isnan(global_std)
    assert global_std > _FALLBACK_STD_EPSILON

    ep_mean, ep_std = stats.branch_stats("ep-nan", "ep")
    # use_shrinkage 默认 True，但退化索引不参与 shrinkage 混合（第 110 行
    # non_degenerate 过滤），所以 ep-nan 的取值应精确等于 global 统计量，
    # 而不是 shrunk 之后的某个中间值
    assert ep_mean[idx] == pytest.approx(global_mean, rel=1e-6)
    assert ep_std[idx] == pytest.approx(global_std, rel=1e-6)
    assert ep_std[idx] != pytest.approx(_FALLBACK_STD_EPSILON)
    assert nan_col not in stats.degenerate_columns("ep")


def test_shrinkage_pulls_sparse_endpoint_toward_global():
    """稀疏 endpoint（n 远小于 shrinkage_k）的 shrunk std 应比纯局部估计更接近
    global std——防止小样本方差估计噪声主导偏离量计算。

    注意：对比对象是同一 endpoint 的 shrunk vs 纯局部估计，而非 shrunk vs 另一个
    endpoint。原始 fixture（ep-dense n=500 + ep-sparse n=3）曾用 dense 的局部 std
    做对比基准，但 dense 占了 fit 集合 99.4% 的行数，global std 本质上就是
    dense 的局部 std（两者几乎恒等），导致该基准与 shrinkage 强度无关，
    500 次随机种子验证 100% 失败——这是 fixture 构造缺陷，不是 shrinkage
    公式（`std_shrunk = (n/(n+k))*std_local + (k/(n+k))*std_global`）本身的问题。
    """
    df = _synth_df({"ep-dense": 500, "ep-sparse": 3}, seed=2)
    stats_on = EndpointBaselineStats(
        red_cols=RED_COLS, svc_cols=SVC_COLS, shrinkage_k=10, use_shrinkage=True
    )
    stats_off = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS, use_shrinkage=False)
    stats_on.fit(df)
    stats_off.fit(df)
    _, sparse_shrunk_std = stats_on.branch_stats("ep-sparse", "ep")
    _, sparse_local_std = stats_off.branch_stats("ep-sparse", "ep")
    global_std = df[RED_COLS[0]].std()
    idx = 0
    assert abs(sparse_shrunk_std[idx] - global_std) < abs(sparse_local_std[idx] - global_std)


def test_shrinkage_disabled_uses_pure_local_stats():
    """需要 fit 集合里至少两个 endpoint，否则 global 统计量退化为该唯一 endpoint
    自身的局部统计量，shrinkage 在数学上是恒等变换（无论 k 取何值），
    无法体现"关闭 shrinkage 改变取值"的对比意图——ep-anchor 提供一个规模
    悬殊的第二 endpoint，让 global 统计量实际偏离 ep-sparse 的局部值。"""
    df = _synth_df({"ep-sparse": 3, "ep-anchor": 200}, seed=3)
    stats_on = EndpointBaselineStats(
        red_cols=RED_COLS, svc_cols=SVC_COLS, shrinkage_k=10, use_shrinkage=True
    )
    stats_off = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS, use_shrinkage=False)
    stats_on.fit(df)
    stats_off.fit(df)
    _, std_on = stats_on.branch_stats("ep-sparse", "ep")
    _, std_off = stats_off.branch_stats("ep-sparse", "ep")
    local_std = df[df["endpoint_key"] == "ep-sparse"][RED_COLS[0]].std()
    assert std_off[0] == pytest.approx(local_std, rel=1e-6)
    assert std_on[0] != pytest.approx(local_std, rel=1e-3)  # shrinkage 应改变取值


def test_unknown_endpoint_raises_key_error():
    """fit 时未见过的 endpoint_id 查表必须显式报错，不能静默返回 global 统计量——
    调用方（ReliabilityGatedFusion）需要知道这是数据契约错误还是正常的未知 endpoint
    退化路径，二者语义不同，不该在这一层混淆。"""
    df = _synth_df({"ep1": 50})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    with pytest.raises(KeyError):
        stats.branch_stats("ep-never-seen", "ep")


def test_save_load_roundtrip(tmp_path):
    df = _synth_df({"ep1": 50, "ep2": 30})
    stats = EndpointBaselineStats(red_cols=RED_COLS, svc_cols=SVC_COLS)
    stats.fit(df)
    path = tmp_path / "endpoint_baseline_stats.json"
    stats.save(path)
    loaded = EndpointBaselineStats.load(path)
    for ep in ("ep1", "ep2"):
        for branch in ("ep", "svc"):
            m1, s1 = stats.branch_stats(ep, branch)
            m2, s2 = loaded.branch_stats(ep, branch)
            np.testing.assert_allclose(m1, m2)
            np.testing.assert_allclose(s1, s2)
