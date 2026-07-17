import numpy as np
import pandas as pd
import pytest

from src.data.endpoint_baseline_stats import EndpointBaselineStats

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
