import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.normalization import Normalizer


@pytest.fixture
def normal_df():
    return pd.DataFrame(
        {
            "endpoint_key": ["ep1"] * 4 + ["ep2"] * 4,
            "endpoint_red__trace_request_count": [0, 10, 20, 30, 100, 200, 300, 400],
        }
    )


def test_per_endpoint_minmax_fit_transform(normal_df):
    norm = Normalizer(
        rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")},
    )
    norm.fit(normal_df)
    out = norm.transform(normal_df)
    ep1_vals = out.loc[normal_df["endpoint_key"] == "ep1", "endpoint_red__trace_request_count"]
    assert ep1_vals.min() == pytest.approx(0.0)
    assert ep1_vals.max() == pytest.approx(1.0)


def test_anomaly_case_uses_normal_stats(normal_df):
    """故障 case 用 Normal 拟合的参数，超过 1 的值允许出现（不 clip）。"""
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    anomaly_df = pd.DataFrame(
        {
            "endpoint_key": ["ep1"],
            "endpoint_red__trace_request_count": [60],  # 超出 Normal ep1 的 max=30
        }
    )
    out = norm.transform(anomaly_df)
    # (60 - 0) / (30 - 0) = 2.0
    assert out["endpoint_red__trace_request_count"].iloc[0] == pytest.approx(2.0)


def test_stats_roundtrip_json(tmp_path, normal_df):
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    norm.save(tmp_path / "stats.json")
    norm2 = Normalizer.load(tmp_path / "stats.json")
    pd.testing.assert_frame_equal(norm.transform(normal_df), norm2.transform(normal_df))


def test_group_all_nan_in_fit_keeps_other_cases_untouched():
    """某 group（如某 service 的 metric 列）在 fit 集合（Normal）里全 NaN 时，
    不应把这组 nan 统计量通过减法/除法扩散到其他 case 里这个 group 本来完好的数值。
    对应 Normal 采集缺口（如 cAdvisor 掉线）曾把其他 case 的真实 metric 数值污染成 NaN 的 bug。
    """
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a"] * 3,
            "service_metric__cpu_usage_rate": [float("nan")] * 3,  # Normal 采集缺失
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    other_case_df = pd.DataFrame(
        {
            "service_name": ["svc-a", "svc-a"],
            "service_metric__cpu_usage_rate": [0.1, 0.6],  # 真实数值，来自数据完好的 case
        }
    )
    out = norm.transform(other_case_df)
    assert out["service_metric__cpu_usage_rate"].tolist() == [0.1, 0.6]


def test_group_all_nan_in_fit_does_not_affect_other_groups():
    """全 NaN 的 group 只跳过自身，同一列里其他数据完好的 group 仍正常归一化。"""
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a", "svc-a", "svc-b", "svc-b"],
            "service_metric__cpu_usage_rate": [float("nan"), float("nan"), 0.0, 10.0],
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    out = norm.transform(fit_df)
    svc_b_vals = out.loc[out["service_name"] == "svc-b", "service_metric__cpu_usage_rate"]
    assert svc_b_vals.tolist() == pytest.approx([0.0, 1.0])
    svc_a_vals = out.loc[out["service_name"] == "svc-a", "service_metric__cpu_usage_rate"]
    assert svc_a_vals.isna().all()


def test_global_scope_all_nan_in_fit_keeps_original_values():
    fit_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [float("nan"), float("nan")]})
    norm = Normalizer(rules={"endpoint_red__trace_5xx_rate": ("global", "min_max")})
    norm.fit(fit_df)

    other_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [0.2, 0.9]})
    out = norm.transform(other_df)
    assert out["endpoint_red__trace_5xx_rate"].tolist() == [0.2, 0.9]


def test_group_zero_variance_in_fit_keeps_original_values():
    """fit 集合该 group 只有单一取值（hi-lo=0）时，min-max 无法提供有效 scale，
    必须像全 NaN 一样跳过归一化保留原值。回归测试：修复前的实现用 max(hi-lo, 1e-9)
    托底除数，零方差退化组会把 eval 里任何非零差值放大 1e9 倍，产出 1e9~1e13 量级的
    离谱数值（曾在 artifacts/contract_v1/eval_all.parquet 的
    endpoint_red__client_latency_p95 / latency_divergence 两列上实际发生）。
    """
    fit_df = pd.DataFrame(
        {
            "endpoint_key": ["ep-degenerate"] * 3,
            "endpoint_red__client_latency_p95": [22679.9015, 22679.9015, 22679.9015],
        }
    )
    norm = Normalizer(rules={"endpoint_red__client_latency_p95": ("per_endpoint", "min_max")})
    norm.fit(fit_df)

    eval_df = pd.DataFrame(
        {
            "endpoint_key": ["ep-degenerate"],
            "endpoint_red__client_latency_p95": [5417.588],  # 真实 eval 值，偏离 fit 常量
        }
    )
    out = norm.transform(eval_df)
    assert out["endpoint_red__client_latency_p95"].iloc[0] == pytest.approx(5417.588)
    assert norm.skipped_groups()["endpoint_red__client_latency_p95"] == ["ep-degenerate"]


def test_global_scope_zero_variance_in_fit_keeps_original_values():
    fit_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [0.0, 0.0, 0.0]})
    norm = Normalizer(rules={"endpoint_red__trace_5xx_rate": ("global", "min_max")})
    norm.fit(fit_df)

    other_df = pd.DataFrame({"endpoint_red__trace_5xx_rate": [0.2, 0.9]})
    out = norm.transform(other_df)
    assert out["endpoint_red__trace_5xx_rate"].tolist() == [0.2, 0.9]


def test_group_zero_variance_does_not_affect_other_groups():
    """同一列里零方差 group 跳过归一化保留原值，同时数据完好的 group 仍正常归一化——
    这是真实生产形态（部分低频 endpoint 在 fit 窗口内零方差退化，其余 endpoint 正常）。
    对应 NaN 版 test_group_all_nan_in_fit_does_not_affect_other_groups 的零方差等价，
    覆盖 transform() 里"跳过分支"与"除法分支"在同一次调用里共存的交互路径。
    """
    fit_df = pd.DataFrame(
        {
            "endpoint_key": ["ep-degenerate"] * 2 + ["ep-normal"] * 2,
            "endpoint_red__client_latency_p95": [500.0, 500.0, 100.0, 300.0],
        }
    )
    norm = Normalizer(rules={"endpoint_red__client_latency_p95": ("per_endpoint", "min_max")})
    norm.fit(fit_df)

    eval_df = pd.DataFrame(
        {
            "endpoint_key": ["ep-degenerate", "ep-normal", "ep-normal"],
            "endpoint_red__client_latency_p95": [880.0, 100.0, 300.0],
        }
    )
    out = norm.transform(eval_df)
    vals = out["endpoint_red__client_latency_p95"]
    # 零方差 group 保留原始量纲，不被 1e9 放大
    assert vals.iloc[0] == pytest.approx(880.0)
    # 正常 group 仍按 [100, 300] 的 min-max 归一化
    assert vals.iloc[1] == pytest.approx(0.0)  # (100-100)/(300-100)
    assert vals.iloc[2] == pytest.approx(1.0)  # (300-100)/(300-100)
    assert norm.skipped_groups()["endpoint_red__client_latency_p95"] == ["ep-degenerate"]


def test_group_narrow_but_nonzero_variance_still_normalizes():
    """gap 极窄但非零（远大于 1e-9 阈值）的 group 不算退化，仍正常归一化——钉住阈值
    边界，防止未来把"窄方差"误并入"零方差"跳过路径。"""
    fit_df = pd.DataFrame(
        {
            "endpoint_key": ["ep"] * 2,
            "endpoint_red__client_latency_p95": [100.0, 100.001],  # gap=1e-3，非退化
        }
    )
    norm = Normalizer(rules={"endpoint_red__client_latency_p95": ("per_endpoint", "min_max")})
    norm.fit(fit_df)

    assert "endpoint_red__client_latency_p95" not in norm.skipped_groups()
    out = norm.transform(fit_df)
    vals = out["endpoint_red__client_latency_p95"]
    assert vals.iloc[0] == pytest.approx(0.0)
    assert vals.iloc[1] == pytest.approx(1.0)


def test_group_partially_nan_in_fit_uses_non_nan_subset():
    """group 只是部分 NaN（非全 NaN）时，fit 用 pandas 默认 skipna 语义从非 NaN 子集
    算统计量，正常参与归一化——不同于全 NaN 时的跳过路径。这里显式钉住该行为，
    避免未来重构在"部分 NaN"和"全 NaN"两条路径之间引入混淆。
    """
    fit_df = pd.DataFrame(
        {
            "service_name": ["svc-a"] * 4,
            "service_metric__cpu_usage_rate": [float("nan"), 0.1, float("nan"), 0.3],
        }
    )
    norm = Normalizer(rules={"service_metric__cpu_usage_rate": ("per_service", "min_max")})
    norm.fit(fit_df)

    lo, hi = norm._stats["service_metric__cpu_usage_rate"].by_group["svc-a"]
    assert (lo, hi) == pytest.approx((0.1, 0.3))

    out = norm.transform(fit_df)
    vals = out["service_metric__cpu_usage_rate"]
    assert vals.iloc[1] == pytest.approx(0.0)  # (0.1-0.1)/(0.3-0.1)
    assert vals.iloc[3] == pytest.approx(1.0)  # (0.3-0.1)/(0.3-0.1)
    assert vals.iloc[0] != vals.iloc[0]  # 原本就是 NaN 的行仍是 NaN（NaN != NaN）
    assert vals.iloc[2] != vals.iloc[2]


# ---------------------------------------------------------------------------
# per-case z-score + 三级 shrinkage（contract v2，Task 11）
# ---------------------------------------------------------------------------

_ZCOL = "endpoint_red__latency"


def _zscore_df(rows):
    return pd.DataFrame(
        rows,
        columns=["case_id", "endpoint_key", "service_name", _ZCOL],
    )


def test_per_case_groups_are_independent_across_cases():
    """同一 endpoint 在不同 case 的 fit 分布不同时，各自用各自的 mean/std——
    per-case 参照系的核心：caseA 的 100 在它自己的分布里是中心点（z=0），
    caseB 的 100 在高电平分分布里是负偏离。"""
    # 每组 40 行（w=40/50=0.8，接近真实 median 16 行里的大组）：组内统计量主导，
    # case 间的水平差不被 shrinkage 抹平
    fit_df = _zscore_df(
        [("cA", "ep1", "svc1", v) for v in [90.0, 100.0, 110.0, 120.0] * 10]
        + [("cB", "ep1", "svc1", v) for v in [1090.0, 1100.0, 1110.0, 1120.0] * 10]
    )
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)
    eval_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 100.0),
            ("cB", "ep1", "svc1", 100.0),
        ]
    )
    out = norm.transform(eval_df)[_ZCOL].tolist()
    # 同样的原始值 100：在 cA（mean=105）附近 → z≈0；在 cB（mean=1105）是强负偏离。
    # 两个 case 的统计量互不影响，这是 per-case 参照系的核心。
    assert abs(out[0]) < 1.0
    assert out[1] < -2.0
    assert out[1] < out[0] - 1.0


def test_per_case_zscore_math_exact():
    """z-score 数学正确性：只有一个 (case, endpoint) 组时，组/汇总/全局三层是
    同一批行、统计量逐位相同，shrinkage 退化为恒等，z 必须等于手算的
    (x-mean)/std（ddof=1）。多组下的收缩混合由 test_shrinkage_weight_* 单独钉。"""
    seq = [175.0, 200.0, 225.0, 200.0] * 10  # 40 行，单一 (cA, ep1) 组
    fit_df = _zscore_df([("cA", "ep1", "svc1", v) for v in seq])
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)
    eval_df = _zscore_df([("cA", "ep1", "svc1", 250.0)])
    out = norm.transform(eval_df)[_ZCOL].iloc[0]
    expected = (250.0 - 200.0) / pd.Series(seq).std()
    assert out == pytest.approx(expected)


def test_shrinkage_weight_n_less_than_k_manual():
    """n=2 < k=10 且 fallback 统计量已知：构造 2 个 case——cA 只有 2 行（稀疏组），
    cB 有 40 行锚定跨 case 汇总层；全局层故意放到第三极。手算 mean_eff/std_eff。"""
    # cA: 10, 20 → mean=15, std(ddof=1)=sqrt(50)=7.0711
    # cB: 0 重复 40 次 → mean=0, std=0（零方差，走全局层）
    # 跨 case 汇总（ep1 全部 42 行）：mean = (30+0)/42 = 0.7142857
    # 汇总 std(ddof=1) 由 pandas 算
    rows = [("cA", "ep1", "svc1", 10.0), ("cA", "ep1", "svc1", 20.0)]
    rows += [("cB", "ep1", "svc1", 0.0) for _ in range(40)]
    fit_df = _zscore_df(rows)
    # 全局层（所有行，同列；本测试只有 ep1 一个 endpoint，所以全局 == 跨 ep 汇总）
    pooled = pd.Series([10.0, 20.0] + [0.0] * 40)
    ep_std = pooled.std()  # 跨 case 汇总 std
    ep_mean = pooled.mean()  # 0.7142857

    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)

    w = 2 / 12  # n/(n+k)
    mean_eff = w * 15.0 + (1 - w) * ep_mean
    std_eff = w * np.sqrt(50.0) + (1 - w) * ep_std
    eval_df = _zscore_df([("cA", "ep1", "svc1", 18.0)])
    out = norm.transform(eval_df)[_ZCOL].iloc[0]
    assert out == pytest.approx((18.0 - mean_eff) / std_eff)


def test_zero_variance_falls_back_to_endpoint_then_global_then_sentinel():
    """三级回退链：cA 组零方差；endpoint 汇总层恰好也零方差（cB 同常数）；
    全局层放入第三个 endpoint 的真实方差。最终必须用全局 std，|z| 有限。
    再补一个"全局也零方差"的列验证链尾 1.0 哨兵——输出恰为 x - mean，|z|<1e3。"""
    rows = [
        ("cA", "ep1", "svc1", 7.0),
        ("cA", "ep1", "svc1", 7.0),  # 组内零方差
        ("cB", "ep1", "svc1", 7.0),
        ("cB", "ep1", "svc1", 7.0),  # endpoint 汇总层也零方差
        ("cA", "ep2", "svc2", 0.0),
        ("cA", "ep2", "svc2", 10.0),  # 全局层唯一方差来源
        ("cB", "ep2", "svc2", 0.0),
        ("cB", "ep2", "svc2", 10.0),
    ]
    fit_df = _zscore_df(rows)
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)
    eval_df = _zscore_df([("cA", "ep1", "svc1", 207.0)])
    out = norm.transform(eval_df)[_ZCOL].iloc[0]
    # mean 回退到全局 mean=7（ep1 常数 7 与 ep2 的 0/10 加权），std 回退到全局 std；
    # 无论如何不得出现 1e9 量级
    assert abs(out) < 1e3
    assert np.isfinite(out)
    # 207 显著偏离正常分布，z 必须是大的正向偏离但不离谱
    assert out > 5.0

    # 链尾哨兵：整列全局零方差（全 7.0）→ std=1.0 哨兵，mean 按 n 收缩后≈7
    const_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 7.0),
            ("cB", "ep1", "svc1", 7.0),
        ]
    )
    norm2 = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm2.fit(const_df)
    eval_const = _zscore_df([("cA", "ep1", "svc1", 1007.0)])
    out2 = norm2.transform(eval_const)[_ZCOL].iloc[0]
    # std 哨兵 1.0 → z = 1000（不是 1e12）；断言不出现 1e9 爆值
    assert out2 == pytest.approx(1000.0)
    assert abs(out2) < 1e3 + 100  # 哨兵语义下恰好 1000，留余量防 mean 微偏


def test_unseen_case_falls_back_to_endpoint_stats_without_raw_passthrough():
    """transform 时出现 fit 未见过的 (case, endpoint)：回退到该 endpoint 跨 case
    汇总统计量，不抛异常，也绝不透传原始量纲（per-case 下原始值跨 case 不可比）。"""
    fit_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 10.0),
            ("cA", "ep1", "svc1", 20.0),
            ("cA", "ep1", "svc1", 30.0),
            ("cA", "ep1", "svc1", 40.0),  # ep1 汇总 mean=25
            ("cA", "ep2", "svc2", 1000.0),
            ("cA", "ep2", "svc2", 2000.0),  # 让全局层远离 ep1
        ]
    )
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)
    unseen_df = _zscore_df([("cUNSEEN", "ep1", "svc1", 25.0)])
    out = norm.transform(unseen_df)[_ZCOL].iloc[0]
    # 必须落在 endpoint 级 z 尺度（约 0），而不是原始值 25
    assert out == pytest.approx(0.0, abs=1e-6)
    assert abs(out) < 5.0

    # 连 endpoint 都没见过 → 全局层，仍然归一化
    unseen_ep = _zscore_df([("cX", "ep999", "svc9", 500.0)])
    out_ep = norm.transform(unseen_ep)[_ZCOL].iloc[0]
    assert np.isfinite(out_ep)
    assert abs(out_ep) < 1e3
    assert out_ep != 500.0  # 不是透传


def test_per_case_zscore_roundtrip_json(tmp_path):
    """三级统计量（组 / endpoint 汇总 / 全局 + n）JSON 往返后 transform 逐值相同，
    且组合键以可读的 "case::endpoint" 字符串落盘。"""
    fit_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 10.0),
            ("cA", "ep1", "svc1", 20.0),
            ("cB", "ep1", "svc1", 30.0),
            ("cB", "ep1", "svc1", 40.0),
            ("cA", "ep2", "svc2", 100.0),
            ("cB", "ep2", "svc2", 200.0),
        ]
    )
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)
    norm.save(tmp_path / "z.json")

    raw = json.loads((tmp_path / "z.json").read_text())
    # 组合键可读、带分隔符
    assert "cA::ep1" in raw[_ZCOL]["by_group"]
    # 三级统计量都在
    assert "by_endpoint" in raw[_ZCOL]
    assert "global" in raw[_ZCOL]

    norm2 = Normalizer.load(tmp_path / "z.json")
    eval_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 17.0),
            ("cB", "ep2", "svc2", 250.0),
            ("cUNSEEN", "ep1", "svc1", 35.0),
        ]
    )
    pd.testing.assert_series_equal(
        norm.transform(eval_df)[_ZCOL],
        norm2.transform(eval_df)[_ZCOL],
        check_names=False,
    )


def test_per_case_service_scope_uses_case_service_composite_key(tmp_path):
    """per_case_service 与 endpoint 版结构对称：组键 (case_id, service_name)，
    汇总层按 service。inner join 数据集上两 scope 恰好同组数（entry 017），
    但键的第二维必须是 service_name。"""
    # 每组 20 行（w=20/30=2/3）：cA 围绕 20、cB 围绕 1020，组内 std 相同
    fit_df = _zscore_df(
        [("cA", "ep1", "svc1", v) for v in [10.0, 30.0] * 10]
        + [("cB", "ep1", "svc1", v) for v in [1010.0, 1030.0] * 10]
    )
    norm = Normalizer({_ZCOL: ("per_case_service", "z_score")})
    norm.fit(fit_df)
    norm.save(tmp_path / "s.json")
    raw = json.loads((tmp_path / "s.json").read_text())
    assert "cA::svc1" in raw[_ZCOL]["by_group"]
    assert "svc1" in raw[_ZCOL]["by_endpoint"]  # 汇总层键是 service

    eval_df = _zscore_df([("cA", "ep1", "svc1", 20.0), ("cB", "ep1", "svc1", 1020.0)])
    out = norm.transform(eval_df)[_ZCOL].tolist()
    # 两 case 关于 service 汇总均值对称 → z 等幅反号（shrinkage 把各自中心往
    # 汇总层拉，所以"在自身 case 中心"不再是 z=0，这是 shrinkage 的预期行为）
    assert out[0] == pytest.approx(-out[1])
    # 手算：w=2/3，汇总 mean=520，group std=10.26（重复序列 ddof=1），
    # 汇总 std 由 pandas 给
    pooled = pd.Series([10.0, 30.0] * 10 + [1010.0, 1030.0] * 10)
    w = 20 / 30
    mean_eff_a = w * 20.0 + (1 - w) * pooled.mean()
    std_eff = w * pd.Series([10.0, 30.0] * 10).std() + (1 - w) * pooled.std()
    assert out[0] == pytest.approx((20.0 - mean_eff_a) / std_eff)


def test_old_minmax_json_format_still_loads(tmp_path):
    """v1 时代落盘的 normalization_stats.json（by_group=[lo,hi]，无 by_endpoint/
    global/n 字段）必须仍能 load，且 transform 结果与旧逻辑一致——JSON 向后兼容。"""
    legacy = {
        "endpoint_red__trace_request_count": {
            "scope": "per_endpoint",
            "method": "min_max",
            "by_group": {"ep1": [0.0, 30.0]},
        }
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(legacy))
    norm = Normalizer.load(p)
    out = norm.transform(
        pd.DataFrame({"endpoint_key": ["ep1"], "endpoint_red__trace_request_count": [60.0]})
    )
    assert out["endpoint_red__trace_request_count"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# scope×method 合法配对守卫 + 小样本边界（Task 11 review 追加）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scope",
    ["per_endpoint", "per_service"],
)
def test_grouped_scope_rejects_z_score_at_construction(scope):
    """per_endpoint/per_service 只有单列分组键，z_score 的两级回退
    （组 → 跨 case 汇总 → 全局）无从定义：旧实现 fit 静默成功、transform 才在
    group_cols[1] 抛裸 IndexError。必须在构造期就带说明地拒绝。"""
    with pytest.raises(ValueError, match="z_score"):
        Normalizer({_ZCOL: (scope, "z_score")})


def test_legal_scope_method_combos_accepted():
    """合法配对白名单：三 scope × min_max + （global / 两 per-case scope）× z_score。"""
    df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 10.0),
            ("cA", "ep1", "svc1", 30.0),
            ("cB", "ep1", "svc1", 1010.0),
            ("cB", "ep1", "svc1", 1030.0),
        ]
    )
    # 不应抛异常即通过；global z_score 走单级（无上层）路径，确认其确实可用
    for rules in (
        {_ZCOL: ("global", "min_max")},
        {_ZCOL: ("per_endpoint", "min_max")},
        {_ZCOL: ("per_service", "min_max")},
        {_ZCOL: ("per_case_endpoint", "z_score")},
        {_ZCOL: ("per_case_service", "z_score")},
        {_ZCOL: ("global", "z_score")},
    ):
        Normalizer(rules).fit(df)

    g = Normalizer({_ZCOL: ("global", "z_score")})
    g.fit(df)
    out = g.transform(df)[_ZCOL]
    # 全局 z=(x-global_mean)/global_std，全局 mean=520
    assert np.isfinite(out).all()
    assert (out[df[_ZCOL] < 520] < 0).all()
    assert (out[df[_ZCOL] > 520] > 0).all()


def test_zscore_all_nan_group_n0_uses_upper_stats_and_passes_nan_through():
    """n=0（某 (case,endpoint) 组在 fit 段全 NaN）：组内无统计量，w=0 直接取
    endpoint 跨 case 汇总层；该组的 NaN 输入行 transform 后仍透传 NaN（不被
    填成 0 或放大）。另一正常组不受影响。"""
    fit_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", float("nan")),
            ("cA", "ep1", "svc1", float("nan")),  # cA/ep1 全 NaN → n=0
            ("cB", "ep1", "svc1", 10.0),
            ("cB", "ep1", "svc1", 30.0),  # ep1 汇总层唯一有效来源
        ]
    )
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)

    eval_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", float("nan")),
            ("cA", "ep1", "svc1", 20.0),  # 未知行值，但 cA/ep1 是已知退化组→走汇总层
            ("cB", "ep1", "svc1", 20.0),
        ]
    )
    out = norm.transform(eval_df)[_ZCOL].tolist()
    assert out[0] != out[0]  # NaN 透传
    # cA/ep1 与 cB/ep1 都落在 ep1 汇总层（mean=20, std=sqrt(200)=14.14，ddof=1）
    # cB 组 n=2 仍向汇总层收缩；cA 退化等价 w=0 完全用汇总层
    assert np.isfinite(out[1]) and np.isfinite(out[2])
    assert abs(out[1]) < 1e3
    # 20 是 ep1 汇总均值 → cA（纯汇总层）z 恰为 0
    assert out[1] == pytest.approx(0.0, abs=1e-9)


def test_zscore_single_row_group_n1_std_falls_back_output_finite():
    """n=1：单样本无法估 std（ddof=1 下 NaN），std 必须走上层回退而非除零/放大；
    输出有限、量级正常。组内 mean 取该行值，但按 w=1/11 向汇总层收缩。"""
    fit_df = _zscore_df(
        [
            ("cA", "ep1", "svc1", 20.0),  # cA/ep1 只有 1 行 → n=1, std=NaN
            ("cB", "ep1", "svc1", 10.0),
            ("cB", "ep1", "svc1", 30.0),  # 提供 ep1 汇总层方差
        ]
    )
    norm = Normalizer({_ZCOL: ("per_case_endpoint", "z_score")})
    norm.fit(fit_df)

    eval_df = _zscore_df([("cA", "ep1", "svc1", 120.0)])
    out = norm.transform(eval_df)[_ZCOL].iloc[0]
    assert np.isfinite(out)
    assert abs(out) < 1e3  # 不得出现 1e9 爆值（std 走了上层，不是 1e-9 托底）
    # 120 远高于 ep1 正常区间，应为正偏离
    assert out > 1.0
