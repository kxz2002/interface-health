"""特征判别力分析（ep2_0729 数据集）· jupytext light 格式脚本。

本文件是 notebooks/feature_discriminability.ipynb 的可版本控制源。用
`jupytext --to notebook` 生成 .ipynb；改动请改本文件而不是 .ipynb，避免 diff 噪音。

分析目标：在 2026-07-28/29 单批次数据集上量化 18 维契约特征各自对"正常 vs 异常
endpoint"的判别力，找出高相关与低相关特征，并给出图形化结论。

三条必须遵守的协议约束（都来自历史踩坑，违反任意一条结论就不可信）：

1. **within-case 负样本**（history/entries/017）：负样本取同一个故障 case 内
   baseline/recover 阶段的行，不能拿 Normal case 的行。旧协议用跨 run Normal 当
   负样本，service_only 对 ABORT 的 AUROC 从 0.822 虚高到 0.922——虚高部分测的是
   "哪个采集 run"而不是故障。本数据集虽然 Normal 与故障 case 同批采集、混杂已大幅
   缓解，但 Normal 仍是独立 run，故仍按 within-case 为主协议。

2. **原始量纲**（history/entries/007、013）：判别力度量用 raw_features.parquet，
   不用 contract 的归一化表。Normalizer 是 per-endpoint/per-service min-max 且只在
   单个 Normal case 的 train_fit 上 fit；本数据集里 trace_error_rate /
   client_error_rate / *_5xx_rate 等列在 Normal 上恒为 0（零方差退化），被
   Normalizer 跳过而保留原量纲，其余列已归一化——同一张表内量纲混杂，跨特征
   可比性被破坏。AUROC 本身对单调变换不变，但分布图与效应量必须在原量纲上读。

3. **区分两种 NaN**：全 NaN 列意味着该模态"未构建"，不等于"无判别力"，必须从排名
   里剔除而非记为 0.5。本数据集 18 维全部构建完成（service_log 缺测仅 0.33%，远好于
   历史批次的大面积 NaN），故该保护当前不触发，但逻辑保留——它是复用本 notebook 分析
   其他数据集时的必要防线。其余列的 NaN 是真实缺测（如分位数样本量不足触发计算门限）。

标签口径：正样本用 is_endpoint_anomaly（= is_target_endpoint AND phase=='inject'），
只在 label_granularity=='endpoint' 的 16 个 Lv_E case 上有精确含义；11 个 case 级
故障（Lv_S/Lv_P/Lv_D）单独分层，其 is_endpoint_anomaly 退化为 case 级 is_anomaly。
"""

# %% [markdown]
# # 特征判别力分析 · ep2_0729
#
# 数据集：`~/Code/Repos/AnoMod/processed_data/2026_0729`（28 case，2026-07-28/29 单批次）
#
# | 家族 | case 数 | 标签精度 |
# |---|---|---|
# | `Lv_E_HTTP{ABORT,DELAY,PATCH,REPLACE}` × {assurance,order,travel,travel2} | 16 | endpoint 级精确 |
# | `Lv_S`（KILLPOD×3, DNSFAIL, HTTPABORT） | 5 | case 级 |
# | `Lv_P`（CPU, DISKIO, NETLOSS） | 3 | case 级 |
# | `Lv_D`（cachelimit, CONN_POOL, TRANSACTION） | 3 | case 级 |
# | `Normal` | 1 | 纯负样本 |

# %%
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CONTRACT = REPO / "artifacts/contract_ep2_0729_withlog"
FIGDIR = REPO / "outputs/figures/feature_discriminability"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        # 图里有中文标签，缺省 DejaVu Sans 会渲染成豆腐块。这两个字体在本机实测存在；
        # 换机器若缺失，matplotlib 会退回缺省字体并告警，不会崩，但中文会不可读。
        "font.sans-serif": ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)
sns.set_palette("colorblind")

# %% [markdown]
# ## 1 · 载入数据与特征分组

# %%
raw = pd.read_parquet(CONTRACT / "raw_features.parquet")
schema = json.loads((CONTRACT / "schema.json").read_text())
GROUPS = {name: g["columns"] for name, g in schema["feature_groups"].items()}
FEATURES = [c for cols in GROUPS.values() for c in cols]

# 未构建 vs 真实缺测：全 NaN 列本轮无法评估，必须从判别力表里剔除而不是记为 0.5
UNBUILT = [c for c in FEATURES if raw[c].isna().all()]
EVALUABLE = [c for c in FEATURES if c not in UNBUILT]

raw["family"] = raw["anomaly_type"].str.extract(r"^(Lv_[A-Z]|Normal)")
raw["http_kind"] = raw["anomaly_type"].str.extract(r"Lv_E_HTTP([A-Z]+)_")
raw["target_svc"] = raw["anomaly_type"].str.extract(r"Lv_E_HTTP[A-Z]+_(\w+)$")

print(f"行数 {len(raw)} · case {raw.case_id.nunique()} · endpoint {raw.endpoint_key.nunique()}")
print(f"可评估特征 {len(EVALUABLE)} / {len(FEATURES)}")
if UNBUILT:
    print(f"本轮未构建（全 NaN，不参与判别力排名）: {UNBUILT}")
print("\nphase × label_granularity:")
print(pd.crosstab(raw.phase, raw.label_granularity))

# %% [markdown]
# ## 2 · 数据质量前置体检
#
# 判别力排名前必须先看清哪些特征根本没有可用信号——常量列与高缺测列的"低判别力"
# 是数据采集问题，与"该特征在原理上无用"是两回事，结论里必须分开表述。

# %%
qc = pd.DataFrame(
    {
        "nan_pct": raw[FEATURES].isna().mean() * 100,
        "n_unique": raw[FEATURES].nunique(),
        "std": raw[FEATURES].std(),
        "min": raw[FEATURES].min(),
        "median": raw[FEATURES].median(),
        "max": raw[FEATURES].max(),
    }
).round(4)
qc["status"] = np.select(
    [
        qc.index.isin(UNBUILT),
        qc.n_unique <= 1,
        qc.nan_pct > 80,
        qc.nan_pct > 20,
    ],
    ["未构建", "常量(零信息)", "极高缺测", "高缺测"],
    default="可用",
)
qc["modality"] = [next(m for m, cols in GROUPS.items() if c in cols) for c in qc.index]
display_cols = ["modality", "status", "nan_pct", "n_unique", "std", "min", "median", "max"]
print(qc[display_cols].to_string())

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# 缺测率
order = qc.sort_values("nan_pct", ascending=False).index
colors = {
    "未构建": "#999999",
    "常量(零信息)": "#d62728",
    "极高缺测": "#ff7f0e",
    "高缺测": "#f0c000",
    "可用": "#2ca02c",
}
axes[0].barh(
    range(len(order)),
    qc.loc[order, "nan_pct"],
    color=[colors[qc.loc[c, "status"]] for c in order],
)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(
    [c.replace("endpoint_red__", "ep·").replace("service_", "svc·") for c in order], fontsize=8
)
axes[0].invert_yaxis()
axes[0].set_xlabel("缺测率 %")
axes[0].set_title("特征缺测率与可用性", fontweight="bold")
axes[0].legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=v) for v in colors.values()],
    labels=list(colors),
    fontsize=7,
    loc="lower right",
)

# 取值多样性（log 尺度，常量列一眼可见）
axes[1].barh(
    range(len(order)),
    qc.loc[order, "n_unique"].clip(lower=0.7),
    color=[colors[qc.loc[c, "status"]] for c in order],
)
axes[1].set_yticks(range(len(order)))
axes[1].set_yticklabels([])
axes[1].set_xscale("log")
axes[1].axvline(2, color="k", ls="--", lw=0.8)
axes[1].set_xlabel("distinct 取值数（log）")
axes[1].set_title("取值多样性：左侧贴线者为常量列", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGDIR / "01_data_quality.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3 · 判别力度量
#
# 每个特征在每个故障 case 内单独评分（within-case），再跨 case 聚合。三个互补指标：
#
# - **AUROC**：秩相关，对单调变换与量纲不变。`0.5` 为随机；报告时用 `|AUROC-0.5|`
#   衡量强度，方向单独记录（低于 0.5 意味着异常时该特征反而下降，同样是信号）。
# - **Cliff's delta**：非参数效应量，等价于 `2·AUROC-1`，读作"随机取一正一负样本，
#   正样本更大的概率优势"。
# - **Mann-Whitney p**：显著性，仅作辅助；正样本每 case 仅 40 行，p 值意义有限。
#
# NaN 处理：**按 case 逐列 dropna**，不做填补。用 train 均值填补会把"缺测模式"
# 本身当成信号（inject 与 baseline 阶段缺测密度天然不同），污染判别力度量——这是
# `analyze_modality_observability.py` 里已注明的同类陷阱。


# %%
def score_case(df: pd.DataFrame, feats: list[str], min_n: int = 8) -> dict:
    """单 case 内：正=is_endpoint_anomaly，负=同 case baseline/recover。

    delta 是 Cliff's delta，用 2*AUROC-1 换算而非 O(n·m) 两两比较——两者数学等价。
    """
    pos_mask = df.is_endpoint_anomaly.fillna(False).astype(bool)
    neg_mask = df.phase.isin(["baseline", "recover"]) & ~pos_mask
    out = {}
    for c in feats:
        p = df.loc[pos_mask, c].dropna().to_numpy()
        n = df.loc[neg_mask, c].dropna().to_numpy()
        if len(p) < min_n or len(n) < min_n or np.ptp(np.r_[p, n]) == 0:
            out[c] = dict(auroc=np.nan, delta=np.nan, pval=np.nan, n_pos=len(p), n_neg=len(n))
            continue
        a = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
        try:
            pv = stats.mannwhitneyu(p, n, alternative="two-sided").pvalue
        except ValueError:
            pv = np.nan
        out[c] = dict(auroc=a, delta=2 * a - 1, pval=pv, n_pos=len(p), n_neg=len(n))
    return out


records = []
for case_id, g in raw[raw.family != "Normal"].groupby("case_id"):
    res = score_case(g, EVALUABLE)
    meta = g.iloc[0]
    for feat, m in res.items():
        records.append(
            dict(
                case_id=case_id,
                anomaly_type=meta.anomaly_type,
                family=meta.family,
                http_kind=meta.http_kind,
                granularity=meta.label_granularity,
                feature=feat,
                modality=next(mo for mo, cols in GROUPS.items() if feat in cols),
                **m,
            )
        )
per_case = pd.DataFrame(records)
per_case["strength"] = (per_case.auroc - 0.5).abs()
print(f"逐 case × 特征评分记录：{len(per_case)}")
print(f"其中有效（样本量足够且非常量）：{per_case.auroc.notna().sum()}")

# %% [markdown]
# ### 3.1 · 排名指标的选择：为什么不能用跨 case 平均
#
# 只用 16 个 `Lv_E` case——它们有 `target_endpoint`，正样本是"目标 endpoint 在 inject
# 窗内"的精确标注。case 级故障标签下正样本是整个 case 的所有 endpoint，语义不同，
# 混在一起算会稀释结论。
#
# **两个必须避开的错误指标**（本分析实际先后踩中，结论完全相反，故写在这里钉住）：
#
# 1. `|mean(AUROC) − 0.5|`（先平均再取绝对值）：一个特征在 8 个 case 里升、8 个里降，
#    signed AUROC 相互抵消，均值回到 0.5，判别力被抹成 0。`client_error_rate` 在
#    ABORT/REPLACE 上 AUROC≈0.93、在 DELAY/PATCH 上≈0.43，抵消后 mean=0.681、
#    看起来平平无奇，实际它是这批数据里最干净的信号。
#
# 2. `mean(|AUROC − 0.5|)`（先取绝对值再平均）：修掉了抵消，但会把**不稳定**误判为
#    **强判别**。`service_log__event_rate` 在 4 个 ABORT case（同一种故障、只是打在
#    4 个不同 service）上的 AUROC 是 0.000 / 0.789 / 0.008 / 0.022——方向都在翻，
#    但每个都离 0.5 很远，于是 `mean|·|` 把它排到第 1 名。真实信号不可能在同种故障的
#    4 次注入之间翻转方向；这个量测到的是"日志来自哪个 service"，不是故障。
#
# **本分析采用的指标**：以「故障子类型（family）」为聚合单元，因为同一 family 的 4 个
# case 是同种故障打在 4 个不同 service 上、属于可重复实验：
#
# - `within_fam_std`：family 内 signed AUROC 的标准差（跨 4 个 service 平均）。**这是
#   可信度的主指标**——小 = 同种故障下表现可复现；大 = 结果由 service 身份而非故障决定。
# - `best_fam_effect`：该特征表现最强的那个 family 的 signed 效应量 `AUROC − 0.5`，
#   配 `best_fam_std` 一起读。特征的价值是"对某类故障强且稳"，不是"对所有故障都中等"。
# - `n_fam_stable`：`family 内 std < 0.05` 的 family 数（满分 4），衡量稳定覆盖广度。

# %%
lve = per_case[per_case.granularity == "endpoint"].copy()
lve["signed"] = lve.auroc - 0.5  # 保留方向：正=异常时升高，负=异常时降低

fam_stats = (
    lve.dropna(subset=["auroc"])
    .groupby(["modality", "feature", "http_kind"])
    .signed.agg(["mean", "std", "count"])
    .reset_index()
)

rows = []
for (mod, feat), g in fam_stats.groupby(["modality", "feature"]):
    best = g.loc[g["mean"].abs().idxmax()]
    rows.append(
        dict(
            modality=mod,
            feature=feat,
            within_fam_std=g["std"].mean(),
            best_fam=best.http_kind,
            best_fam_effect=best["mean"],
            best_fam_std=best["std"],
            best_fam_auroc=best["mean"] + 0.5,
            n_fam_stable=int((g["std"] < 0.05).sum()),
            n_fam_covered=len(g),
        )
    )
rank = pd.DataFrame(rows)
# 排序：先看是否稳定（n_fam_stable 降序），同档内比最强 family 的效应量
rank = rank.sort_values(
    ["n_fam_stable", "best_fam_effect"],
    key=lambda s: s.abs() if s.name == "best_fam_effect" else s,
    ascending=False,
)
rank["direction"] = np.where(rank.best_fam_effect >= 0, "异常↑", "异常↓")
rank["可信度"] = np.select(
    [rank.within_fam_std < 0.05, rank.within_fam_std < 0.15],
    ["高(family内可复现)", "中"],
    default="低(结果随service漂移)",
)
print(rank.round(4).to_string(index=False))

# 未通过样本量/常量门限、完全无法评估的特征单独列出，避免它们在图里以 0 值冒充"低判别力"
unscored = sorted(set(EVALUABLE) - set(rank.feature))
if unscored:
    print("\n无法评估（全常量或有效样本不足，非'低判别力'）:", unscored)

# %% [markdown]
# 左图 = 效应量（最强 family 的 signed AUROC−0.5，带该 family 内跨 service 误差棒），
# 右图 = 稳定性（family 内 std，越短越可信）。**两张图要一起读**：右图长的条，其左图
# 的效应量不可信。

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
r = rank.sort_values(
    ["n_fam_stable", "best_fam_effect"], key=lambda s: s.abs() if s.name == "best_fam_effect" else s
)
ycolors = {"endpoint_red": "#1f77b4", "service_metric": "#ff7f0e", "service_log": "#2ca02c"}
ylabels = [c.replace("endpoint_red__", "ep·").replace("service_", "svc·") for c in r.feature]

axes[0].barh(
    range(len(r)),
    r.best_fam_effect,
    xerr=r.best_fam_std.fillna(0),
    color=[ycolors[m] for m in r.modality],
    error_kw=dict(lw=0.8, capsize=2, alpha=0.6),
)
for i, (_, row) in enumerate(r.iterrows()):
    off = 0.012 if row.best_fam_effect >= 0 else -0.012
    axes[0].text(
        row.best_fam_effect + off * (1 + (row.best_fam_std or 0) * 3),
        i,
        f"{row.best_fam}·AUROC={row.best_fam_auroc:.3f}",
        va="center",
        ha="left" if row.best_fam_effect >= 0 else "right",
        fontsize=7,
    )
axes[0].axvline(0, color="k", lw=1)
axes[0].set_yticks(range(len(r)))
axes[0].set_yticklabels(ylabels, fontsize=8)
axes[0].set_xlabel("最强故障子类型下的效应量  (AUROC − 0.5)")
axes[0].set_title("① 效应量：正=异常时升高，负=异常时降低", fontweight="bold")
axes[0].set_xlim(-0.62, 0.75)
axes[0].legend(
    handles=[plt.Rectangle((0, 0), 1, 1, color=ycolors[m]) for m in ycolors],
    labels=list(ycolors),
    fontsize=8,
    loc="lower left",
)

stab_color = [
    "#2ca02c" if s < 0.05 else "#f0c000" if s < 0.15 else "#d62728" for s in r.within_fam_std
]
axes[1].barh(range(len(r)), r.within_fam_std, color=stab_color)
for i, (_, row) in enumerate(r.iterrows()):
    axes[1].text(
        row.within_fam_std + 0.006, i, f"稳定 {row.n_fam_stable}/4", va="center", fontsize=7
    )
axes[1].axvline(0.05, color="k", ls="--", lw=0.9)
axes[1].set_xlabel("故障子类型内标准差（跨 4 个 service，越小越可复现）")
axes[1].set_title("② 稳定性：红色=结果由 service 身份决定，不可信", fontweight="bold")
axes[1].set_xlim(left=0)

plt.suptitle("Lv_E（endpoint 级精确标签，16 case）特征判别力", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGDIR / "02_discriminability_ranking.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.2 · 按故障子类型分层：同一特征对不同故障的判别力差异极大
#
# entry 017 的核心教训之一：16 个 `Lv_E` case 绝不能混为一谈。ABORT/REPLACE 是
# 错误类信号、DELAY 是延迟类、PATCH 是语义损坏（响应可能仍是 HTTP 200）。

# %%
pivot = lve.pivot_table(index="feature", columns="http_kind", values="auroc", aggfunc="mean")
pivot = pivot.reindex(rank.feature.tolist())

fig, ax = plt.subplots(figsize=(8.2, 6))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".3f",
    cmap="RdBu_r",
    center=0.5,
    vmin=0,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "AUROC（0.5=随机，>0.5 异常时升高，<0.5 异常时降低）"},
    ax=ax,
)
ax.set_yticklabels(
    [c.replace("endpoint_red__", "ep·").replace("service_", "svc·") for c in pivot.index],
    rotation=0,
    fontsize=8,
)
ax.set_xlabel("HTTP 故障子类型")
ax.set_ylabel("")
ax.set_title("特征 × 故障子类型 判别力矩阵", fontweight="bold")
plt.tight_layout()
plt.savefig(FIGDIR / "03_feature_by_faulttype.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.3 · 模态层面汇总
#
# 左图把每个模态的特征放到「效应量 × 稳定性」平面上——**只有左上区（效应量大、std 小）
# 才是真信号**。右图按 signed 效应量看模态对各故障子类型的响应，保留方向。

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

for mod, g in rank.groupby("modality"):
    axes[0].scatter(
        g.within_fam_std,
        g.best_fam_effect.abs(),
        s=70,
        alpha=0.8,
        label=mod,
        color=ycolors[mod],
        edgecolor="k",
        linewidth=0.5,
    )
for _, row in rank.iterrows():
    axes[0].annotate(
        row.feature.replace("endpoint_red__", "ep·").replace("service_", "svc·"),
        (row.within_fam_std, abs(row.best_fam_effect)),
        fontsize=6.5,
        xytext=(4, 3),
        textcoords="offset points",
    )
axes[0].axvline(0.05, color="k", ls="--", lw=0.9)
axes[0].set_xlabel("故障子类型内标准差（→ 越右越不可复现）")
axes[0].set_ylabel("|效应量|（↑ 越高越强）")
axes[0].set_title("效应量 × 稳定性：仅左上区可信", fontweight="bold")
axes[0].legend(fontsize=8)

fam_mod = (
    fam_stats.assign(abs_eff=lambda d: d["mean"].abs())
    .sort_values("abs_eff")
    .groupby(["modality", "http_kind"])
    .last()["mean"]
    .unstack()
)
fam_mod.T.plot(kind="bar", ax=axes[1], color=[ycolors[m] for m in fam_mod.index], width=0.78)
axes[1].axhline(0, color="k", lw=1)
axes[1].set_ylabel("该模态最佳特征的 signed 效应量")
axes[1].set_xlabel("")
axes[1].set_title("模态 × 故障子类型（保留方向）", fontweight="bold")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(fontsize=8, title=None)

plt.tight_layout()
plt.savefig(FIGDIR / "04_modality_summary.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4 · Top 特征的分布形态
#
# AUROC 只给秩序信息，看不出"信号长什么样"。对判别力最强的几个特征画原始量纲分布，
# 确认信号是否可用单阈值分开（这直接决定它对 One-Class 检测器是否友好）。

# %%
top_feats = rank.head(6).feature.tolist()
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, feat in zip(axes.ravel(), top_feats):
    d = raw[raw.label_granularity == "endpoint"].copy()
    d["组"] = np.where(
        d.is_endpoint_anomaly.fillna(False).astype(bool),
        "异常(inject·目标)",
        "正常(baseline/recover)",
    )
    vals = d[feat].dropna()
    # 重尾特征用 symlog，否则少数极端值会把主体压成一条线
    heavy = vals.abs().max() > 20 * (vals.abs().quantile(0.99) + 1e-9)
    sns.violinplot(data=d, x="组", y=feat, ax=ax, cut=0, inner="quartile", density_norm="width")
    if heavy or vals.abs().max() > 1000:
        ax.set_yscale("symlog")
    a = rank.loc[rank.feature == feat, "best_fam_auroc"].iloc[0]
    ax.set_title(
        f"{feat.replace('endpoint_red__','ep·').replace('service_','svc·')}\nAUROC={a:.3f}",
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(FIGDIR / "05_top_feature_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5 · 特征间相关性与冗余
#
# 高判别力特征若彼此高度相关，实际提供的独立信息远少于维度数。这直接关系到
# "18 维等权距离把集中信号平均掉"（entry 017 的信号淹没）——冗余维度越多，
# 单个强信号维度在 L2 距离里的占比越被摊薄。

# %%
corr = raw.loc[raw.label_granularity == "endpoint", EVALUABLE].corr(method="spearman")
labels = [c.replace("endpoint_red__", "ep·").replace("service_", "svc·") for c in corr.columns]

fig, ax = plt.subplots(figsize=(8.8, 7.2))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    xticklabels=labels,
    yticklabels=labels,
    annot_kws={"size": 6.5},
    cbar_kws={"label": "Spearman ρ"},
    ax=ax,
)
ax.set_title("特征间 Spearman 相关（Lv_E 行）", fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=7.5)
plt.yticks(fontsize=7.5)
plt.tight_layout()
plt.savefig(FIGDIR / "06_feature_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6 · 信号淹没验证：单特征 oracle vs 全向量距离
#
# 复现 entry 017 的核心现象并量化：最佳单特征能达到的 AUROC，与把所有可用特征
# 等权拼成向量后用欧氏距离打分的 AUROC 相比差多少。这是"为什么需要非等权融合"
# 的直接证据。


# %%
def naive_distance_auroc(g: pd.DataFrame, feats: list[str], normal_ref: pd.DataFrame) -> float:
    """等权欧氏距离：以 Normal case 的均值/标准差做 z-score，再取 L2 范数当异常分。

    参照集用 Normal case 而非本 case baseline——模仿 One-Class 检测器"只见过正常
    数据"的设定。z-score 是必要的：不做标准化时量纲最大的列（latency 到 8e4）会
    单独决定距离，那测的是量纲而不是融合。
    """
    mu, sd = normal_ref[feats].mean(), normal_ref[feats].std().replace(0, np.nan)
    z = (g[feats] - mu) / sd
    score = np.sqrt((z**2).mean(axis=1, skipna=True))  # 均值而非和，容忍逐行 NaN 数不同
    pos = g.is_endpoint_anomaly.fillna(False).astype(bool)
    neg = g.phase.isin(["baseline", "recover"]) & ~pos
    y = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    s = np.r_[score[pos], score[neg]]
    ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) == 2 else np.nan


normal_ref = raw[raw.family == "Normal"]
ep_feats = [c for c in GROUPS["endpoint_red"] if c in EVALUABLE]

rows = []
for kind, g_kind in raw[raw.label_granularity == "endpoint"].groupby("http_kind"):
    orc, naive_ep, naive_all = [], [], []
    for _, g in g_kind.groupby("case_id"):
        sc = score_case(g, EVALUABLE)
        # 取 max(a, 1-a) 而不是 signed a：单特征检测器的打分方向是自由的（乘 -1 即可
        # 反向），AUROC=0.26 与 0.74 是同等强度的信号。若直接报 signed 值，会与
        # 下面固定方向的距离分数不可比——ABORT 的最佳单特征 signed AUROC=0.259 看着
        # 比 naive 的 0.625 还差，实际它更强（0.741）。这是本分析初版的一处误报。
        best = max(
            (max(v["auroc"], 1 - v["auroc"]) for v in sc.values() if pd.notna(v["auroc"])),
            default=np.nan,
        )
        orc.append(best)
        naive_ep.append(naive_distance_auroc(g, ep_feats, normal_ref))
        naive_all.append(naive_distance_auroc(g, EVALUABLE, normal_ref))
        # 距离分数方向是固定的（离正常中心越远=越异常），不做 max(a,1-a) 翻转——
        # 翻转它等于承认"离正常越近越异常"，那不是 One-Class 检测器的合法用法。
    best_feat = (
        lve[lve.http_kind == kind]
        .groupby("feature")
        .signed.mean()
        .abs()
        .idxmax()
        .replace("endpoint_red__", "ep·")
        .replace("service_", "svc·")
    )
    rows.append(
        dict(
            http_kind=kind,
            oracle_single=np.nanmean(orc),
            naive_endpoint_only=np.nanmean(naive_ep),
            naive_all=np.nanmean(naive_all),
            best_feature=best_feat,
        )
    )
submersion = pd.DataFrame(rows)
print(submersion.round(3).to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(submersion))
w = 0.27
ax.bar(x - w, submersion.oracle_single, w, label="最佳单特征 (oracle)", color="#2ca02c")
ax.bar(x, submersion.naive_endpoint_only, w, label="等权距离 · endpoint_red", color="#1f77b4")
ax.bar(x + w, submersion.naive_all, w, label="等权距离 · 全部可用维", color="#ff7f0e")
ax.axhline(0.5, color="k", ls="--", lw=1, label="随机")
for i, row in submersion.iterrows():
    ax.text(
        i - w, row.oracle_single + 0.015, row.best_feature, ha="center", fontsize=6.5, rotation=90
    )
ax.set_xticks(x)
ax.set_xticklabels(submersion.http_kind)
ax.set_ylabel("AUROC")
ax.set_ylim(0, 1.15)
ax.set_title("信号淹没：集中信号 vs 等权融合", fontweight="bold")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIGDIR / "07_signal_submersion.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7 · case 级故障（Lv_S / Lv_P / Lv_D）对照
#
# 这 11 个 case 无 `target_endpoint`，标签退化为 case 级。它们对 service 级模态的
# 判别力预期显著高于 `Lv_E`——KILLPOD/CPU/DISKIO 直接改变容器资源指标。列出来是为
# 验证"service_metric 低判别力"是故障类型特异的，不是该模态整体无用。

# %%
case_lvl = per_case[per_case.granularity == "case"]
if not case_lvl.empty:
    fam_pivot = case_lvl.pivot_table(
        index="feature", columns="family", values="auroc", aggfunc="mean"
    )
    fam_pivot = fam_pivot.reindex([f for f in rank.feature if f in fam_pivot.index])
    combined = pivot.join(fam_pivot)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    sns.heatmap(
        combined,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0.5,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "AUROC"},
        ax=ax,
    )
    ax.set_yticklabels(
        [c.replace("endpoint_red__", "ep·").replace("service_", "svc·") for c in combined.index],
        rotation=0,
        fontsize=8,
    )
    ax.axvline(len(pivot.columns), color="k", lw=2.5)
    ax.set_title(
        "左：Lv_E endpoint 级故障（精确标签） | 右：case 级故障（近似标签）",
        fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(FIGDIR / "08_endpoint_vs_case_level.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 8 · 结论汇总表

# %%
summary = rank[
    [
        "modality",
        "feature",
        "best_fam",
        "best_fam_auroc",
        "best_fam_effect",
        "best_fam_std",
        "within_fam_std",
        "n_fam_stable",
        "direction",
        "可信度",
    ]
].copy()

# 判定同时要求"效应量够大"与"family 内可复现"。只看效应量会把 service_log__event_rate
# 这类随 service 漂移的噪声判成高判别力（本分析初版实际踩中，见 §3.1）。
summary["判定"] = np.select(
    [
        (summary.within_fam_std < 0.05) & (summary.best_fam_effect.abs() >= 0.25),
        (summary.within_fam_std < 0.05) & (summary.best_fam_effect.abs() >= 0.10),
        summary.within_fam_std < 0.15,
    ],
    ["高判别力(强且稳)", "中判别力(稳但弱)", "存疑(中等稳定)"],
    default="不可用(随service漂移)",
)
summary = summary.merge(qc[["status", "nan_pct"]], left_on="feature", right_index=True)
summary = summary.sort_values(
    ["within_fam_std", "best_fam_effect"],
    key=lambda s: s.abs() if s.name == "best_fam_effect" else s,
    ascending=[True, False],
)
print(summary.round(4).to_string(index=False))

# 常量/无法评估的特征不进上表（它们不是"低判别力"，是"无信号可测"），单列说明
zero_info = qc[qc.status.isin(["常量(零信息)", "未构建"])].index.tolist()
print("\n零信息或未构建（不参与判别力排名）:")
for c in zero_info:
    print(
        f"  {c}: {qc.loc[c,'status']}, nan={qc.loc[c,'nan_pct']}%, n_unique={qc.loc[c,'n_unique']}"
    )

out_csv = FIGDIR / "discriminability_summary.csv"
summary.to_csv(out_csv, index=False)
per_case.to_csv(FIGDIR / "per_case_scores.csv", index=False)
fam_stats.to_csv(FIGDIR / "family_level_effects.csv", index=False)
submersion.to_csv(FIGDIR / "signal_submersion.csv", index=False)
print(f"\n汇总表 → {out_csv}")
print(f"图 → {FIGDIR}")
