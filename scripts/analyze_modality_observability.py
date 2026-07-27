"""C · 模态观测覆盖诊断（去跨run混杂版）。

为 "Reliability ≠ Observability" 融合模型提供设计依据（entry 017）。三个核心问题：
1. 信号淹没：单特征 oracle AUROC vs 朴素全向量 SVDD AUROC —— 集中信号是否被等权距离平均掉？
2. 故障可观测性分层：哪个模态/特征载哪类 HTTP 故障（ABORT/DELAY/PATCH/REPLACE）？
3. 跨run混杂：跨run Normal 负样本 vs 同case baseline/recover 负样本，service 分支虚高几何？

协议：正样本 = is_endpoint_anomaly==1；负样本默认 = 同 case 的 baseline/recover 行（同 run，
去跨run混杂）。另单列跨run协议做对照，暴露现有 eval_all 协议的混杂。
复用已归一化 contract parquet + DeepSVDD，不碰生产代码/configs/dvc。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.models.deep_svdd import DeepSVDD

SUBSETS = {
    "endpoint_only": ["endpoint_red"],
    "service_only": ["service_metric", "service_log"],
    "all": ["endpoint_red", "service_metric", "service_log"],
}
SEEDS = [1, 2, 3, 42]
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 64
REP_DIM = 32
HTTP_SUBTYPES = ["ABORT", "DELAY", "PATCH", "REPLACE"]


def _load(parquet: Path, groups: dict, sub_groups: list, tr_means: pd.Series):
    df = pd.read_parquet(parquet)
    cols = [c for g in sub_groups for c in groups[g]]
    x = df[cols].fillna(tr_means[cols]).fillna(0.0).values.astype(np.float32)
    return df, torch.tensor(x)


def _train(x_train: torch.Tensor, seed: int) -> DeepSVDD:
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DeepSVDD(input_dim=x_train.shape[1], hidden_dim=HIDDEN_DIM, rep_dim=REP_DIM)
    m.init_center(x_train)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    m.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        m.svdd_loss(x_train).backward()
        opt.step()
    m.eval()
    return m


def _auroc(y, s):
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def _pos_neg(df: pd.DataFrame, mask_http: str | None, cross_run: bool):
    """正 = 该(子)故障的 is_endpoint_anomaly；负 = 同case baseline/recover（cross_run=False）
    或 跨run Normal（cross_run=True）。返回布尔正样本索引 + 负样本索引。"""
    if mask_http:
        fault = df[df["_http"] == mask_http]
    else:
        fault = df[df["_fam"] == "Lv_E"]
    pos = fault[fault["is_endpoint_anomaly"] == 1]
    if cross_run:
        neg = df[df["anomaly_type"] == "Normal"]
    else:
        neg = fault[fault["phase"].isin(["baseline", "recover"])]
    return pos, neg


def _eval_subset(df: pd.DataFrame, scores: np.ndarray, cross_run: bool) -> dict:
    """按 HTTP 子类型算 AUROC（正=inject-target, 负按 cross_run 选池）。"""
    d = df.copy()
    d["_s"] = scores
    res = {}
    for h in HTTP_SUBTYPES:
        pos, neg = _pos_neg(d, h, cross_run)
        both = pd.concat([pos.assign(_y=1), neg.assign(_y=0)])
        res[h] = _auroc(both["_y"], both["_s"])
    # Lv_E 整体
    pos, neg = _pos_neg(d, None, cross_run)
    both = pd.concat([pos.assign(_y=1), neg.assign(_y=0)])
    res["Lv_E_all"] = _auroc(both["_y"], both["_s"])
    return res


def _oracle_single_feature(df: pd.DataFrame, groups: dict) -> dict:
    """单特征 oracle：每个 HTTP 子类型下，取 endpoint_red 各列单独的最佳 |AUROC-0.5| 特征。
    用同case负样本。证明'信号存在但被朴素全向量 SVDD 淹没'。"""
    ep_cols = groups["endpoint_red"]
    out = {}
    for h in HTTP_SUBTYPES:
        pos, neg = _pos_neg(df, h, cross_run=False)
        both = pd.concat([pos.assign(_y=1), neg.assign(_y=0)])
        best_c, best_a = None, 0.5
        for c in ep_cols:
            a = _auroc(both["_y"], both[c].fillna(0.0))
            if a is not None and abs(a - 0.5) > abs(best_a - 0.5):
                best_a, best_c = a, c
        out[h] = (best_c.replace("endpoint_red__", "") if best_c else "—", best_a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract-dir", default="artifacts/contract_v1")
    ap.add_argument("--out", default="artifacts/modality_observability/report.md")
    args = ap.parse_args()

    cdir = Path(args.contract_dir)
    schema = json.loads((cdir / "schema.json").read_text())
    groups = {n: g["columns"] for n, g in schema["feature_groups"].items()}
    allc = [c for cols in groups.values() for c in cols]
    tr_means = pd.read_parquet(cdir / "train.parquet")[allc].mean()

    def tag(df):
        df = df.copy()
        df["_fam"] = df["anomaly_type"].str.extract(r"(Lv_[A-Z])")
        df["_http"] = df["anomaly_type"].str.extract(r"Lv_E_HTTP([A-Z]+)_")
        return df

    # 训练三个子集 × seeds，收集 within-case 与 cross-run 两套 AUROC
    within, cross = {}, {}
    for sname, sgroups in SUBSETS.items():
        _, x_tr = _load(cdir / "train.parquet", groups, sgroups, tr_means)
        eval_df, x_ev = _load(cdir / "eval_all.parquet", groups, sgroups, tr_means)
        eval_df = tag(eval_df)
        w_seeds, c_seeds = [], []
        for s in SEEDS:
            model = _train(x_tr, s)
            with torch.no_grad():
                sc = model.score(x_ev).numpy()
            w_seeds.append(_eval_subset(eval_df, sc, cross_run=False))
            c_seeds.append(_eval_subset(eval_df, sc, cross_run=True))
        within[sname], cross[sname] = w_seeds, c_seeds
        print(f"[{sname}] dim={x_tr.shape[1]} done")

    oracle = _oracle_single_feature(tag(pd.read_parquet(cdir / "eval_all.parquet")), groups)
    _write_report(within, cross, oracle, Path(args.out))


def _agg(seed_list: list, key: str) -> str:
    vals = [m[key] for m in seed_list if m.get(key) is not None]
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "—"


def _write_report(within, cross, oracle, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    subs = list(within.keys())
    rows = HTTP_SUBTYPES + ["Lv_E_all"]
    L = [
        "# C · 模态观测覆盖诊断（去跨run混杂）",
        "",
        f"Deep SVDD × {len(SEEDS)} seeds{SEEDS}。正样本=is_endpoint_anomaly。AUROC=mean±std。",
        "",
    ]
    L += [
        "## 表1 · within-case 协议（负=同case baseline/recover，去混杂）",
        "",
        "| HTTP子类型 | " + " | ".join(subs) + " | 单特征oracle |",
        "|---|" + "|".join(["---"] * (len(subs) + 1)) + "|",
    ]
    for r in rows:
        oc = f"{oracle[r][1]:.3f} ({oracle[r][0]})" if r in oracle else "—"
        L.append(f"| {r} | " + " | ".join(_agg(within[s], r) for s in subs) + f" | {oc} |")
    L += [
        "",
        "## 表2 · cross-run 协议（负=跨run Normal，暴露混杂虚高）",
        "",
        "| HTTP子类型 | " + " | ".join(subs) + " |",
        "|---|" + "|".join(["---"] * len(subs)) + "|",
    ]
    for r in rows:
        L.append(f"| {r} | " + " | ".join(_agg(cross[s], r) for s in subs) + " |")
    L += [
        "",
        "## 读法",
        "- 单特征oracle >> 全向量SVDD ⇒ 信号存在但被朴素等权距离淹没（信号淹没）。",
        "- 表2 service_only 明显高于表1 ⇒ 该子类型的'检测'部分来自跨run身份，非故障（混杂）。",
        "- oracle≈0.5（PATCH）⇒ 该故障对所有 RED 特征隐形，需响应体校验。",
    ]
    out.write_text("\n".join(L) + "\n")
    print(f"报告 → {out}")


if __name__ == "__main__":
    main()
