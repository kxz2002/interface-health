"""Downstream anomaly-detection benchmark on the released per-endpoint dataset.

Pure standalone script — does NOT touch data/ files, DVC, or the contract pipeline.
Reads only tt_fused_15s.csv per case, filters to 26 published runs, and runs three
detectors (DeepSVDD, OneClassSVM, IsolationForest) on two feature arms
(F-inherited = 18 dims without the 3 added client fields, F-full = 21 dims)
under two training-pool regimes (Normal-only vs Normal+baseline).

Both pool regimes hold out a temporal tail of each Normal case (and, for
"expanded", each fault case's baseline phase) as genuine normal-state eval
negatives — see build_train_pool and TRAIN_POOL_FRACTION. An earlier version
put 100% of that data into train, leaving "expanded" eval with zero true-
normal rows (PR #26 review, Critical #1). Any artifacts produced before this
fix used the old all-into-train pool and must be re-run.

Reproduces the 23,244-window / 639-positive / 2.7% positive-rate figures
from the paper's Table `tab:scale`. Does NOT include the rel_pos / z-score
trivial baselines in the headline output (per user direction); they are
still computed and stored in run metadata for internal reference only.

Outputs: artifacts/downstream_benchmark/{feature_set}_{pool}_{model}_seed{seed}/
  - scores.parquet  (sample_id, score, y_true, case_id, phase, endpoint_key, is_target_endpoint)
  - metrics.json    (overall + per-case macro AUROC/AUPRC, plus internal baselines
                     and run provenance: git commit, data root, feature columns)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Repo-local imports (project is installed via `pip install -e .`)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

from src.contracts.split_fault_phase import split_fault_phase_temporal  # noqa: E402
from src.models.deep_svdd import DeepSVDD  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

LOG = logging.getLogger("benchmark_downstream")

# ---------------------------------------------------------------------------
# Paper-aligned 26 runs (drop TRANSACTION_timeout which has only 4 inject rows)
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDE_CASES = ["Lv_D_TRANSACTION_timeout_20260728T230311Z_em"]

# Fraction of each case's temporally-earliest windows absorbed into the one-class
# train pool; the remaining tail stays in eval as genuine normal-state negatives.
# Applies to every Normal case (both pool modes) and, under "expanded", to each
# fault case's baseline phase. See build_train_pool for why eval must retain
# some real normal rows (PR #26 review, Critical #1).
TRAIN_POOL_FRACTION = 0.8

# 21 numeric features present in tt_fused_15s.csv (col 8-16 trace + 29-40 client)
FEATURE_COLS_FULL = [
    "trace_request_count",
    "trace_latency_mean",
    "trace_latency_p50",
    "trace_latency_p95",
    "trace_latency_p99",
    "trace_error_rate",
    "trace_5xx_rate",
    "trace_4xx_rate",
    "trace_status_coverage",
    "client_request_count",
    "client_error_rate",
    "client_2xx_rate",
    "client_4xx_rate",
    "client_content_length_mean",
    "client_content_length_rel_shift",
    "client_body_hash_mismatch_rate",
    "client_latency_mean",
    "client_latency_p50",
    "client_latency_p95",
    "client_latency_p99",
    "client_5xx_rate",
]

# The 3 added client-side fields (paper §Client-side response observability)
ADDED_CLIENT_COLS = {
    "client_content_length_mean",
    "client_content_length_rel_shift",
    "client_body_hash_mismatch_rate",
}

FEATURE_COLS_INHERITED = [c for c in FEATURE_COLS_FULL if c not in ADDED_CLIENT_COLS]

# Columns where "no events this window" genuinely implies rate/count == 0 — 0-fill
# is semantically correct here. Everything else in FEATURE_COLS_FULL is a latency
# statistic or a content-integrity measure, where a NaN means "we don't know",
# not "the value is 0"; 0-filling those manufactures a fake best-case observation
# (PR #26 review, Critical #3 — the old code 0-filled all 21 columns uniformly and
# mislabeled the comment as a "rate-column convention").
RATE_ZEROFILL_COLS = {
    "trace_request_count",
    "trace_error_rate",
    "trace_5xx_rate",
    "trace_4xx_rate",
    "trace_status_coverage",
    "client_request_count",
    "client_error_rate",
    "client_2xx_rate",
    "client_4xx_rate",
    "client_5xx_rate",
}

# The 2 added fields whose NaN rate differs sharply between target and non-target
# endpoints during inject (measured on new_merge: 49% vs 15% NaN) — part of the
# F-inherited -> F-full AUROC gain may be the model keying on "this field went
# missing on the target endpoint" rather than on real content-integrity signal.
# client_content_length_mean is excluded: its NaN rate is 0% on both sides, so an
# indicator for it carries no information (measured on new_merge, see PR #26 review
# Critical #2 discussion).
MISSING_INDICATOR_COLS = {
    "client_content_length_rel_shift",
    "client_body_hash_mismatch_rate",
}

# Ablation feature sets isolating the missingness confound above (Critical #2):
# missing_indicator_only measures the missingness signal in isolation; full's own
# AUROC minus this tells us how much of the F-full gain is real content signal.
FEATURE_COLS_MISSING_INDICATOR_ONLY = sorted(MISSING_INDICATOR_COLS)
FEATURE_COLS_FULL_WITH_MISSING_INDICATOR = FEATURE_COLS_FULL + sorted(MISSING_INDICATOR_COLS)

# Columns explicitly excluded from features (labels, weak labels, anomaly signals, metadata)
EXCLUDE_COLS = {
    "case_id",
    "anomaly_type",
    "timestamp_window",
    "endpoint_key",
    "method",
    "normalized_path",
    "endpoint_service",
    "weak_is_anomaly",
    "label_confidence",
    "latency_anomaly_signal",
    "error_anomaly_signal",
    "5xx_anomaly_signal",
    "phase",
    "inject_start_ms",
    "inject_end_ms",
    "target_service",
    "target_endpoint",
    "anomaly_level",
    "is_target_endpoint",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_fused_corpus(
    data_root: Path, exclude: Iterable[str], expected_n_cases: int | None = None
) -> pd.DataFrame:
    """Concatenate per-case tt_fused_15s.csv into a single frame, dropping excluded cases.

    Missing per-case files are still skipped with a warning (a case genuinely
    absent from data_root is not this script's problem to fix), but a case
    count mismatch against expected_n_cases now raises instead of relying on a
    LOG.warning that a future rerun could easily miss in scrollback (PR #26
    review, Important #4) — the paper's "26 runs" framing depends on this
    count, and a silent drop to e.g. 20/26 would still emit a plausible AUROC.
    """
    cases = sorted(p for p in data_root.iterdir() if p.is_dir())
    excl = set(exclude)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for c in cases:
        if c.name in excl:
            continue
        f = c / "_pipeline_out" / "tt_fused_15s.csv"
        if not f.exists():
            LOG.warning("missing fused csv, skip: %s", f)
            missing.append(c.name)
            continue
        df = pd.read_csv(f)
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No fused csv found under {data_root} after exclusions {excl}")
    if expected_n_cases is not None and len(frames) != expected_n_cases:
        raise RuntimeError(
            f"loaded {len(frames)} cases, expected {expected_n_cases} "
            f"(data_root={data_root}, excluded={sorted(excl)}, missing_files={missing})"
        )
    out = pd.concat(frames, ignore_index=True)
    LOG.info("loaded %d rows from %d cases (excluded %s)", len(out), len(frames), sorted(excl))
    return out


def _temporal_train_mask(df: pd.DataFrame, subset_mask: np.ndarray, fraction: float) -> np.ndarray:
    """Which rows of `subset_mask` fall in the temporally-earliest `fraction` of
    their case's time windows, via the repo's existing per-case window splitter
    (same anti-leakage convention as split_v1.split_normal_rows_temporal: no
    eval window ever precedes a train window within the same case).
    """
    subset = df.loc[subset_mask, ["case_id", "timestamp_window"]]
    if subset.empty:
        return np.zeros(len(df), dtype=bool)
    train_part, _eval_part = split_fault_phase_temporal(
        subset, fraction=fraction, case_col="case_id", time_col="timestamp_window"
    )
    train_keys = pd.MultiIndex.from_frame(
        train_part[["case_id", "timestamp_window"]].drop_duplicates()
    )
    row_keys = pd.MultiIndex.from_frame(df[["case_id", "timestamp_window"]])
    return subset_mask & np.asarray(row_keys.isin(train_keys))


def build_train_pool(
    df: pd.DataFrame, pool_mode: str, fraction: float = TRAIN_POOL_FRACTION
) -> np.ndarray:
    """Boolean mask over rows indicating membership in the one-class training pool.

    Both regimes hold out the temporally-latest (1 - fraction) windows of every
    Normal case as eval negatives, instead of absorbing 100% of Normal into
    train — eval must retain genuine normal-state rows, or "expanded" degenerates
    to an eval set with zero true negatives (PR #26 review, Critical #1). The
    train/eval boundary is a per-case temporal window cut (via
    _temporal_train_mask / split_fault_phase_temporal), not a random row split.

    normal_only  : earliest `fraction` of each Normal case's windows
    expanded     : normal_only + earliest `fraction` of each fault case's
                   baseline-phase windows
    """
    is_normal_case = df["case_id"].str.startswith("Normal_").to_numpy()
    normal_train = _temporal_train_mask(df, is_normal_case, fraction)
    if pool_mode == "normal_only":
        return normal_train
    if pool_mode == "expanded":
        is_baseline_fault = (~is_normal_case) & (df["phase"].to_numpy() == "baseline")
        baseline_train = _temporal_train_mask(df, is_baseline_fault, fraction)
        return normal_train | baseline_train
    raise ValueError(f"unknown pool_mode: {pool_mode}")


def build_labels(df: pd.DataFrame) -> np.ndarray:
    """y_true = (phase == 'inject') AND (is_target_endpoint == 1).

    is_target_endpoint is float in the source csv (0.0/1.0 with possible NaN);
    the paper's endpoint-granularity definition requires an explicit == 1.
    """
    inject = df["phase"].to_numpy() == "inject"
    is_tgt = pd.to_numeric(df["is_target_endpoint"], errors="coerce").fillna(0).to_numpy() == 1
    return (inject & is_tgt).astype(np.int8)


def select_features(df: pd.DataFrame, feature_set: str) -> tuple[list[str], np.ndarray]:
    """Returns (column names, raw values). NaN is intentionally still present
    outside RATE_ZEROFILL_COLS — see impute_missing, which needs pool_mask
    (unknown at this point) to fill non-rate columns from train-pool
    statistics only, rather than a data-independent guess.

    missing_indicator_only / full_with_missing_indicator isolate the
    missingness confound flagged in PR #26 review Critical #2: on new_merge,
    client_content_length_rel_shift / client_body_hash_mismatch_rate are NaN
    on 49% of target-endpoint inject rows vs 15% of non-target rows, so part
    of F-full's AUROC gain over F-inherited may be the detector keying on
    "this field went missing here" rather than on real content-integrity
    signal. Comparing full vs full_with_missing_indicator's AUROC gain over
    missing_indicator_only tells apart the two effects.
    """
    if feature_set == "inherited":
        cols = FEATURE_COLS_INHERITED
        X = df[cols].to_numpy(dtype=np.float64)
    elif feature_set == "full":
        cols = FEATURE_COLS_FULL
        X = df[cols].to_numpy(dtype=np.float64)
    elif feature_set == "missing_indicator_only":
        cols = FEATURE_COLS_MISSING_INDICATOR_ONLY
        X = df[cols].isna().to_numpy(dtype=np.float64)  # 1.0 = missing, 0.0 = present
    elif feature_set == "full_with_missing_indicator":
        indicator_cols = FEATURE_COLS_MISSING_INDICATOR_ONLY
        X_base = df[FEATURE_COLS_FULL].to_numpy(dtype=np.float64)
        X_ind = df[indicator_cols].isna().to_numpy(dtype=np.float64)
        cols = FEATURE_COLS_FULL + [f"{c}__is_missing" for c in indicator_cols]
        X = np.concatenate([X_base, X_ind], axis=1)
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return cols, X


def impute_missing(X: np.ndarray, cols: list[str], pool_mask: np.ndarray) -> np.ndarray:
    """Fill remaining per-column NaN: RATE_ZEROFILL_COLS -> 0 (no events this
    window genuinely means rate/count is 0, a safe convention); every other
    column (latency stats, content-integrity fields) -> the train pool's own
    mean for that column, computed from non-NaN train-pool rows only. 0-filling
    a missing latency/content value would manufacture a fake best-case
    observation and bias both training and eval (PR #26 review Critical #3);
    using eval rows' own mean would leak eval statistics into the fill.
    *_is_missing indicator columns never contain NaN (they are isna() output),
    so they pass through this function unchanged.
    """
    X = X.copy()
    for j, c in enumerate(cols):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if not nan_mask.any():
            continue
        if c in RATE_ZEROFILL_COLS:
            col[nan_mask] = 0.0
        else:
            pool_vals = col[pool_mask]
            pool_vals = pool_vals[~np.isnan(pool_vals)]
            col[nan_mask] = float(pool_vals.mean()) if len(pool_vals) else 0.0
        X[:, j] = col
    return X


# ---------------------------------------------------------------------------
# Normalisation (fit on train only, transform all)
# ---------------------------------------------------------------------------
def fit_normalizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    # Avoid divide-by-zero on constant columns (e.g. trace_5xx in Normal)
    std = np.where(std < 1e-9, 1.0, std)
    return mean, std


def apply_normalizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Trivial baselines (internal only — NOT included in headline metrics by design)
# ---------------------------------------------------------------------------
def rel_pos_score(df: pd.DataFrame) -> np.ndarray:
    """Order of window within its case. Higher = later in the run.

    This is the trivial baseline that beats every learning model on the current
    dataset (entry 027). We compute it for internal reference but do NOT
    report it in the user-facing benchmark output.
    """
    order = df.groupby("case_id").cumcount().to_numpy(dtype=np.float64)
    return order


def per_case_zscore_score(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Per-case L2 z-score. Another trivial baseline from entry 027."""
    case_ids = df["case_id"].to_numpy()
    out = np.zeros(len(X), dtype=np.float64)
    for cid in np.unique(case_ids):
        mask = case_ids == cid
        Xi = X[mask]
        mu = Xi.mean(axis=0)
        sd = Xi.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        out[mask] = np.linalg.norm((Xi - mu) / sd, axis=1)
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass
class DetectorResult:
    model: str
    feature_set: str
    pool_mode: str
    seed: int
    n_features: int
    n_train: int
    n_eval: int
    n_pos_eval: int
    elapsed_sec: float
    metrics: dict
    # Provenance (CLAUDE.md experiment-tracking requirement: config + commit +
    # dataset version + seed) — seed is already a field above; commit/data_root/
    # feature_columns were missing entirely before this fix (PR #26 review,
    # Important #5).
    git_commit: str
    data_root: str
    feature_columns: list[str]
    n_cases_loaded: int


def fit_deep_svdd(
    X_train: np.ndarray,
    seed: int,
    hidden_dim: int = 64,
    rep_dim: int = 32,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> DeepSVDD:
    import torch

    set_seed(seed)
    input_dim = X_train.shape[1]
    svdd = DeepSVDD(input_dim=input_dim, hidden_dim=hidden_dim, rep_dim=rep_dim)
    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    svdd.init_center(X_t)
    optim = torch.optim.Adam(svdd.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        optim.zero_grad()
        loss = svdd.svdd_loss(X_t)
        loss.backward()
        optim.step()
    return svdd


def score_deep_svdd(svdd: DeepSVDD, X: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return svdd.score(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()


def fit_ocsvm(X_train: np.ndarray, seed: int) -> OneClassSVM:
    # nu chosen generously; this is a smoke-level run, not a tuned baseline.
    # seed is unused (sklearn's rbf OneClassSVM has no randomness), kept only
    # so fit_deep_svdd/fit_ocsvm/fit_iforest share one call signature in main().
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_train)


def score_ocsvm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    # decision_function: higher = more inlier; we want higher = more anomalous
    return -model.decision_function(X)


def fit_iforest(X_train: np.ndarray, seed: int) -> IsolationForest:
    return IsolationForest(n_estimators=100, random_state=seed, n_jobs=1).fit(X_train)


def score_iforest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    # score_samples: higher = more normal; we want higher = more anomalous
    return -model.score_samples(X)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def per_case_macro(
    metric_fn, y_true: np.ndarray, y_score: np.ndarray, case_ids: np.ndarray
) -> tuple[float | None, int]:
    """Per-case macro of metric_fn, then mean across cases that had >=1 pos & >=1 neg."""
    vals: list[float] = []
    n_qualifying = 0
    for cid in np.unique(case_ids):
        m = case_ids == cid
        yt, ys = y_true[m], y_score[m]
        if len(np.unique(yt)) < 2:
            continue
        v = metric_fn(yt, ys)
        if v is not None:
            vals.append(v)
            n_qualifying += 1
    if not vals:
        return None, 0
    return float(np.mean(vals)), n_qualifying


def evaluate(y_true: np.ndarray, y_score: np.ndarray, case_ids: np.ndarray) -> dict:
    """Overall + per-case macro for AUROC and AUPRC."""
    overall_auroc = safe_auc(y_true, y_score)
    overall_auprc = safe_auprc(y_true, y_score)
    per_case_auroc, n_auroc = per_case_macro(safe_auc, y_true, y_score, case_ids)
    per_case_auprc, n_auprc = per_case_macro(safe_auprc, y_true, y_score, case_ids)
    return {
        "overall_auroc": overall_auroc,
        "overall_auprc": overall_auprc,
        "per_case_auroc_macro": per_case_auroc,
        "per_case_auprc_macro": per_case_auprc,
        "n_cases_with_both_classes_for_auroc": n_auroc,
        "n_cases_with_both_classes_for_auprc": n_auprc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--feature-set",
        choices=["inherited", "full", "missing_indicator_only", "full_with_missing_indicator"],
        default="full",
    )
    p.add_argument("--pool-mode", choices=["normal_only", "expanded"], default="expanded")
    p.add_argument("--model", choices=["deep_svdd", "ocsvm", "iforest"], default="deep_svdd")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=Path("data/new_merge"))
    p.add_argument("--exclude-cases", nargs="*", default=DEFAULT_EXCLUDE_CASES)
    p.add_argument("--expected-n-cases", type=int, default=26)
    p.add_argument("--out-root", type=Path, default=Path("artifacts/downstream_benchmark"))
    p.add_argument("--epochs", type=int, default=50)
    return p.parse_args()


def _git_commit() -> str:
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)

    t0 = time.time()
    df = load_fused_corpus(args.data_root, args.exclude_cases, args.expected_n_cases)
    n_cases_loaded = df["case_id"].nunique()
    cols, X_raw = select_features(df, args.feature_set)
    y_true = build_labels(df)
    pool_mask = build_train_pool(df, args.pool_mode)
    X = impute_missing(X_raw, cols, pool_mask)
    case_ids = df["case_id"].to_numpy()

    LOG.info(
        "feature_set=%s (%d cols) | pool=%s | model=%s | seed=%d",
        args.feature_set,
        len(cols),
        args.pool_mode,
        args.model,
        args.seed,
    )
    LOG.info(
        "total rows: %d | train pool: %d | pos in eval: %d",
        len(df),
        int(pool_mask.sum()),
        int(y_true[~pool_mask].sum()),
    )

    X_train = X[pool_mask]
    mean, std = fit_normalizer(X_train)
    X_all_norm = apply_normalizer(X, mean, std)
    X_train_norm = X_all_norm[pool_mask]
    X_eval_norm = X_all_norm[~pool_mask]
    y_eval = y_true[~pool_mask]
    case_eval = case_ids[~pool_mask]

    if args.model == "deep_svdd":
        t_fit = time.time()
        svdd = fit_deep_svdd(X_train_norm, args.seed, epochs=args.epochs)
        fit_sec = time.time() - t_fit
        t_score = time.time()
        scores_eval = score_deep_svdd(svdd, X_eval_norm)
        score_sec = time.time() - t_score
    elif args.model == "ocsvm":
        t_fit = time.time()
        m = fit_ocsvm(X_train_norm, args.seed)
        fit_sec = time.time() - t_fit
        t_score = time.time()
        scores_eval = score_ocsvm(m, X_eval_norm)
        score_sec = time.time() - t_score
    elif args.model == "iforest":
        t_fit = time.time()
        m = fit_iforest(X_train_norm, args.seed)
        fit_sec = time.time() - t_fit
        t_score = time.time()
        scores_eval = score_iforest(m, X_eval_norm)
        score_sec = time.time() - t_score
    else:
        raise ValueError(args.model)

    metrics = evaluate(y_eval, scores_eval, case_eval)

    # Internal-only trivial baselines (rel_pos, per-case zscore) — NOT in headline
    rel_pos_eval = rel_pos_score(df)[~pool_mask]
    zscore_eval = per_case_zscore_score(X_all_norm, df)[~pool_mask]
    metrics["_internal_only_rel_pos_overall_auroc"] = safe_auc(y_eval, rel_pos_eval)
    metrics["_internal_only_rel_pos_per_case_auroc_macro"] = per_case_macro(
        safe_auc, y_eval, rel_pos_eval, case_eval
    )[0]
    metrics["_internal_only_zscore_overall_auroc"] = safe_auc(y_eval, zscore_eval)
    metrics["_internal_only_zscore_per_case_auroc_macro"] = per_case_macro(
        safe_auc, y_eval, zscore_eval, case_eval
    )[0]

    run_dir = args.out_root / f"{args.feature_set}_{args.pool_mode}_{args.model}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scores_df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_eval)),
            "score": scores_eval,
            "y_true": y_eval,
            "case_id": case_eval,
            "phase": df.loc[~pool_mask, "phase"].to_numpy(),
            "endpoint_key": df.loc[~pool_mask, "endpoint_key"].to_numpy(),
            "is_target_endpoint": pd.to_numeric(
                df.loc[~pool_mask, "is_target_endpoint"], errors="coerce"
            )
            .fillna(0)
            .to_numpy(),
        }
    )
    scores_df.to_parquet(run_dir / "scores.parquet", index=False)

    result = DetectorResult(
        model=args.model,
        feature_set=args.feature_set,
        pool_mode=args.pool_mode,
        seed=args.seed,
        n_features=len(cols),
        n_train=int(pool_mask.sum()),
        n_eval=int((~pool_mask).sum()),
        n_pos_eval=int(y_eval.sum()),
        elapsed_sec=time.time() - t0,
        metrics=metrics,
        git_commit=_git_commit(),
        data_root=str(args.data_root),
        feature_columns=cols,
        n_cases_loaded=int(n_cases_loaded),
    )
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    LOG.info(
        "DONE in %.1fs (fit=%.1fs score=%.1fs) | overall_auroc=%s per_case_auroc=%s | "
        "overall_auprc=%s per_case_auprc=%s",
        result.elapsed_sec,
        fit_sec,
        score_sec,
        metrics["overall_auroc"],
        metrics["per_case_auroc_macro"],
        metrics["overall_auprc"],
        metrics["per_case_auprc_macro"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
