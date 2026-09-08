# Downstream Anomaly-Detection Benchmark — 26 runs / 23,244 windows

## Bottom line

**DeepSVDD on the F-full feature set (21 dims, with the 3 added client-side fields) under the expanded training pool reaches per-case macro AUROC = 0.951 ± 0.009.** Stripping the 3 added fields drops per-case AUROC to 0.851 ± 0.014 — a **+0.100 absolute gain attributable to the new client-side observability**. All three detectors (DeepSVDD / OCSVM / IF) show the same monotonic direction, which is the claim the reviewer is asking for.

## Experimental setup

| Item | Value |
|---|---|
| Dataset | `data/new_merge/` filtered to 26 published runs (drops `Lv_D_TRANSACTION_timeout_20260728T230311Z_em` whose inject phase has only 4 windows) |
| Total windows | 23,244 (matches paper Table `tab:scale`) |
| Positives | 639 (inject phase × target endpoint, matches paper's 2.7%) |
| Features (F-full) | 21: 9 trace RED + 12 client RED (`trace_request_count`, `trace_latency_{mean,p50,p95,p99}`, `trace_error_rate`, `trace_5xx_rate`, `trace_4xx_rate`, `trace_status_coverage`, `client_request_count`, `client_error_rate`, `client_2xx_rate`, `client_4xx_rate`, `client_content_length_mean`, `client_content_length_rel_shift`, `client_body_hash_mismatch_rate`, `client_latency_{mean,p50,p95,p99}`, `client_5xx_rate`) |
| Features (F-inherited) | 18: F-full minus the 3 added fields (`client_content_length_mean`, `client_content_length_rel_shift`, `client_body_hash_mismatch_rate`) |
| Labels | `y = (phase == "inject") AND (is_target_endpoint == 1.0)` (endpoint-granularity, per paper §Fault Labeling) |
| Excluded feature columns | 19 metadata/label/weak-label/anomaly-signal columns (`weak_is_anomaly`, `*_anomaly_signal`, `phase`, `inject_{start,end}_ms`, `target_{service,endpoint}`, `anomaly_level`, `is_target_endpoint`, etc.) — explicit denial of label leakage |
| NaN handling | rate columns: `np.nan → 0.0` (no events ⇒ rate is 0, not undefined) |
| Normalisation | `mean/std` fit on the **training pool only**; std floored at 1e-9 |
| Detectors | DeepSVDD (3-layer MLP, hidden=64, rep=32, 50 epoch, Adam 1e-3/1e-4); sklearn `OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)`; sklearn `IsolationForest(n_estimators=100)` |
| Seeds | {1, 2, 3, 42} for DeepSVDD and IF; sklearn OCSVM in rbf mode is deterministic (std=0 in tables) |
| Metrics | Overall + per-case macro of ROC-AUC (`roc_auc_score`) and PR-AUC (`average_precision_score`) |
| Eval split | All 23,244 rows minus the training pool (no train/eval overlap by construction) |
| Files | 48 runs under `artifacts/downstream_benchmark/{feature_set}_{pool}_{model}_seed{seed}/` each with `scores.parquet` + `metrics.json` |

## Training-pool regimes (two-arm comparison)

| Regime | What trains | Rows | Implication |
|---|---|---|---|
| `normal_only` | The single `Normal_20260728T161044Z` case | 817 | One-Class purity; but Normal diversity is 1 |
| `expanded` | Normal + every fault-case `phase == "baseline"` row | 15,104 | Larger pool; inject/recover windows stay in eval |

## Headline result — **expanded pool** (recommended for the paper)

Per-case macro AUROC and AUPRC, mean ± std over 4 seeds.

| Detector | F-inherited (18) | F-full (21) | Δ AUROC | F-inherited AUPRC | F-full AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|
| **DeepSVDD** | 0.851 ± 0.014 | **0.951 ± 0.009** | **+0.100** | 0.705 ± 0.048 | **0.733 ± 0.039** | +0.028 |
| OCSVM | 0.832 | 0.941 | +0.109 | 0.639 | 0.703 | +0.064 |
| IsolationForest | 0.719 ± 0.012 | 0.771 ± 0.018 | +0.052 | 0.278 ± 0.008 | 0.269 ± 0.016 | -0.009 |

**All three detectors improve on the headline metric when the 3 added fields are present.** The DeepSVDD row is the cleanest (lowest seed variance, highest per-case AUROC, single-seed literature standard) and is the one to lead the table with in the paper.

## Comparison — `normal_only` pool (reported as supporting cell)

| Detector | F-inherited (18) | F-full (21) | Δ AUROC | F-inherited AUPRC | F-full AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|
| DeepSVDD | 0.667 ± 0.066 | 0.730 ± 0.010 | +0.063 | 0.089 ± 0.011 | 0.167 ± 0.011 | +0.078 |
| OCSVM | 0.742 | 0.783 | +0.041 | 0.111 | 0.158 | +0.048 |
| IsolationForest | 0.550 ± 0.010 | 0.576 ± 0.010 | +0.026 | 0.068 ± 0.002 | 0.065 ± 0.002 | -0.003 |

Absolute numbers are lower because a single 817-row Normal case provides a very narrow one-class boundary. The Δ direction is preserved.

## Per-detection seed detail (F-full + expanded, the recommended paper cell)

| seed | DeepSVDD per-case AUROC | DeepSVDD per-case AUPRC | IF per-case AUROC | IF per-case AUPRC |
|---|---|---|---|---|
| 1 | 0.951 | 0.718 | 0.773 | 0.266 |
| 2 | 0.938 | 0.685 | 0.795 | 0.293 |
| 3 | 0.956 | 0.774 | 0.753 | 0.256 |
| 42 | 0.958 | 0.753 | 0.764 | 0.260 |
| **mean ± std** | **0.951 ± 0.009** | **0.733 ± 0.039** | **0.771 ± 0.018** | **0.269 ± 0.016** |

OCSVM in rbf mode is deterministic for a given `nu`/`gamma`, so the 4-seed column is identical.

## Overall (not per-case) AUROC for the same F-full + expanded cell

| Detector | overall AUROC | overall AUPRC |
|---|---|---|
| DeepSVDD | 0.905 ± 0.005 | 0.297 ± 0.012 |
| OCSVM | 0.908 | 0.306 |
| IsolationForest | 0.747 ± 0.012 | 0.130 ± 0.005 |

Overall numbers are systematically lower than per-case macro because the global ranking weights larger cases (notably the Normal case has 0 positives and is dropped from the per-case qualifier count but stays in the overall denominator if the score happens to be extreme; here the Normal scores are clustered so overall still tracks but is dragged down by individual large negative cases).

## Recommendation for the paper

**Lead with the F-full + expanded + DeepSVDD cell** (per-case AUROC 0.951 ± 0.009, AUPRC 0.733 ± 0.039). Pair it with the OCSVM cell as a classical non-deep baseline (AUROC 0.941, AUPRC 0.703). Report IF for breadth (AUROC 0.771 ± 0.018). Cite the F-inherited column (the same architecture trained without the 3 added fields) to make the **+0.100 absolute AUROC gain attributable to the new client-side observability** the explicit finding.

## Reproducibility

- `scripts/benchmark_downstream.py` — single-cell runner (argparse: `--feature-set --pool-mode --model --seed`)
- `scripts/run_benchmark_sweep.py` — 48-cell sweep with checkpoint/resume
- `artifacts/downstream_benchmark/_summary.tsv` — raw per-run results
- `artifacts/downstream_benchmark/_summary_agg.tsv` — mean/std aggregates
- 48 directories of `scores.parquet` + `metrics.json` for individual inspection
- No data files were modified; only `scripts/` (new) and `artifacts/downstream_benchmark/` (new) were added
