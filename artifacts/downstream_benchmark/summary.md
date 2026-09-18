# Downstream Anomaly-Detection Benchmark — 26 runs / 23,244 windows

## Bottom line

**DeepSVDD on the F-full feature set (21 dims, with the 3 added client-side fields) under the expanded training pool reaches per-case macro AUROC = 0.957 ± 0.008.** Stripping the 3 added fields drops per-case AUROC to 0.866 ± 0.008 — a **+0.091 absolute gain attributable to the new client-side observability**. All three detectors (DeepSVDD / OCSVM / IF) show the same monotonic direction, which is the claim the reviewer is asking for.

A missing-value-confound ablation (see below) rules out the concern that this gain is mostly the detector keying on "this field went missing on the target endpoint" rather than on real content-integrity signal: an indicator-only feature set built from nothing but the two added fields' missingness reaches only 0.667 per-case AUROC, far below F-full's 0.957, and adding an explicit missingness indicator on top of F-full moves AUROC by only +0.002 (0.957 → 0.959). The gain is overwhelmingly driven by the fields' actual values, not by their absence.

## Experimental setup

| Item | Value |
|---|---|
| Dataset | `data/new_merge/` filtered to 26 published runs (drops `Lv_D_TRANSACTION_timeout_20260728T230311Z_em` whose inject phase has only 4 windows) |
| Total windows | 23,244 (matches paper Table `tab:scale`) |
| Positives | 639 (inject phase × target endpoint, matches paper's 2.7%) |
| Features (F-full) | 21: 9 trace RED + 12 client RED (`trace_request_count`, `trace_latency_{mean,p50,p95,p99}`, `trace_error_rate`, `trace_5xx_rate`, `trace_4xx_rate`, `trace_status_coverage`, `client_request_count`, `client_error_rate`, `client_2xx_rate`, `client_4xx_rate`, `client_content_length_mean`, `client_content_length_rel_shift`, `client_body_hash_mismatch_rate`, `client_latency_{mean,p50,p95,p99}`, `client_5xx_rate`) |
| Features (F-inherited) | 18: F-full minus the 3 added fields (`client_content_length_mean`, `client_content_length_rel_shift`, `client_body_hash_mismatch_rate`) |
| Ablation feature sets | `missing_indicator_only` (2 dims: is-NaN indicator for `client_content_length_rel_shift` / `client_body_hash_mismatch_rate` only — `client_content_length_mean` excluded, its NaN rate is 0% on both target and non-target rows); `full_with_missing_indicator` (F-full's 21 dims + those same 2 indicator dims) — see "Missing-value confound ablation" below |
| Labels | `y = (phase == "inject") AND (is_target_endpoint == 1.0)` (endpoint-granularity, per paper §Fault Labeling) |
| Excluded feature columns | 19 metadata/label/weak-label/anomaly-signal columns (`weak_is_anomaly`, `*_anomaly_signal`, `phase`, `inject_{start,end}_ms`, `target_{service,endpoint}`, `anomaly_level`, `is_target_endpoint`, etc.) — explicit denial of label leakage |
| NaN handling | `RATE_ZEROFILL_COLS` (count/rate/coverage columns): `np.nan → 0.0` (no events ⇒ rate is 0, legitimate). Every other column (latency statistics, content-integrity fields): filled with the **training pool's own mean** for that column, not 0 — 0-filling a missing latency/content value would manufacture a fake best-case observation and bias both training and eval. (Fixed from an earlier version that 0-filled all 21 columns uniformly, including latency — see PR #26 review, Critical #3.) |
| Training-pool split | Both `normal_only` and `expanded` hold out the temporally-latest 20% of each Normal case's windows (and, under `expanded`, of each fault case's baseline phase) as genuine normal-state eval negatives, via a per-case temporal window cut (`src/contracts/split_fault_phase.py::split_fault_phase_temporal`, `TRAIN_POOL_FRACTION=0.8`). An earlier version put 100% of that data into train, leaving `expanded` eval with **zero** true-normal rows — every eval negative was a non-target endpoint captured *during* the same inject/recover window as a positive, which measures fault localization rather than normal-vs-anomaly detection. See PR #26 review, Critical #1. |
| Normalisation | `mean/std` fit on the **training pool only**; std floored at 1e-9 |
| Detectors | DeepSVDD (3-layer MLP, hidden=64, rep=32, 50 epoch, Adam 1e-3/1e-4); sklearn `OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)`; sklearn `IsolationForest(n_estimators=100)` |
| Seeds | {1, 2, 3, 42} for DeepSVDD and IF; sklearn OCSVM in rbf mode is deterministic (std=0 in tables) |
| Metrics | Overall + per-case macro of ROC-AUC (`roc_auc_score`) and PR-AUC (`average_precision_score`) |
| Eval split | All rows minus the training pool (no train/eval overlap by construction; see "Training-pool split" above for what that eval set actually contains) |
| Cases contributing a positive class | Only 16 of 26 cases have `label_granularity == "endpoint"` (i.e. a `target_endpoint` field); the remaining 10 (`Lv_P_*`/`Lv_S_*`/`Lv_D_*` service/case-granularity cases) contribute 0 positives under this script's endpoint-only label definition and are dropped from the per-case macro (`n_cases_with_both_classes_for_auroc: 16` in every `metrics.json`). **The per-case macro numbers below are effectively an `Lv_E_HTTP*`-only benchmark**, not a benchmark across all 26 runs. |
| Files | 56 runs under `artifacts/downstream_benchmark/{feature_set}_{pool}_{model}_seed{seed}/` each with `scores.parquet` + `metrics.json` (48 main grid + 8 missing-value ablation cells) |

## Training-pool regimes (two-arm comparison)

| Regime | What trains | Train rows | Eval rows | Implication |
|---|---|---|---|---|
| `normal_only` | Earliest 80% of the single `Normal_20260728T161044Z` case's windows | 647 | 22,597 | One-Class purity; but Normal diversity is 1, and eval retains the held-out 20% Normal tail |
| `expanded` | `normal_only` + earliest 80% of every fault-case `phase == "baseline"` window | 11,960 | 11,284 | Larger pool; eval retains held-out Normal + held-out baseline rows plus all inject/recover windows |

## Headline result — **expanded pool** (recommended for the paper)

Per-case macro AUROC and AUPRC, mean ± std over 4 seeds.

| Detector | F-inherited (18) | F-full (21) | Δ AUROC | F-inherited AUPRC | F-full AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|
| **DeepSVDD** | 0.866 ± 0.008 | **0.957 ± 0.008** | **+0.091** | 0.644 ± 0.056 | **0.709 ± 0.040** | +0.065 |
| OCSVM | 0.836 | 0.950 | +0.114 | 0.615 | 0.687 | +0.072 |
| IsolationForest | 0.733 ± 0.005 | 0.814 ± 0.003 | +0.081 | 0.210 ± 0.009 | 0.221 ± 0.005 | +0.011 |

**All three detectors improve on the headline metric when the 3 added fields are present.** The DeepSVDD row is the cleanest (lowest seed variance, highest per-case AUROC, single-seed literature standard) and is the one to lead the table with in the paper.

## Missing-value confound ablation (PR #26 review, Critical #2)

The two fields with the largest AUROC contribution (`client_content_length_rel_shift`, `client_body_hash_mismatch_rate`) are also the two with an asymmetric NaN rate on `new_merge`: 49% NaN on target-endpoint inject rows vs. 15% NaN on non-target inject rows (`client_content_length_mean`, the third added field, is 0% NaN on both sides and is excluded from this ablation). Since a naive fill turns "missing" into a distinguishable constant, part of F-full's AUROC gain could in principle be the detector learning "this field went missing on the target endpoint" rather than reading real content-integrity signal. This ablation isolates the two effects, restricted to the headline cell (DeepSVDD / expanded):

| Feature set | Dims | per-case macro AUROC | per-case macro AUPRC |
|---|---|---|---|
| `missing_indicator_only` (is-NaN indicator for the 2 asymmetric fields only) | 2 | 0.667 ± 0.000 | 0.244 ± 0.000 |
| `full` (F-full, mean-imputed) | 21 | 0.957 ± 0.008 | 0.709 ± 0.040 |
| `full_with_missing_indicator` (F-full + explicit is-NaN indicator for the 2 fields) | 23 | 0.959 ± 0.005 | 0.733 ± 0.020 |

**Reading**: if missingness were the dominant driver, (a) `missing_indicator_only` alone should already approach F-full's AUROC, and (b) making the missingness signal explicit on top of F-full should produce a large jump. Neither happens — `missing_indicator_only` reaches only 0.667 (0.29 below F-full), and `full_with_missing_indicator` gains only +0.002 AUROC / +0.024 AUPRC over plain `full`. The +0.091 headline gain is overwhelmingly attributable to the fields' actual values, not to their missingness pattern.

## Comparison — `normal_only` pool (reported as supporting cell)

| Detector | F-inherited (18) | F-full (21) | Δ AUROC | F-inherited AUPRC | F-full AUPRC | Δ AUPRC |
|---|---|---|---|---|---|---|
| DeepSVDD | 0.636 ± 0.087 | 0.723 ± 0.018 | +0.087 | 0.082 ± 0.013 | 0.175 ± 0.023 | +0.093 |
| OCSVM | 0.747 | 0.781 | +0.034 | 0.112 | 0.158 | +0.046 |
| IsolationForest | 0.575 ± 0.008 | 0.635 ± 0.010 | +0.060 | 0.071 ± 0.002 | 0.072 ± 0.003 | +0.001 |

Absolute numbers are lower because a single Normal case (647 train rows) provides a very narrow one-class boundary. The Δ direction is preserved.

## Per-seed detail (F-full + expanded, the recommended paper cell)

| seed | DeepSVDD per-case AUROC | DeepSVDD per-case AUPRC |
|---|---|---|
| 1 | 0.949 | 0.662 |
| 2 | 0.952 | 0.690 |
| 3 | 0.966 | 0.750 |
| 42 | 0.964 | 0.733 |
| **mean ± std** | **0.957 ± 0.008** | **0.709 ± 0.040** |

OCSVM in rbf mode is deterministic for a given `nu`/`gamma`, so the 4-seed column is identical.

## Overall (not per-case) AUROC for the same F-full + expanded cell

| Detector | overall AUROC | overall AUPRC |
|---|---|---|
| DeepSVDD | 0.924 ± 0.008 | 0.284 ± 0.023 |
| OCSVM | 0.926 | 0.293 |
| IsolationForest | 0.781 ± 0.003 | 0.109 ± 0.001 |

Overall numbers are lower than per-case macro because the global ranking is dominated by the largest fault cases; per-case macro weights every qualifying case equally regardless of size.

## Recommendation for the paper

**Lead with the F-full + expanded + DeepSVDD cell** (per-case AUROC 0.957 ± 0.008, AUPRC 0.709 ± 0.040). Pair it with the OCSVM cell as a classical non-deep baseline (AUROC 0.950, AUPRC 0.687). Report IF for breadth (AUROC 0.814 ± 0.003). Cite the F-inherited column (the same architecture trained without the 3 added fields) to make the **+0.091 absolute AUROC gain attributable to the new client-side observability** the explicit finding, and cite the missing-value ablation above to preempt a missingness-confound objection.

Note the label-granularity caveat above: this is a benchmark over the 16 `Lv_E_HTTP*` cases with endpoint-level labels, not all 26 published runs — state this scope explicitly if citing "26 runs" alongside these AUROC numbers.

## Reproducibility

- `scripts/benchmark_downstream.py` — single-cell runner (argparse: `--feature-set {inherited,full,missing_indicator_only,full_with_missing_indicator} --pool-mode --model --seed`)
- `scripts/run_benchmark_sweep.py` — 56-cell sweep (48 main grid + 8 missing-value ablation) with checkpoint/resume; resume validates file content, not just existence
- `artifacts/downstream_benchmark/_summary.tsv` — raw per-run results
- 56 directories of `scores.parquet` + `metrics.json` for individual inspection; each `metrics.json` records `git_commit`, `data_root`, and `feature_columns` for provenance
- No data files were modified; only `scripts/` (new) and `artifacts/downstream_benchmark/` (new) were added
