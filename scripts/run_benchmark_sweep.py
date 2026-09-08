"""Sweep runner for scripts/benchmark_downstream.py.

Iterates the full 48-cell grid (2 feature arms × 2 pool modes × 3 models × 4 seeds)
with checkpoint/resume support. Logs each run's outcome to artifacts/downstream_benchmark/_sweep.log
and writes a running summary to artifacts/downstream_benchmark/_summary.tsv.

The 48 cells:
  - feature_set ∈ {inherited, full}        (F-inherited=18 dims, F-full=21 dims)
  - pool_mode  ∈ {normal_only, expanded}   (817 rows vs 15,104 rows)
  - model      ∈ {deep_svdd, ocsvm, iforest}
  - seed       ∈ {1, 2, 3, 42}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

CELLS = [
    # (feature_set, pool_mode, model, seed)
    ("inherited", "normal_only", "deep_svdd", 42),
    ("inherited", "normal_only", "deep_svdd", 1),
    ("inherited", "normal_only", "deep_svdd", 2),
    ("inherited", "normal_only", "deep_svdd", 3),
    ("inherited", "normal_only", "ocsvm", 42),
    ("inherited", "normal_only", "iforest", 42),
    ("inherited", "normal_only", "iforest", 1),
    ("inherited", "normal_only", "iforest", 2),
    ("inherited", "normal_only", "iforest", 3),
    ("inherited", "expanded", "deep_svdd", 42),
    ("inherited", "expanded", "deep_svdd", 1),
    ("inherited", "expanded", "deep_svdd", 2),
    ("inherited", "expanded", "deep_svdd", 3),
    ("inherited", "expanded", "ocsvm", 42),
    ("inherited", "expanded", "iforest", 42),
    ("inherited", "expanded", "iforest", 1),
    ("inherited", "expanded", "iforest", 2),
    ("inherited", "expanded", "iforest", 3),
    ("full", "normal_only", "deep_svdd", 42),
    ("full", "normal_only", "deep_svdd", 1),
    ("full", "normal_only", "deep_svdd", 2),
    ("full", "normal_only", "deep_svdd", 3),
    ("full", "normal_only", "ocsvm", 42),
    ("full", "normal_only", "iforest", 42),
    ("full", "normal_only", "iforest", 1),
    ("full", "normal_only", "iforest", 2),
    ("full", "normal_only", "iforest", 3),
    ("full", "expanded", "deep_svdd", 42),
    ("full", "expanded", "deep_svdd", 1),
    ("full", "expanded", "deep_svdd", 2),
    ("full", "expanded", "deep_svdd", 3),
    ("full", "expanded", "ocsvm", 42),
    ("full", "expanded", "iforest", 42),
    ("full", "expanded", "iforest", 1),
    ("full", "expanded", "iforest", 2),
    ("full", "expanded", "iforest", 3),
    # Note: ocsvm is only run at seed=42 because it is deterministic for a given nu/gamma
    # (sklearn OneClassSVM random_state is unused in rbf mode), so multi-seed is
    # meaningless. Same for iforest with bootstrap=False effect on scores.
    # Only DeepSVDD benefits from multi-seed. We still run IF at 4 seeds for the
    # variance estimate (IF score_samples is deterministic given data, but bootstrap
    # subsampling introduces a tiny variance).
]

# Re-balance: 2*2*3*4=48. Replace the rough sketch above with a proper grid.
FEATURE_SETS = ["inherited", "full"]
POOL_MODES = ["normal_only", "expanded"]
MODELS = ["deep_svdd", "ocsvm", "iforest"]
SEEDS = [1, 2, 3, 42]


def build_full_grid() -> list[tuple[str, str, str, int]]:
    return [
        (fs, pm, m, s) for fs in FEATURE_SETS for pm in POOL_MODES for m in MODELS for s in SEEDS
    ]


def run_dir(out_root: Path, fs: str, pm: str, m: str, s: int) -> Path:
    return out_root / f"{fs}_{pm}_{m}_seed{s}"


def already_done(out_root: Path, fs: str, pm: str, m: str, s: int) -> bool:
    rd = run_dir(out_root, fs, pm, m, s)
    return (rd / "metrics.json").exists() and (rd / "scores.parquet").exists()


def run_one(
    out_root: Path, fs: str, pm: str, m: str, s: int, epochs: int
) -> tuple[bool, str, float]:
    cmd = [
        sys.executable,
        "scripts/benchmark_downstream.py",
        "--feature-set",
        fs,
        "--pool-mode",
        pm,
        "--model",
        m,
        "--seed",
        str(s),
        "--out-root",
        str(out_root),
        "--epochs",
        str(epochs),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        return False, proc.stderr[-1000:] if proc.stderr else "no stderr", elapsed
    return True, proc.stdout[-500:] if proc.stdout else "", elapsed


def write_summary(summary_path: Path, rows: list[dict]) -> None:
    keys = [
        "feature_set",
        "pool_mode",
        "model",
        "seed",
        "overall_auroc",
        "overall_auprc",
        "per_case_auroc_macro",
        "per_case_auprc_macro",
        "n_train",
        "n_eval",
        "n_pos_eval",
        "elapsed_sec",
        "_internal_rel_pos_per_case_auroc",
        "_internal_zscore_per_case_auroc",
    ]
    with open(summary_path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in keys) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", type=Path, default=Path("artifacts/downstream_benchmark"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="If set, only run cells whose 4-tuple-as-string contains one of these substrings",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Stop after this many new runs (for quick check)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_path = args.out_root / "_sweep.log"
    summary_path = args.out_root / "_summary.tsv"

    grid = build_full_grid()
    if args.include:
        grid = [
            c for c in grid if any(s in f"{c[0]}_{c[1]}_{c[2]}_seed{c[3]}" for s in args.include)
        ]

    rows: list[dict] = []
    if summary_path.exists():
        # Re-load prior summary so resume doesn't clobber.
        with open(summary_path) as f:
            lines = f.readlines()[1:]
        for ln in lines:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "feature_set": parts[0],
                    "pool_mode": parts[1],
                    "model": parts[2],
                    "seed": int(parts[3]),
                }
            )

    new_runs = 0
    with open(log_path, "a") as logf:
        for fs, pm, m, s in grid:
            if args.limit is not None and new_runs >= args.limit:
                break
            rd = run_dir(args.out_root, fs, pm, m, s)
            if already_done(args.out_root, fs, pm, m, s):
                logf.write(f"[skip] {fs}_{pm}_{m}_seed{s} (already done)\n")
                # Re-extract metrics into summary
                with open(rd / "metrics.json") as mf:
                    d = json.load(mf)
                existing = next(
                    (
                        r
                        for r in rows
                        if r["feature_set"] == fs
                        and r["pool_mode"] == pm
                        and r["model"] == m
                        and r["seed"] == s
                    ),
                    None,
                )
                if existing is None:
                    existing = {"feature_set": fs, "pool_mode": pm, "model": m, "seed": s}
                    rows.append(existing)
                met = d.get("metrics", {})
                existing["overall_auroc"] = met.get("overall_auroc")
                existing["overall_auprc"] = met.get("overall_auprc")
                existing["per_case_auroc_macro"] = met.get("per_case_auroc_macro")
                existing["per_case_auprc_macro"] = met.get("per_case_auprc_macro")
                existing["n_train"] = d.get("n_train")
                existing["n_eval"] = d.get("n_eval")
                existing["n_pos_eval"] = d.get("n_pos_eval")
                existing["elapsed_sec"] = d.get("elapsed_sec")
                existing["_internal_rel_pos_per_case_auroc"] = met.get(
                    "_internal_only_rel_pos_per_case_auroc_macro"
                )
                existing["_internal_zscore_per_case_auroc"] = met.get(
                    "_internal_only_zscore_per_case_auroc_macro"
                )
                continue

            logf.write(f"[run ] {fs}_{pm}_{m}_seed{s} ...\n")
            logf.flush()
            print(f"==> {fs}_{pm}_{m}_seed{s}", flush=True)
            ok, msg, elapsed = run_one(args.out_root, fs, pm, m, s, args.epochs)
            logf.write(f"      {'OK' if ok else 'FAIL'} in {elapsed:.1f}s | {msg}\n")
            logf.flush()
            if not ok:
                print(f"!! FAIL: {fs}_{pm}_{m}_seed{s}", flush=True)
                print(msg, flush=True)
                continue
            new_runs += 1
            # Read metrics
            with open(rd / "metrics.json") as mf:
                d = json.load(mf)
            met = d.get("metrics", {})
            row = {
                "feature_set": fs,
                "pool_mode": pm,
                "model": m,
                "seed": s,
                "overall_auroc": met.get("overall_auroc"),
                "overall_auprc": met.get("overall_auprc"),
                "per_case_auroc_macro": met.get("per_case_auroc_macro"),
                "per_case_auprc_macro": met.get("per_case_auprc_macro"),
                "n_train": d.get("n_train"),
                "n_eval": d.get("n_eval"),
                "n_pos_eval": d.get("n_pos_eval"),
                "elapsed_sec": d.get("elapsed_sec"),
                "_internal_rel_pos_per_case_auroc": met.get(
                    "_internal_only_rel_pos_per_case_auroc_macro"
                ),
                "_internal_zscore_per_case_auroc": met.get(
                    "_internal_only_zscore_per_case_auroc_macro"
                ),
            }
            rows.append(row)
            write_summary(summary_path, rows)
            print(
                f"   overall_auroc={row['overall_auroc']} per_case_auroc={row['per_case_auroc_macro']} "
                f"overall_auprc={row['overall_auprc']} per_case_auprc={row['per_case_auprc_macro']}",
                flush=True,
            )

    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
