"""Audit immutable SN prior-GRU score artifacts across fixed seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.service.modeling.sn_adapter import build_sn_prepared


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(scores: pd.DataFrame, prepared) -> dict:
    labels = prepared.labels.set_index("sample_id").label.reindex(scores.sample_id).to_numpy()
    known = pd.notna(labels)
    y = labels[known].astype(int)
    values = scores.score.to_numpy(float)[known]
    if len(np.unique(y)) != 2:
        return {"auroc": None, "ap": None, "rows": int(len(y))}
    return {
        "auroc": float(roc_auc_score(y, values)),
        "ap": float(average_precision_score(y, values)),
        "rows": int(len(y)),
        "positive_prevalence": float((y == 1).mean()),
        "constant_score_ap_baseline": float((y == 1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = build_sn_prepared(args.prepared_root)
    records = []
    expected = None
    for root in args.run:
        manifest = json.loads((root / "manifest.json").read_text())
        scores = pd.read_parquet(root / "run/scores.parquet")
        calibration = pd.read_parquet(root / "run/calibration_scores.parquet")
        ids = {
            "fit": manifest["fit_ids"],
            "calibration": manifest["calibration_ids"],
            "test": manifest["test_ids"],
        }
        expected_ids = {
            "fit": sorted(prepared.lists.fit),
            "calibration": sorted(prepared.lists.calibration),
            "test": sorted(prepared.lists.main_test),
        }
        if ids != expected_ids or manifest.get("lists_hash") != prepared.lists.lists_hash:
            raise ValueError(f"Manifest membership/hash mismatch: {root}")
        for name, expected_digest in manifest.get("prepared_artifacts", {}).items():
            artifact = args.prepared_root / "sn" / name
            if not artifact.is_file() or _digest(artifact) != expected_digest:
                raise ValueError(f"Prepared artifact digest mismatch: {root}/{name}")
        if expected is None:
            expected = ids
        if ids != expected:
            raise ValueError(f"Frozen ID mismatch: {root}")
        if (
            scores.sample_id.duplicated().any()
            or not np.isfinite(scores.score.to_numpy(float)).all()
        ):
            raise ValueError(f"Invalid score rows: {root}")
        if (
            calibration.sample_id.duplicated().any()
            or not np.isfinite(calibration.score.to_numpy(float)).all()
        ):
            raise ValueError(f"Invalid calibration rows: {root}")
        if set(scores.sample_id) != set(expected["test"]):
            raise ValueError(f"Test score coverage mismatch: {root}")
        if set(calibration.sample_id) != set(expected["calibration"]):
            raise ValueError(f"Calibration score coverage mismatch: {root}")
        metrics = _metric(scores, prepared)
        published = json.loads((root / "report.json").read_text())["proxy_metrics"]
        if metrics["auroc"] is None or published["auroc"] is None:
            if metrics["auroc"] != published["auroc"] or metrics["ap"] != published["ap"]:
                raise ValueError(f"Published metric mismatch: {root}")
        elif (
            abs(metrics["auroc"] - published["auroc"]) > 1e-12
            or abs(metrics["ap"] - published["ap"]) > 1e-12
        ):
            raise ValueError(f"Published metric mismatch: {root}")
        records.append({"run": str(root), "seed": manifest["seed"], "proxy_metrics": metrics})
    result = {
        "protocol": "sn_prior_gru_role_proxy_v1",
        "runs": records,
        "frozen_ids_match": True,
        "source_label_status": "all Unknown",
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
