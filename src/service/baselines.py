"""Frozen Normal-only Service references, using benchmark pure functions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.benchmark_downstream import (
    apply_normalizer,
    fit_iforest,
    fit_normalizer,
    fit_ocsvm,
    score_iforest,
    score_ocsvm,
)
from src.service.protocol import (
    PROTOCOL_ID,
    SCORE_COLUMNS,
    ServiceProtocol,
    make_grid,
    make_indices,
    validate_scores,
)


@dataclass
class BaselineBundle:
    protocol_id: str
    models: dict[tuple[str, str, str], object]
    stats: dict[str, object]
    references: dict[tuple[str, str], dict] = field(default_factory=dict)


def _aligned(features, index, protocol):
    """Reconstruct frozen identities/partitions; never trust externally supplied fit flags."""
    if features.sample_id.duplicated().any() or index.sample_id.duplicated().any():
        raise ValueError("duplicate sample_id")
    expected = (
        pd.concat(
            [make_grid(c, protocol.config["services"]) for c in protocol.config["cases"]],
            ignore_index=True,
        )
        .set_index("sample_id")
        .sort_index()
    )
    f = features.set_index("sample_id").sort_index()
    i = index.set_index("sample_id").sort_index()
    if not f.index.equals(expected.index) or not i.index.equals(expected.index):
        raise ValueError("sample IDs must match full frozen grid")
    keys = ["case_id", "canonical_service_id", "window_start_ms", "window_end_ms"]
    for key in keys:
        if not f[key].equals(expected[key]) or not i[key].equals(expected[key]):
            raise ValueError(f"identity mismatch: {key}")
    derived = make_indices(features, protocol).set_index("sample_id").sort_index()
    if not i.partition.equals(derived.partition):
        raise ValueError("partition differs from frozen Normal/Fault temporal split")
    if not pd.api.types.is_bool_dtype(f.resource_valid) or not pd.api.types.is_bool_dtype(
        i.paired_resource_eligible
    ):
        raise ValueError("eligibility must be boolean")
    if not i.paired_resource_eligible.equals(f.resource_valid):
        raise ValueError("index eligibility differs from resource_valid")
    values = f[protocol.config["feature_groups"]["R3"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values[f.resource_valid.to_numpy()]).all():
        raise ValueError("eligible R3 rows must have finite values")
    return f, i


def calibrate_score(values: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    c = np.asarray(calibration, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if not len(c) or not np.isfinite(c).all() or not np.isfinite(values).all():
        raise ValueError("calibration requires finite nonempty reference and values")
    # searchsorted(left) counts ties as >=; do not break saturated ties.
    return np.searchsorted(np.sort(c), values, side="left") / (len(c) + 1.0)


def _raw(method, x, models):
    if method == "L2":
        return np.linalg.norm(x, axis=1)
    if method == "max":
        return np.max(np.abs(x), axis=1)
    if method == "OCSVM":
        return score_ocsvm(models[method], x)
    if method == "IF":
        return score_iforest(models[method], x)
    raise ValueError(f"unsupported method {method}")


def fit_baselines(features, index, protocol):
    f, i = _aligned(features, index, protocol)
    bundle = BaselineBundle(
        PROTOCOL_ID,
        {},
        {
            "feature_groups": protocol.config["feature_groups"],
            "seed": protocol.config["seed"],
            "references": [],
        },
    )
    for svc in protocol.config["services"]:
        eligible = f.canonical_service_id.eq(svc) & f.resource_valid
        fit_ids = f.index[eligible & i.partition.eq("normal_fit")].tolist()
        cal_ids = f.index[eligible & i.partition.eq("normal_calibration")].tolist()
        for group, cols in protocol.config["feature_groups"].items():
            ref = dict(
                columns=list(cols),
                fit_ids=fit_ids,
                calibration_ids=cal_ids,
                models={},
                calibration={},
                mean=None,
                std=None,
                status="Untrained",
                reason=None,
            )
            bundle.references[(svc, group)] = ref
            if (
                len(fit_ids) < protocol.config["min_fit_rows"]
                or len(cal_ids) < protocol.config["min_calibration_rows"]
            ):
                ref["reason"] = (
                    f"insufficient Normal reference: fit={len(fit_ids)}, calibration={len(cal_ids)}"
                )
                continue
            mean, std = fit_normalizer(f.loc[fit_ids, cols].to_numpy(dtype=np.float64))
            fit_x = apply_normalizer(f.loc[fit_ids, cols].to_numpy(dtype=np.float64), mean, std)
            cal_x = apply_normalizer(f.loc[cal_ids, cols].to_numpy(dtype=np.float64), mean, std)
            ref.update(
                mean=mean,
                std=std,
                status="Trained",
                constant_columns=[
                    c for c, sd in zip(cols, f.loc[fit_ids, cols].std(ddof=0)) if sd < 1e-9
                ],
            )
            try:
                ref["models"] = {"OCSVM": fit_ocsvm(fit_x, 42), "IF": fit_iforest(fit_x, 42)}
            except Exception as error:
                raise RuntimeError(
                    f"fit failed for service={svc}, group={group}: {error}"
                ) from error
            for method in protocol.config["methods"]:
                ref["calibration"][method] = _raw(method, cal_x, ref["models"])
                if method in ref["models"]:
                    bundle.models[(svc, group, method)] = ref["models"][method]
    return bundle


def score_baselines(features, index, bundle, protocol):
    f, i = _aligned(features, index, protocol)
    if (
        bundle.protocol_id != PROTOCOL_ID
        or bundle.stats["feature_groups"] != protocol.config["feature_groups"]
    ):
        raise ValueError("bundle protocol/feature order mismatch")
    frames = []
    for svc in protocol.config["services"]:
        rows = f[f.canonical_service_id.eq(svc)]
        for group in protocol.config["feature_groups"]:
            ref = bundle.references[(svc, group)]
            valid = rows.resource_valid.to_numpy()
            trained = ref["status"] == "Trained"
            x = (
                apply_normalizer(
                    rows.loc[valid, ref["columns"]].to_numpy(dtype=np.float64),
                    ref["mean"],
                    ref["std"],
                )
                if trained
                else None
            )
            for method in protocol.config["methods"]:
                raw = np.full(len(rows), np.nan)
                cal = raw.copy()
                if trained and valid.any():
                    raw[valid] = _raw(method, x, ref["models"])
                    cal[valid] = calibrate_score(raw[valid], ref["calibration"][method])
                result = rows[
                    [
                        "case_id",
                        "canonical_service_id",
                        "window_start_ms",
                        "window_end_ms",
                        "score_time_lower_bound_ms",
                    ]
                ].reset_index()
                result["partition"] = i.loc[rows.index, "partition"].to_numpy()
                result["feature_group"] = group
                result["method"] = method
                result["anomaly_score"] = raw
                result["calibrated_score"] = cal
                result["score_status"] = np.where(
                    valid, "Scored" if trained else "Untrained", "Unknown"
                )
                result["quality_reasons"] = [
                    list(reasons) + ([] if trained else [ref["reason"]])
                    for reasons in rows.quality_reasons
                ]
                result["protocol_id"] = PROTOCOL_ID
                result["model_id"] = f"{PROTOCOL_ID}::{svc}::{group}::{method}"
                frames.append(result[list(SCORE_COLUMNS)])
    scores = pd.concat(frames, ignore_index=True).sort_values(
        ["sample_id", "feature_group", "method"], ignore_index=True
    )
    validate_scores(scores)
    return scores


def save_bundle(bundle: BaselineBundle, out: Path) -> None:
    out = Path(out)
    if out.exists():
        raise ValueError(f"bundle output already exists: {out}")
    out.mkdir(parents=True)
    joblib.dump(bundle, out / "models.joblib")
    metadata, calibration, training = [], [], []
    for (service, group), ref in bundle.references.items():
        metadata.append(
            dict(
                service=service,
                group=group,
                columns=ref["columns"],
                status=ref["status"],
                reason=ref["reason"],
                mean=None if ref["mean"] is None else ref["mean"].tolist(),
                std=None if ref["std"] is None else ref["std"].tolist(),
                constant_columns=ref.get("constant_columns", []),
                params={m: model.get_params() for m, model in ref["models"].items()},
                effective_gamma=ref["models"]["OCSVM"]._gamma if ref["models"] else None,
            )
        )
        for partition, ids in [
            ("normal_fit", ref["fit_ids"]),
            ("normal_calibration", ref["calibration_ids"]),
        ]:
            training.extend(
                dict(
                    sample_id=s,
                    canonical_service_id=service,
                    feature_group=group,
                    partition=partition,
                )
                for s in ids
            )
        for method, values in ref["calibration"].items():
            calibration.extend(
                dict(
                    sample_id=s,
                    canonical_service_id=service,
                    feature_group=group,
                    method=method,
                    anomaly_score=float(v),
                    calibrated_score=float(c),
                )
                for s, v, c in zip(ref["calibration_ids"], values, calibrate_score(values, values))
            )
    (out / "preprocessing.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    pd.DataFrame(training).to_parquet(out / "training_ids.parquet", index=False)
    pd.DataFrame(calibration).to_parquet(out / "calibration_scores.parquet", index=False)


def load_bundle(out: Path) -> BaselineBundle:
    bundle = joblib.load(Path(out) / "models.joblib")
    if not isinstance(bundle, BaselineBundle) or bundle.protocol_id != PROTOCOL_ID:
        raise ValueError("invalid Service bundle")
    return bundle
