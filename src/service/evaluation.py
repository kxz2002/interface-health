"""Evidence-separated state diagnostics and fixed controlled-target retrieval."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_downstream import safe_auc, safe_auprc
from src.service.protocol import ServiceProtocol


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    known = frame.label.notna()
    finite = np.isfinite(frame.anomaly_score)
    valid = known & finite & frame.score_status.eq("Scored")
    y = frame.loc[valid, "label"].to_numpy(dtype=int)
    s = frame.loc[valid, "anomaly_score"].to_numpy(dtype=float)
    return dict(
        n_expected=len(frame),
        n_known=int(known.sum()),
        n_unknown=int((~known).sum()),
        n_unscored=int((known & ~valid).sum()),
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
        auroc=safe_auc(y, s),
        auprc=safe_auprc(y, s),
    )


def evaluate_s1(
    scores: pd.DataFrame, evidence: pd.DataFrame, index: pd.DataFrame
) -> dict[str, object]:
    """Resource WINDOW state only; instance/execution/availability are other tasks."""
    if index.sample_id.duplicated().any():
        raise ValueError("duplicate index sample IDs")
    conflicts = []
    admitted = []
    for row in evidence.to_dict("records"):
        reasons = []
        if row["unit"] != "WINDOW":
            reasons.append("unit_not_service_window")
        if row["state_class"] not in ("RESOURCE_CHANGE", "NO_OBSERVED_CHANGE"):
            reasons.append("state_not_resource")
        if row["evidence_grade"] != "INDEPENDENT_STATE":
            reasons.append("not_independent_state")
        if row["review_status"] != "confirmed":
            reasons.append("not_confirmed")
        if not row["heldout_from_model"]:
            reasons.append("not_heldout")
        if len(row["depends_on_input_sources"]):
            reasons.append("input_source_overlap")
        if pd.isna(row["independent_label"]):
            reasons.append("unknown_label")
        if (
            "KILL" in str(row.get("fact_type", "")).upper()
            or "TERMINATION" in str(row.get("fact_type", "")).upper()
        ):
            reasons.append("instance_termination_not_resource_gold")
        if reasons:
            conflicts.append(dict(evidence_id=row["evidence_id"], reasons=";".join(reasons)))
        else:
            admitted.append(row)
    independent = []
    for (case, service, group, method), frame in scores.groupby(
        ["case_id", "canonical_service_id", "feature_group", "method"]
    ):
        matching = [
            r for r in admitted if r["case_id"] == case and r["canonical_service_id"] == service
        ]
        for state in sorted({r["state_class"] for r in matching}):
            f = frame.copy()
            labels = []
            for sample in f.itertuples():
                if sample.partition in ("normal_fit", "normal_calibration"):
                    labels.append(None)
                    continue
                found = {
                    int(r["independent_label"])
                    for r in matching
                    if r["state_class"] == state
                    and r["start_ms"] <= sample.window_start_ms
                    and r["end_ms"] >= sample.window_end_ms
                }
                labels.append(next(iter(found)) if len(found) == 1 else None)
                if len(found) > 1:
                    conflicts.append(
                        dict(evidence_id=sample.sample_id, reasons="conflicting_window_labels")
                    )
            f["label"] = labels
            independent.append(
                dict(
                    case_id=case,
                    canonical_service_id=service,
                    state_class=state,
                    feature_group=group,
                    method=method,
                    **_metrics(f),
                )
            )
    proxy = scores.merge(
        index[["sample_id", "proxy_exposure_label"]], on="sample_id", validate="many_to_one"
    )
    proxy_rows = []
    for keys, frame in proxy.groupby(
        ["case_id", "canonical_service_id", "feature_group", "method"]
    ):
        if frame.proxy_exposure_label.notna().any():
            frame = frame.rename(columns={"proxy_exposure_label": "label"})
            proxy_rows.append(
                dict(
                    zip(["case_id", "canonical_service_id", "feature_group", "method"], keys),
                    **_metrics(frame),
                )
            )
    n_known = sum(r["n_known"] for r in independent)
    return dict(
        independent_s1=dict(
            n_known=n_known,
            auroc=None,
            auprc=None,
            rows=independent,
            reason=(
                "No matching independent resource WINDOW labels"
                if not n_known
                else "Metrics stratified by case/service/resource state; no pooled mixed-state metric"
            ),
        ),
        s1_proxy_controlled_exposure=dict(
            event_count=len({r["case_id"] for r in proxy_rows}),
            rows=proxy_rows,
            note="Controlled exposure association only; baseline is unexposed, not confirmed healthy; not independent state gold",
        ),
        evidence_conflicts=conflicts,
    )


def aggregate_events(
    scores: pd.DataFrame, events: pd.DataFrame, candidates: list[str], protocol: ServiceProtocol
) -> pd.DataFrame:
    if len(set(candidates)) != len(candidates) or events.event_id.duplicated().any():
        raise ValueError("duplicate candidates/events")
    rows = []
    width = protocol.config["window_ms"]
    for event in events.itertuples():
        start = math.ceil(event.evaluation_start_ms / width) * width
        expected = max(0, (event.evaluation_end_ms - start) // width)
        for service in candidates:
            for group in protocol.config["feature_groups"]:
                for method in protocol.config["methods"]:
                    f = scores[
                        (scores.case_id == event.case_id)
                        & (scores.canonical_service_id == service)
                        & (scores.feature_group == group)
                        & (scores.method == method)
                        & (scores.window_start_ms >= start)
                        & (scores.window_end_ms <= event.evaluation_end_ms)
                    ]
                    if f.sample_id.duplicated().any() or len(f) > expected:
                        raise ValueError("duplicate or excessive event windows")
                    valid = f[
                        f.score_status.eq("Scored")
                        & np.isfinite(f.calibrated_score)
                        & np.isfinite(f.anomaly_score)
                    ]
                    n = len(valid)
                    coverage = n / expected if expected else 0.0
                    k = math.ceil(protocol.config["event_top_fraction"] * n)
                    ok = (
                        expected > 0
                        and coverage >= protocol.config["event_min_coverage"]
                        and not f.score_status.eq("Untrained").any()
                    )
                    reasons = sorted({str(r) for values in f.quality_reasons for r in values})
                    if len(f) < expected:
                        reasons.append("missing_score_rows")
                    if coverage < protocol.config["event_min_coverage"]:
                        reasons.append("insufficient_coverage")
                    if f.score_status.eq("Untrained").any():
                        reasons.append("Untrained")
                    rows.append(
                        dict(
                            event_id=event.event_id,
                            case_id=event.case_id,
                            canonical_service_id=service,
                            feature_group=group,
                            method=method,
                            n_expected=expected,
                            n_valid=n,
                            coverage=coverage,
                            top_k=k,
                            event_score=(
                                float(valid.calibrated_score.nlargest(k).mean()) if ok else np.nan
                            ),
                            raw_median=float(valid.anomaly_score.median()) if n else np.nan,
                            raw_top20_mean=(
                                float(valid.anomaly_score.nlargest(k).mean()) if n else np.nan
                            ),
                            status="Scored" if ok else "Unscorable",
                            unknown_reasons=";".join(reasons),
                        )
                    )
    return pd.DataFrame(rows)


def evaluate_s2(
    event_scores: pd.DataFrame, events: pd.DataFrame, protocol: ServiceProtocol
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    arms = (
        event_scores[["feature_group", "method"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for group, method in arms:
        for event in events.itertuples():
            f = event_scores[
                (event_scores.event_id == event.event_id)
                & (event_scores.feature_group == group)
                & (event_scores.method == method)
            ]
            target = f[f.canonical_service_id == event.configured_target_service]
            if len(target) != 1:
                raise ValueError("event requires exactly one target candidate")
            t = target.iloc[0]
            others = f[f.canonical_service_id != event.configured_target_service]
            known = others[others.status.eq("Scored") & np.isfinite(others.event_score)]
            missing = len(others) - len(known)
            ok = t.status == "Scored" and np.isfinite(t.event_score)
            if ok:
                delta = known.event_score - t.event_score
                best = 1 + int((delta > 1e-12).sum())
                worst = best + int((delta.abs() <= 1e-12).sum()) + missing
            else:
                best = worst = None
            rows.append(
                dict(
                    event_id=event.event_id,
                    feature_group=group,
                    method=method,
                    target_service=event.configured_target_service,
                    target_status="Scored" if ok else "Unscorable",
                    target_score=float(t.event_score) if ok else None,
                    rank_best=best,
                    rank_worst=worst,
                    unknown_count=missing,
                    n_candidates=len(f),
                    hit_at_1=int(ok and worst <= 1),
                    hit_at_3=int(ok and worst <= 3),
                    rr=1 / worst if ok else 0.0,
                    target_coverage=float(t.get("coverage", 0)),
                )
            )
    table = pd.DataFrame(rows)
    summaries = []
    for (group, method), f in table.groupby(["feature_group", "method"]):
        valid = f[f.target_status == "Scored"]
        summaries.append(
            dict(
                feature_group=group,
                method=method,
                denominator=len(events),
                hit_at_1=float(f.hit_at_1.mean()),
                hit_at_1_numerator=int(f.hit_at_1.sum()),
                hit_at_3=float(f.hit_at_3.mean()),
                hit_at_3_numerator=int(f.hit_at_3.sum()),
                mrr=float(f.rr.mean()),
                rr_sum=float(f.rr.sum()),
                scorable_events=len(valid),
                scorable_subset_mrr=float(valid.rr.mean()) if len(valid) else None,
            )
        )
    return table, dict(
        task="S2 controlled injection target retrieval",
        arms=summaries,
        note="All events retained; ties and unknown competitors conservative; not full RCA",
    )


def render_report(data_dir: Path, run_dir: Path, evaluation_dir: Path, out: Path) -> None:
    raise NotImplementedError("Report rendering belongs to T6")
