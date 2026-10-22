"""Strict calibration and separate fixed main, diagnostic and unknown scopes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.benchmark_downstream import safe_auc, safe_auprc
from src.service.labeling.acceptance import join_evaluation_scores

from .data import validate_sample_lists

QUANTILES = (0.95, 0.975, 0.99, 0.995)


def annotate_calibration(scores, prepared, expected_ids: set[str]) -> pd.DataFrame:
    _scores(scores, expected_ids)
    if set(scores.sample_id) != expected_ids:
        raise ValueError("Calibration annotation requires exactly the frozen IDs")
    result = _declare(scores, prepared.versions)
    for frame, column in [(prepared.labels, "label"), (prepared.splits, "split")]:
        if frame.sample_id.duplicated().any() or not expected_ids <= set(frame.sample_id):
            raise ValueError("Invalid calibration metadata IDs")
        metadata = frame.set_index("sample_id")[column].reindex(result.sample_id)
        if column in result:
            actual = result[column].to_numpy()
            if not np.array_equal(actual, metadata.to_numpy()):
                raise ValueError(f"Conflicting calibration {column}")
        result[column] = metadata.to_numpy()
    if not result.label.eq(0).fillna(False).all() or not result.split.eq("calibration").all():
        raise ValueError("Calibration annotation requires known-normal calibration IDs")
    return result


def _scores(scores, expected_ids=None):
    if (
        not {"sample_id", "score"} <= set(scores)
        or scores.sample_id.isna().any()
        or scores.sample_id.duplicated().any()
    ):
        raise ValueError("Missing/duplicate score sample IDs")
    if not np.isfinite(scores.score.to_numpy(float)).all():
        raise ValueError("Nonfinite scores")
    if expected_ids is not None and not set(expected_ids) <= set(scores.sample_id):
        raise ValueError("Missing required scores in fixed scope")


def calibrate(scores: pd.DataFrame, expected_ids: set[str], quantiles: list[float]) -> dict:
    _scores(scores, expected_ids)
    if (
        not isinstance(expected_ids, set)
        or not expected_ids
        or set(scores.sample_id) != expected_ids
    ):
        raise ValueError("Calibration scores must exactly match nonempty expected IDs")
    if (
        not {"label", "split"} <= set(scores)
        or not scores.label.eq(0).fillna(False).all()
        or not scores.split.eq("calibration").all()
    ):
        raise ValueError("Calibration requires independent known-normal calibration scores")
    if tuple(quantiles) != QUANTILES:
        raise ValueError("Calibration requires all four predeclared quantiles")
    return dict(
        thresholds={
            str(quantile): float(
                np.quantile(scores.score.to_numpy(float), quantile, method="linear")
            )
            for quantile in quantiles
        },
        method="linear",
        comparison="strict_greater",
        primary_quantile=0.99,
        sample_ids=sorted(expected_ids),
        count=len(expected_ids),
    )


def _declare(frame, versions):
    frame = frame.copy()
    nested = frame.attrs.get("versions", {})
    for name, value in versions.items():
        if (
            (name in frame.attrs and frame.attrs[name] != value)
            or (name in nested and nested[name] != value)
            or (name in frame and not frame[name].eq(value).all())
        ):
            raise ValueError(f"Conflicting {name} declaration")
        frame.attrs[name] = value
    return frame


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _classification(frame, threshold):
    positive = frame.label.eq(1).fillna(False).to_numpy(bool)
    negative = frame.label.eq(0).fillna(False).to_numpy(bool)
    alert = frame.score.to_numpy(float) > threshold
    tp, fp = int((positive & alert).sum()), int((negative & alert).sum())
    fn, tn = int((positive & ~alert).sum()), int((negative & ~alert).sum())
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        positive_count=tp + fn,
        negative_count=fp + tn,
        precision=_ratio(tp, tp + fp),
        recall=_ratio(tp, tp + fn),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
        normal_alert_rate=_ratio(fp, fp + tn),
        alert_count=int(alert.sum()),
        alert_rate=_ratio(int(alert.sum()), len(frame)),
    )


def _summary(frame, thresholds, grouped=False):
    known = frame[frame.label.notna()]
    if not known.label.isin([0, 1]).all():
        raise ValueError("Invalid evaluation labels")
    labels = known.label.to_numpy(int)
    scores = known.score.to_numpy(float)
    count = len(np.unique(labels))
    result = dict(
        sample_ids=sorted(frame.sample_id),
        rows=len(frame),
        known_rows=len(known),
        auroc=safe_auc(labels, scores),
        ap=safe_auprc(labels, scores),
        auc_reason="empty" if not count else "single_class" if count == 1 else None,
        classification={
            name: _classification(frame, threshold) for name, threshold in thresholds.items()
        },
        score_distribution=(
            {
                str(quantile): float(np.quantile(frame.score, quantile))
                for quantile in [0, 0.25, 0.5, 0.75, 1]
            }
            if len(frame)
            else None
        ),
    )
    if grouped:
        for column, key in [
            ("case_id", "by_case"),
            ("canonical_service_id", "by_service"),
            ("label_source", "by_source"),
        ]:
            result[key] = {
                str(name): _summary(part, thresholds)
                for name, part in frame.groupby(column, dropna=False)
            }
        cases = result["by_case"].values()
        for metric in ["auroc", "ap"]:
            eligible = [case[metric] for case in cases if case[metric] is not None]
            result[f"case_macro_{metric}"] = float(np.mean(eligible)) if eligible else None
        result["case_macro_reason"] = (
            "no_two_class_cases" if not any(case["auroc"] is not None for case in cases) else None
        )
    return result


def _coverage(frame, expected, scored):
    result = dict(
        expected=len(expected), scored=len(expected & scored), missing_ids=sorted(expected - scored)
    )
    for column, key in [
        ("case_id", "by_case"),
        ("canonical_service_id", "by_service"),
        ("label_source", "by_source"),
    ]:
        result[key] = {
            str(name): dict(
                total=len(part),
                expected=int(part.sample_id.isin(expected).sum()),
                scored=int(part.sample_id.isin(expected & scored).sum()),
            )
            for name, part in frame.groupby(column, dropna=False)
        }
    return result


def evaluate_run(
    tables,
    lists,
    scores: pd.DataFrame,
    thresholds: dict,
    *,
    score_scope="extended_test",
    source_binding=None,
) -> dict:
    if score_scope not in {"test", "extended_test"}:
        raise ValueError("Evaluation requires test or extended_test scores")
    lists_hash = validate_sample_lists(lists)
    if source_binding is not None:
        original = _declare(scores, source_binding["versions"])
        if (
            "lists_hash" not in original
            or not original.lists_hash.eq(source_binding["lists_hash"]).all()
        ):
            raise ValueError("Original score source binding mismatch")
        scores = original.copy()
        scores.attrs = {}
        for name in [*source_binding["versions"], "lists_hash", "label", "label_source", "split"]:
            if name in scores:
                scores = scores.rename(columns={name: f"source_{name}"})
        scores["lists_hash"] = lists_hash
    if "lists_hash" not in scores or not scores.lists_hash.eq(lists_hash).all():
        raise ValueError("Score lists_hash differs from fixed evaluation lists")
    if scores.attrs.get("lists_hash", lists_hash) != lists_hash:
        raise ValueError("Score lists_hash declaration mismatch")
    if (
        "model_version" in scores.attrs
        and "model_version" in scores
        and not scores.model_version.eq(scores.attrs["model_version"]).all()
    ):
        raise ValueError("Score model_version declaration mismatch")
    versions = tables.versions
    required_versions = {"label_version", "split_version", "evaluation_version"}
    if not required_versions <= set(versions) or not getattr(tables, "protocol", None):
        raise ValueError("Evaluation requires persisted input versions and protocol")
    versions = {name: versions[name] for name in required_versions}
    if tables.protocol.get("evaluation_version") != versions["evaluation_version"]:
        raise ValueError("Protocol evaluation version mismatch")
    scores = _declare(scores, versions)
    labels, splits = _declare(tables.labels, versions), _declare(tables.splits, versions)
    _scores(scores)
    for column in [
        "model_version",
        "lists_hash",
        "config_sha256",
        "checkpoint_sha256",
        "baseline_sha256",
    ]:
        if column in scores and (scores[column].isna().any() or scores[column].nunique() != 1):
            raise ValueError(f"Conflicting score {column}")
    if set(scores.sample_id) - set(labels.sample_id):
        raise ValueError("Scores contain unknown grid sample IDs")
    for column in [
        "dataset_id",
        "run_id",
        "case_id",
        "canonical_service_id",
        "window_start_ms",
        "window_end_ms",
        "label",
        "label_source",
    ]:
        if column in labels and column in scores:
            compared = labels[["sample_id", column]].merge(
                scores[["sample_id", column]],
                on="sample_id",
                validate="one_to_one",
                suffixes=("_input", "_score"),
            )
            left, right = compared[f"{column}_input"], compared[f"{column}_score"]
            if not (left.eq(right).fillna(False) | (left.isna() & right.isna())).all():
                raise ValueError(f"Conflicting score identity/label {column}")
    values = thresholds.get("thresholds")
    if (
        not isinstance(values, dict)
        or not values
        or any(not np.isfinite(value) for value in values.values())
    ):
        raise ValueError("Invalid calibrated thresholds")
    main_ids = set(lists.main_test)
    extended_ids = set(lists.extended_main_test)
    inference_ids = set(lists.extended_inference)
    diagnostic_ids = set(lists.diagnostic_pre) | set(lists.diagnostic_positive)
    grouped_pre, grouped_positive = set(), set()
    for group in lists.diagnostic_groups.values():
        if group["reference_available"] != bool(group["pre"]):
            raise ValueError("Diagnostic reference availability conflicts with fixed IDs")
        grouped_pre.update(group["pre"])
        grouped_positive.update(group["positive"])
        diagnostic_ids.update(group["pre"])
        diagnostic_ids.update(group["positive"])
    if grouped_pre != set(lists.diagnostic_pre) or grouped_positive != set(
        lists.diagnostic_positive
    ):
        raise ValueError("Diagnostic groups differ from fixed diagnostic lists")
    required = main_ids | diagnostic_ids
    if score_scope == "extended_test":
        required |= extended_ids | inference_ids
    _scores(scores, required)
    data = dict(
        labels=labels, splits=splits, evaluation=tables.evaluation, protocol=tables.protocol
    )
    main = join_evaluation_scores(data, scores, main_ids)
    if set(main.sample_id) != main_ids:
        raise ValueError("Main membership differs from fixed main_test list")
    extended = None
    if score_scope == "extended_test":
        extended = join_evaluation_scores(data, scores, extended_ids)
        if set(extended.sample_id) != extended_ids:
            raise ValueError("Extended main membership differs from fixed list")
    from src.service.labeling.acceptance import join_scores

    known_test = labels.loc[labels.label.notna() & labels.sample_id.isin(required), "sample_id"]
    join_scores(labels, splits, scores, set(known_test))
    merged = labels.merge(
        splits[["sample_id", "split"]], on="sample_id", validate="one_to_one"
    ).merge(scores[["sample_id", "score"]], on="sample_id", validate="one_to_one")
    diagnostics = {}
    for name, group in lists.diagnostic_groups.items():
        pre = merged[merged.sample_id.isin(group["pre"])]
        positive = merged[merged.sample_id.isin(group["positive"])]
        if (
            not pre.label.eq(0).fillna(False).all()
            or not pre.label_source.eq("pre_intervention_reference").all()
            or not positive.label.eq(1).fillna(False).all()
        ):
            raise ValueError("Invalid fixed diagnostic label scope")
        combined = pd.concat([pre, positive])
        if combined.case_id.nunique() > 1 or combined.canonical_service_id.nunique() > 1:
            raise ValueError("Diagnostic reference crosses case/service boundary")
        diagnostics[name] = dict(
            status="available" if len(pre) else "unavailable",
            reason=None if len(pre) else "missing_pre_reference",
            pre=_summary(pre, values),
            positive=_summary(positive, values),
            trajectory=combined.sort_values("window_start_ms")[
                ["sample_id", "window_start_ms", "score", "label", "label_source"]
            ].to_dict("records"),
        )
    unknown = merged[merged.sample_id.isin(inference_ids) & merged.label.isna()]
    coverage = _coverage(labels, required, set(scores.sample_id))
    coverage["main"] = _coverage(labels, main_ids, set(scores.sample_id))
    coverage["extended"] = _coverage(labels, extended_ids, set(scores.sample_id))
    coverage["diagnostic"] = _coverage(labels, diagnostic_ids, set(scores.sample_id))
    coverage["inference"] = _coverage(labels, inference_ids, set(scores.sample_id))
    return dict(
        main=_summary(main, values, True),
        extended=(
            _summary(extended, values, True)
            if extended is not None
            else dict(status="not_requested", reason="score_scope_test")
        ),
        within_case_diagnostic=diagnostics,
        unknown=_summary(unknown, values),
        scores=scores.sort_values("sample_id").to_dict("records"),
        coverage=coverage,
        thresholds=thresholds,
        versions=versions,
        score_scope=score_scope,
        score_source_binding=source_binding,
    )
