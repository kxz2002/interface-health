"""Predeclared evaluation membership independent of scores and input validity."""

from __future__ import annotations

import pandas as pd

from .evidence import KEY_COLUMNS
from .rules import IDENTITY_COLUMNS


def _evenly_spaced(rows: pd.DataFrame, count: int) -> set[str]:
    """Select a temporal spread without using scores or pseudo-random state."""
    if count < 0 or count > len(rows):
        raise ValueError("Invalid deterministic selection count")
    if count == 0:
        return set()
    ordered = rows.sort_values(
        ["window_start_ms", "canonical_service_id", "sample_id"]
    ).reset_index(drop=True)
    if count == len(ordered):
        return set(ordered.sample_id)
    if count == 1:
        return {str(ordered.iloc[0].sample_id)}
    positions = [int(round(i * (len(ordered) - 1) / (count - 1))) for i in range(count)]
    return set(ordered.iloc[positions].sample_id)


def _selected_ids(frame: pd.DataFrame, selection: dict) -> tuple[set[str], dict]:
    """Build the v1 per-case balanced test set from immutable labels."""
    if selection.get("mode") not in {"per_case_balanced_v1", "per_case_balanced_v2"}:
        raise ValueError("Unsupported evaluation selection mode")
    excluded = set(selection.get("excluded_cases", []))
    protocol_excluded = set(selection.get("protocol_excluded_cases", excluded))
    if excluded != protocol_excluded:
        raise ValueError("Selection and protocol excluded cases differ")
    normal_case = selection.get("normal_case_id")
    if not isinstance(normal_case, str) or not normal_case:
        raise ValueError("per-case split requires one normal_case_id")
    if normal_case in excluded:
        raise ValueError("The independent normal case cannot be excluded")
    test = frame[frame.split.eq("test") & ~frame.case_id.isin(excluded)].copy()
    positive = test[test.label.eq(1)]
    pre = test[test.label.eq(0) & test.label_source.eq("pre_intervention_reference")]
    selected: set[str] = set()
    per_case = {}
    for case_id in sorted(set(positive.case_id) | set(pre.case_id)):
        if case_id == normal_case:
            continue
        pos = positive[positive.case_id.eq(case_id)]
        neg = pre[pre.case_id.eq(case_id)]
        count = min(len(pos), len(neg))
        if count:
            pos_ids = _evenly_spaced(pos, count)
            neg_ids = _evenly_spaced(neg, count)
            selected.update(pos_ids | neg_ids)
        per_case[case_id] = {
            "positive_eligible": len(pos),
            "negative_eligible": len(neg),
            "selected_positive": count,
            "selected_negative": count,
            "excluded_reason": None if count else "missing_one_class",
        }
    fault_selected = selected.copy()
    frozen_fault_ids = selection.get("frozen_fault_sample_ids")
    if frozen_fault_ids is not None:
        frozen_fault_ids = set(frozen_fault_ids)
        if fault_selected != frozen_fault_ids:
            raise ValueError("Frozen fault selection differs from deterministic selection")
    normal = test[
        test.case_id.eq(normal_case)
        & test.label.eq(0)
        & test.label_source.eq("normal_run_reference")
    ]
    usable_cases = [item for item in per_case.values() if item["selected_negative"]]
    if not usable_cases:
        raise ValueError("No two-class fault case is available for per-case split")
    quota_mode = selection.get("normal_case_quota", "mean_fault_case_matched_negative")
    if quota_mode == "mean_fault_case_matched_negative":
        quota = int(
            round(sum(item["selected_negative"] for item in usable_cases) / len(usable_cases))
        )
    elif type(quota_mode) is int and quota_mode > 0:
        quota = quota_mode
    else:
        raise ValueError("Unsupported normal-case quota")
    if not len(normal):
        raise ValueError("The independent normal case has no eligible reference windows")
    if len(normal) < quota:
        raise ValueError("Independent normal case has fewer windows than its frozen quota")
    normal_ids = _evenly_spaced(normal, quota)
    selected.update(normal_ids)
    per_case[normal_case] = {
        "positive_eligible": 0,
        "negative_eligible": len(normal),
        "selected_positive": 0,
        "selected_negative": len(normal_ids),
        "excluded_reason": None if normal_ids else "missing_normal_reference",
    }
    stats = {
        "normal_case_quota": quota,
        "normal_case_selected": len(normal_ids),
        "fault_case_count": len(usable_cases),
        "selected_positive": int(sum(item["selected_positive"] for item in per_case.values())),
        "selected_negative": int(sum(item["selected_negative"] for item in per_case.values())),
        "by_case": per_case,
    }
    return selected, stats


def evaluation_table(labels, splits, protocol):
    frame = labels.merge(splits[KEY_COLUMNS + ["split"]], on=KEY_COLUMNS, validate="one_to_one")
    excluded_cases = protocol.get("evaluation_excluded_cases", [])
    if (
        not isinstance(excluded_cases, list)
        or any(not isinstance(case_id, str) or not case_id for case_id in excluded_cases)
        or len(set(excluded_cases)) != len(excluded_cases)
    ):
        raise ValueError("evaluation_excluded_cases must be a unique list of nonempty case IDs")
    excluded = frame.case_id.isin(set(excluded_cases))
    eligible = (
        frame.split.eq("test")
        & ~excluded
        & (
            frame.label.eq(1).fillna(False)
            | (frame.label.eq(0).fillna(False) & frame.label_source.eq("normal_run_reference"))
        )
    )
    selection = protocol.get("evaluation_selection")
    selection_stats = None
    if selection is not None:
        selected, selection_stats = _selected_ids(frame, selection)
        eligible = frame.sample_id.isin(selected)
    result = frame[IDENTITY_COLUMNS].copy()
    result["evaluation_eligible"] = eligible.astype(bool)
    result["evaluation_reason"] = "excluded_fault_reference"
    result.loc[excluded, "evaluation_reason"] = "excluded_evaluation_case"
    result.loc[frame.label.isna(), "evaluation_reason"] = "unknown"
    result.loc[~frame.split.eq("test"), "evaluation_reason"] = "not_test"
    result.loc[eligible, "evaluation_reason"] = "known_test_positive_or_normal_reference"
    result["evaluation_version"] = protocol["evaluation_version"]
    if selection_stats is not None:
        result.attrs["selection_stats"] = selection_stats
    return result


def evaluation_counts(evaluation, labels, grid):
    frame = evaluation.merge(
        labels[KEY_COLUMNS + ["label", "label_source"]], on=KEY_COLUMNS, validate="one_to_one"
    )
    validity = [name for name in grid if name.endswith("_valid")]
    frame = frame.merge(grid[KEY_COLUMNS + validity], on=KEY_COLUMNS, validate="one_to_one")
    selected = frame[frame.evaluation_eligible]

    def counts(part):
        return dict(
            rows=len(part),
            negative=int(part.label.eq(0).sum()),
            positive=int(part.label.eq(1).sum()),
        )

    return dict(
        total=counts(selected),
        by_source={name: counts(part) for name, part in selected.groupby("label_source")},
        by_case={name: counts(part) for name, part in selected.groupby("case_id")},
        input_valid={name: counts(selected[selected[name].eq(True)]) for name in validity},
        exclusion_reasons=frame.loc[~frame.evaluation_eligible, "evaluation_reason"]
        .value_counts()
        .to_dict(),
    )
