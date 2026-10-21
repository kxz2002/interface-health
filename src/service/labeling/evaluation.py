"""Predeclared evaluation membership independent of scores and input validity."""

from .evidence import KEY_COLUMNS
from .rules import IDENTITY_COLUMNS


def evaluation_table(labels, splits, protocol):
    frame = labels.merge(splits[KEY_COLUMNS + ["split"]], on=KEY_COLUMNS, validate="one_to_one")
    eligible = frame.split.eq("test") & (
        frame.label.eq(1).fillna(False)
        | (frame.label.eq(0).fillna(False) & frame.label_source.eq("normal_run_reference"))
    )
    result = frame[IDENTITY_COLUMNS].copy()
    result["evaluation_eligible"] = eligible.astype(bool)
    result["evaluation_reason"] = "excluded_fault_reference"
    result.loc[frame.label.isna(), "evaluation_reason"] = "unknown"
    result.loc[~frame.split.eq("test"), "evaluation_reason"] = "not_test"
    result.loc[eligible, "evaluation_reason"] = "known_test_positive_or_normal_reference"
    result["evaluation_version"] = protocol["evaluation_version"]
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
