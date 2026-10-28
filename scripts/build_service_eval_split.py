"""Create a versioned service evaluation bundle from a named split profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.service.labeling.bundle import derive_evaluation_bundle, read_bundle
from src.service.labeling.evaluation import _selected_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    profiles = json.loads(args.profiles.read_text())
    profile = profiles.get("profiles", {}).get(args.profile)
    if not isinstance(profile, dict) or profile.get("kind") not in {
        "per_case_balanced_v1",
        "per_case_balanced_v2",
    }:
        raise ValueError(f"Unknown or non-derived evaluation profile: {args.profile}")
    source_protocol = read_bundle(args.source)["protocol"]
    excluded = profile["excluded_cases"]
    protocol = dict(source_protocol)
    protocol.update(
        evaluation_version=profile["evaluation_version"],
        evaluation_excluded_cases=excluded,
        protocol_id=args.profile,
        evaluation_selection={
            "mode": profile["kind"],
            "normal_case_id": profile["normal_case_id"],
            "excluded_cases": excluded,
            "protocol_excluded_cases": excluded,
            "normal_case_quota": profile["normal_case_quota"],
            "selection_order": profile["selection_order"],
            "matching": profile["matching"],
            "normal_sampling": profile["normal_sampling"],
        },
    )
    if profile["kind"] == "per_case_balanced_v2":
        source_data = read_bundle(args.source)
        frame = source_data["labels"].merge(
            source_data["splits"][
                ["dataset_id", "run_id", "canonical_service_id", "window_start_ms", "split"]
            ],
            on=["dataset_id", "run_id", "canonical_service_id", "window_start_ms"],
            validate="one_to_one",
        )
        v1_selection = {
            "mode": "per_case_balanced_v1",
            "normal_case_id": profile["normal_case_id"],
            "excluded_cases": excluded,
            "protocol_excluded_cases": excluded,
            "normal_case_quota": "mean_fault_case_matched_negative",
        }
        selected_ids, _ = _selected_ids(frame, v1_selection)
        fault_ids = selected_ids - set(
            frame.loc[frame.case_id.eq(profile["normal_case_id"]), "sample_id"]
        )
        protocol["evaluation_selection"]["frozen_fault_sample_ids"] = sorted(fault_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output.parent / f"{args.output.name}.protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"Refusing to overwrite protocol: {protocol_path}")
    with protocol_path.open("x", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = derive_evaluation_bundle(args.source, args.preparation, protocol_path, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
