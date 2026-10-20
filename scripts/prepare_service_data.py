"""Inventory both Service datasets without training or altering raw inputs."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.service.label_split_preparation import write_label_split_tables
from src.service.preparation import cached_sn_inventory, file_digest, selected_tt_inventory
from src.service.window_preparation import write_window_tables


def validate_output(output: Path, tt_root: Path) -> None:
    resolved = output.resolve()
    base = Path(__file__).resolve().parents[1] / "outputs/service"
    if resolved == base.resolve() or not resolved.is_relative_to(base.resolve()):
        raise ValueError("Output must be a fresh run under outputs/service/<run_id>")
    if resolved.is_relative_to(tt_root.resolve()):
        raise ValueError("Output must not be inside raw data")
    if output.exists():
        raise FileExistsError(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-root", type=Path, required=True)
    parser.add_argument("--sn-archive", type=Path, required=True)
    parser.add_argument(
        "--sn-inventory",
        type=Path,
        required=True,
        help="Prior source-verified inventory.json; avoids rescanning all SN members",
    )
    parser.add_argument("--sn-inventory-sha256", required=True)
    parser.add_argument("--sn-profile", choices=["strict", "exploratory"], default="strict")
    parser.add_argument(
        "--sn-profile-config",
        type=Path,
        help="Required explicit assumption config for exploratory profile",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.sn_profile == "exploratory") != (args.sn_profile_config is not None):
        parser.error("exploratory requires --sn-profile-config; strict does not accept it")
    profile = None
    if args.sn_profile_config is not None:
        from src.service.sn_exploratory import load_profile

        profile = load_profile(args.sn_profile_config)
    validate_output(args.output, args.tt_root)
    results = {
        "tt": selected_tt_inventory(args.tt_root),
        "sn": cached_sn_inventory(
            args.sn_inventory,
            args.sn_inventory_sha256,
            args.sn_archive,
            defer_archive_digest=args.sn_profile == "exploratory",
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inventory.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = write_window_tables(
        results,
        args.tt_root,
        args.output,
        sn_profile=args.sn_profile,
        sn_profile_config=profile,
        sn_archive=args.sn_archive,
    )
    summary.update(
        write_label_split_tables(
            results, args.output, sn_profile=args.sn_profile, tt_root=args.tt_root
        )
    )
    summary["python"] = platform.python_version()
    summary["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary["git_status"] = subprocess.check_output(["git", "status", "--short"], text=True)
    summary["resolved_config"] = {k: str(v) for k, v in vars(args).items()}
    summary["resolved_config"]["sn_assumption_profile"] = profile
    if args.sn_profile_config:
        summary["resolved_config"]["sn_profile_config_sha256"] = file_digest(args.sn_profile_config)
    summary["implementation_sha256"] = {
        str(path): file_digest(path)
        for path in [
            Path(__file__),
            Path("src/service/preparation.py"),
            Path("src/service/modality_preparation.py"),
            Path("src/service/window_preparation.py"),
            Path("src/service/sn_preparation.py"),
            Path("src/service/sn_exploratory.py"),
            Path("src/service/label_split_preparation.py"),
        ]
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
