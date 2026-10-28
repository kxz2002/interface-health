#!/usr/bin/env python3
"""Build, independently check, or compare immutable TT A-label side tables."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.service.labeling.acceptance import check_bundle
from src.service.labeling.bundle import (
    _protect,
    _provenance,
    build_bundle,
    derive_evaluation_bundle,
    diff_bundles,
)
from src.service.preparation import file_digest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    arguments = {
        "build": ["tt-root", "preparation", "protocol", "output"],
        "derive-evaluation": ["source", "preparation", "protocol", "output"],
        "check": ["bundle", "preparation"],
        "diff": ["old", "new", "output"],
    }
    for command, names in arguments.items():
        sub = commands.add_parser(command)
        for name in names:
            sub.add_argument("--" + name, required=True, type=Path)
        if command == "check":
            sub.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command in {"build", "diff"}:
            output = args.output.resolve()
            if output == (Path(__file__).resolve().parents[1] / "outputs/service").resolve():
                raise ValueError("Output must be a new run, not outputs/service root")
        if args.command == "build":
            result = build_bundle(args.tt_root, args.preparation, args.protocol, args.output)
        elif args.command == "derive-evaluation":
            result = derive_evaluation_bundle(
                args.source, args.preparation, args.protocol, args.output
            )
        elif args.command == "check":
            receipt = None
            if args.receipt:
                receipt = _protect(args.receipt, [args.bundle, args.preparation])
            result = check_bundle(args.bundle, args.preparation)
            if receipt:
                manifest = args.bundle / "manifest.json"
                if not manifest.is_file():
                    manifest = args.bundle / "labels/manifest.json"
                record = {
                    "status": "success",
                    "bundle_manifest_sha256": file_digest(manifest),
                    "bundle": str(args.bundle.resolve()),
                    "preparation": str(args.preparation.resolve()),
                    "result": result,
                    "provenance": _provenance(),
                }
                receipt.parent.mkdir(parents=True, exist_ok=True)
                with receipt.open("x", encoding="utf-8") as handle:
                    json.dump(record, handle, sort_keys=True, allow_nan=False)
        else:
            result = diff_bundles(args.old, args.new, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"label_service_data: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
