#!/usr/bin/env python3
"""Build, independently check, or compare immutable TT A-label side tables."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.service.labeling.acceptance import check_bundle
from src.service.labeling.bundle import build_bundle, derive_evaluation_bundle, diff_bundles


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
            result = check_bundle(args.bundle, args.preparation)
        else:
            result = diff_bundles(args.old, args.new, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"label_service_data: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
