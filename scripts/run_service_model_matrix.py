"""Run the frozen Task 10 matrix serially through the existing stage CLI."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)
REPOSITORY = Path(__file__).resolve().parents[1]
CLI = REPOSITORY / "scripts/service_model.py"
PROTOTYPE_METHODS = (
    "transformer_cls",
    "transformer_mean",
    "concat_svdd",
    "L2",
    "Max",
    "numeric_IF",
    "numeric_OCSVM",
    "trimodal_IF",
    "trimodal_OCSVM",
)
REPEATED_METHODS = (
    "transformer_cls",
    "transformer_mean",
    "concat_svdd",
    "numeric_IF",
    "trimodal_IF",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--tracking-uri", required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--existing-first-fit", type=Path, required=True)
    result.add_argument("--poll-seconds", type=float, default=30)
    result.add_argument("--wait-timeout-seconds", type=float, default=0)
    result.add_argument("--dry-run", action="store_true")
    return result


def members() -> tuple[tuple[str, int], ...]:
    return tuple((method, 42) for method in PROTOTYPE_METHODS) + tuple(
        (method, seed) for seed in (1, 2) for method in REPEATED_METHODS
    )


def stage_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    commands = []
    prefix = [sys.executable, "-u", str(CLI)]
    common = [
        "--config",
        str(args.config),
        "--prepared",
        str(args.prepared),
        "--tracking-uri",
        args.tracking_uri,
    ]
    for method, seed in members():
        name = f"{method}-seed{seed}"
        paths = {
            stage: args.output_root / f"{name}-{stage}"
            for stage in ("fit", "calibration", "test", "evaluate")
        }
        if (method, seed) == ("transformer_cls", 42):
            paths["fit"] = args.existing_first_fit
        else:
            commands.append(
                (
                    f"{name}/fit",
                    prefix
                    + ["fit"]
                    + common
                    + [
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        "--output",
                        str(paths["fit"]),
                    ],
                )
            )
        commands.append(
            (
                f"{name}/check-fit",
                prefix
                + [
                    "check-run",
                    "--run",
                    str(paths["fit"]),
                ],
            )
        )
        for scope in ("calibration", "test"):
            commands.append(
                (
                    f"{name}/score-{scope}",
                    prefix
                    + ["score"]
                    + common
                    + [
                        "--model-run",
                        str(paths["fit"]),
                        "--score-scope",
                        scope,
                        "--output",
                        str(paths[scope]),
                    ],
                )
            )
            commands.append(
                (
                    f"{name}/check-{scope}",
                    prefix
                    + [
                        "check-run",
                        "--run",
                        str(paths[scope]),
                    ],
                )
            )
        commands.append(
            (
                f"{name}/evaluate",
                prefix
                + ["evaluate"]
                + common
                + [
                    "--model-run",
                    str(paths["fit"]),
                    "--scores",
                    str(paths["test"]),
                    "--calibration-scores",
                    str(paths["calibration"]),
                    "--output",
                    str(paths["evaluate"]),
                ],
            )
        )
        commands.append(
            (
                f"{name}/check-evaluate",
                prefix
                + [
                    "check-run",
                    "--run",
                    str(paths["evaluate"]),
                ],
            )
        )
    return commands


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_existing_fit(args: argparse.Namespace) -> None:
    if str(REPOSITORY) not in sys.path:
        sys.path.insert(0, str(REPOSITORY))
    from src.service.modeling.config import content_hash, load_config

    expected = load_config(args.config)
    expected["model"]["fusion"] = "summary_token"
    expected["training"]["prototype_seed"] = 42
    expected.pop("config_sha256")
    expected["config_sha256"] = content_hash(expected)
    actual = read_json(args.existing_first_fit / "run/config.json")
    if any(actual.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing first fit differs from the frozen requested configuration")
    if actual.get("tracking_uri") != args.tracking_uri:
        raise ValueError("Existing first fit tracking URI mismatch")
    provenance = read_json(args.existing_first_fit / "run/provenance.json")
    if (
        provenance.get("stage") != "fit"
        or provenance.get("method") != "transformer_cls"
        or provenance.get("seed") != 42
        or Path(provenance["parents"]["prepared"]["path"]).resolve() != args.prepared
    ):
        raise ValueError("Existing first fit method, seed, stage or prepared parent mismatch")


def wait_for_first_fit(args: argparse.Namespace) -> None:
    started = time.monotonic()
    while True:
        status = read_json(args.existing_first_fit / "status.json")
        state = status.get("status")
        if state == "failed":
            raise RuntimeError(f"Existing first fit failed; artifacts retained: {status}")
        if state == "success":
            validate_existing_fit(args)
            return
        if state != "running":
            raise ValueError(f"Unexpected existing first fit status: {status}")
        elapsed = time.monotonic() - started
        if args.wait_timeout_seconds and elapsed >= args.wait_timeout_seconds:
            raise TimeoutError("Waiting for existing first fit timed out; artifacts retained")
        LOGGER.info(
            "WAIT existing transformer_cls seed42 fit (%.0fs): %s", elapsed, args.existing_first_fit
        )
        delay = args.poll_seconds
        if args.wait_timeout_seconds:
            delay = min(delay, max(0, args.wait_timeout_seconds - elapsed))
        time.sleep(delay)


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if not 0 < args.poll_seconds <= 60:
        arguments.error("--poll-seconds must be greater than zero and at most 60")
    if not 0 <= args.wait_timeout_seconds < float("inf"):
        arguments.error("--wait-timeout-seconds must be finite and nonnegative")
    for name in ("config", "prepared", "output_root", "existing_first_fit"):
        setattr(args, name, getattr(args, name).resolve())
    commands = stage_commands(args)
    outputs = [
        Path(command[command.index("--output") + 1])
        for _, command in commands
        if "--output" in command
    ]
    if len(outputs) != len(set(outputs)) or args.existing_first_fit in outputs:
        raise ValueError("Matrix output directories overlap")
    print(
        json.dumps(
            {
                "members": [{"method": method, "seed": seed} for method, seed in members()],
                "wait_for_existing_first_fit": str(args.existing_first_fit),
                "new_fit_count": 18,
                "stage_count": len(commands),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        print(
            "WAIT status.json success; fail immediately on failed; verify first-fit binding",
            flush=True,
        )
        for name, command in commands:
            print(json.dumps({"stage": name, "argv": command}), flush=True)
        return 0
    if not args.config.is_file() or not args.prepared.is_dir():
        raise ValueError("Config file and prepared run directory must exist")
    if args.output_root.exists() and (
        not args.output_root.is_dir() or any(args.output_root.iterdir())
    ):
        raise FileExistsError(f"Matrix output root must be absent or empty: {args.output_root}")
    for source in (args.existing_first_fit, args.prepared):
        if (
            source == args.output_root
            or args.output_root in source.parents
            or source in args.output_root.parents
        ):
            raise ValueError("Matrix output root must be separate from existing input runs")
    for output in outputs:
        if output.exists():
            raise FileExistsError(output)
    initial_status = read_json(args.existing_first_fit / "status.json")
    if initial_status.get("status") == "failed":
        raise RuntimeError(f"Existing first fit failed; artifacts retained: {initial_status}")
    validate_existing_fit(args)
    wait_for_first_fit(args)
    for index, (name, command) in enumerate(commands, start=1):
        LOGGER.info("START %d/%d %s argv=%s", index, len(commands), name, json.dumps(command))
        subprocess.run(command, cwd=REPOSITORY, check=True)
        LOGGER.info("SUCCESS %d/%d %s", index, len(commands), name)
    LOGGER.info("Completed all 19 Task 10 members")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError):
        LOGGER.exception("Task 10 matrix stopped; existing artifacts retained")
        raise SystemExit(1)
