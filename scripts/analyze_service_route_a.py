"""Audit and compare the fixed 21-run Service route A experiment matrix."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.service_model import check_run


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def components(frame):
    return {
        column: {
            "mean": float(frame[column].mean()),
            "median": float(frame[column].median()),
            "p95": float(frame[column].quantile(0.95)),
        }
        for column in ["score", "value_loss", "availability_loss"]
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path.home() / "Code/Repos/interface-health/outputs/service"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root
    families = {
        "prior_gru": "temporal-multiseed-a7e7794-gru",
        "bert": "temporal-bert-bf5fdfd-gru",
        "bert_tuned": "temporal-bert-tuned-a89bd1f-gru",
        "no_prior": "route-a-no-prior-8bb258c",
        "global_prior": "route-a-global-prior-3ad3675",
        "bert_zero": "route-a-bert-zero-a8e9e30",
        "bert_reverse": "route-a-bert-reverse-a8e9e30",
    }
    destination = args.output
    if destination.exists():
        raise ValueError("Refusing to overwrite analysis")
    rows = []
    identity = None
    for family, prefix in families.items():
        for seed in [1, 2, 42]:
            run = root / f"{prefix}-seed{seed}-20261023-a"
            if not (run / "manifest.json").exists():
                raise ValueError(f"Missing published experiment: {run}")
            verified = check_run(run)
            manifest = read(run / "manifest.json")
            report = read(run / "evaluation/report.json")
            scores = pd.read_parquet(run / "run/scores.parquet").set_index("sample_id")
            calibration = pd.read_parquet(run / "run/calibration_scores.parquet")
            binding = dict(
                fit=read(run / "run/fit_ids.json"),
                test=sorted(scores.index),
                calibration=sorted(calibration.sample_id),
                main=sorted(report["main"]["sample_ids"]),
                versions=report["versions"],
                prepared=manifest["provenance"]["parents"]["prepared"],
            )
            current_identity = digest(binding)
            if identity is None:
                identity = current_identity
            if identity != current_identity:
                raise ValueError(f"Experiment identity mismatch: {run}")
            main = report["main"]
            grouped = {}
            for group in ["by_case", "by_service"]:
                grouped[group] = {}
                for name, part in main[group].items():
                    selected = scores.loc[part["sample_ids"]]
                    grouped[group][name] = dict(
                        rows=part["rows"],
                        ids_sha256=digest(sorted(part["sample_ids"])),
                        auroc=part["auroc"],
                        ap=part["ap"],
                        auc_reason=part["auc_reason"],
                        classification=part["classification"]["0.99"],
                        components=components(selected),
                    )
            text_record = manifest["artifacts"].get("text_control_summary")
            history = read(run / manifest["artifacts"]["training_history"]["path"])
            environment = manifest["provenance"]["environment"]
            entry = dict(
                family=family,
                seed=seed,
                run_id=run.name,
                manifest_sha256=hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
                identity_sha256=current_identity,
                verification=verified,
                environment_sha256=digest(environment),
                environment=environment,
                training_summary=dict(
                    epochs=len(history),
                    first_loss=history[0]["mean_loss"],
                    last_loss=history[-1]["mean_loss"],
                    minimum_loss=min(row["mean_loss"] for row in history),
                    elapsed_seconds=history[-1]["elapsed_seconds"],
                ),
                config=read(run / "run/config.json"),
                auroc=main["auroc"],
                ap=main["ap"],
                rows=main["rows"],
                classification=main["classification"]["0.99"],
                threshold=report["thresholds"]["thresholds"]["0.99"],
                calibration_components=components(calibration),
                text_control=read(run / text_record["path"]) if text_record else None,
                **grouped,
            )
            rows.append(entry)
            print(
                json.dumps({key: entry[key] for key in ["family", "seed", "auroc", "ap"]}),
                flush=True,
            )

    summary = {}
    for family in families:
        entries = [entry for entry in rows if entry["family"] == family]
        metrics = {}
        for metric in ["auroc", "ap", "recall", "normal_alert_rate"]:
            values = [
                entry[metric] if metric in entry else entry["classification"][metric]
                for entry in entries
            ]
            metrics[metric] = dict(mean=statistics.mean(values), std=statistics.stdev(values))
        summary[family] = metrics
    for metrics in summary.values():
        for metric in ["auroc", "ap"]:
            metrics[metric]["delta_vs_prior_gru"] = (
                metrics[metric]["mean"] - summary["prior_gru"][metric]["mean"]
            )
    result = dict(identity_sha256=identity, runs=rows, summary=summary)
    with destination.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
