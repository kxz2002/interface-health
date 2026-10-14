"""Isolated Service preparation; resource model fitting belongs to score."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf

# Direct script invocation follows the repository's existing entrypoint convention.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.service.metrics import build_metric_windows
from src.service.protocol import (
    EVIDENCE_COLUMNS,
    PROTOCOL_ID,
    load_case_meta,
    load_protocol,
    make_indices,
    reserve_output,
    validate_evidence,
    write_manifest,
)

LOG = logging.getLogger(__name__)
EVIDENCE = REPO / "configs/service/evidence_minimal_v1.json"


def _json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _identify(path: Path, *, small: bool) -> dict:
    stat = path.stat()
    result = dict(
        path=str(path.resolve()),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        identification="sha256" if small else "stat-only; not a content lock",
    )
    if small:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _events(protocol, evidence):
    records = []
    for case in protocol.config["cases"]:
        if not case["is_anomaly_case"]:
            continue
        effects = evidence[
            (evidence.case_id == case["case_id"])
            & evidence.fact_type.isin(["CPU_RESOURCE_CHANGE", "INSTANCE_TERMINATION"])
        ]
        records.append(
            dict(
                event_id=case["case_id"],
                case_id=case["case_id"],
                fault_type=case["role"],
                configured_target_service=case["target_service"],
                configured_start_ms=case["inject_start_ms"],
                configured_end_ms=case["inject_end_ms"],
                first_effect_evidence_ms=int(effects.start_ms.min()) if len(effects) else None,
                effect_evidence_ids=effects.evidence_id.tolist(),
                effect_grade=";".join(sorted(set(effects.evidence_grade))),
                candidate_set_id=PROTOCOL_ID + "::25-services",
                evaluation_start_ms=case["inject_start_ms"] + protocol.config["guard_ms"],
                evaluation_end_ms=min(
                    case["inject_end_ms"] - protocol.config["guard_ms"], case["window_end_ms"]
                ),
                review_status=(
                    "confirmed"
                    if len(effects) and effects.review_status.eq("confirmed").all()
                    else "pending_human_review"
                ),
                causal_ambiguity="CPU change is input-derived; Killing establishes instance termination only; service availability unknown",
            )
        )
    events = pd.DataFrame(records)
    events["first_effect_evidence_ms"] = events.first_effect_evidence_ms.astype("Int64")
    return events


def prepare(
    config_path: Path,
    dataset_path: Path,
    out: Path,
    evidence_path: Path = EVIDENCE,
    *,
    resolved_config: dict | None = None,
) -> Path:
    """Validate selected inputs before any resource read; never overwrite outputs."""
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    if resolved_config is None:
        protocol = load_protocol(Path(config_path), Path(dataset_path))
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="service-resolved-") as folder:
            resolved = Path(folder) / "protocol.yaml"
            resolved.write_text(yaml.safe_dump(resolved_config))
            protocol = load_protocol(resolved, Path(dataset_path))
    roots = [Path(p).resolve() for p in protocol.config["dataset_resolved"]["roots"]]
    protected = [REPO / "data", REPO.parent / "AnoMod", *roots]
    protected.extend(
        p for p in (REPO / "artifacts").glob("*") if p.name.startswith(("contract", "baseline"))
    )
    out = reserve_output(Path(out), protected)
    try:
        (out / ".gitignore").write_text("/*.parquet\n/*.joblib\n")
        evidence_path = Path(evidence_path)
        evidence = pd.DataFrame(
            json.loads(evidence_path.read_text())["evidence"], columns=EVIDENCE_COLUMNS
        )
        for label in ("independent_label", "proxy_label"):
            evidence[label] = pd.array(evidence[label], dtype="Int8")
        validate_evidence(evidence)
        selected = []
        small_inputs = [Path(config_path), Path(dataset_path), evidence_path]
        raw_inputs = []
        for case in protocol.config["cases"]:
            candidates = [
                root / case["case_id"] for root in roots if (root / case["case_id"]).is_dir()
            ]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"case {case['case_id']} requires exactly one dataset directory; found {candidates}"
                )
            folder = candidates[0]
            meta = load_case_meta(folder)
            for key in (
                "case_id",
                "is_anomaly_case",
                "window_start_ms",
                "window_end_ms",
                "inject_start_ms",
                "inject_end_ms",
                "target_service",
                "chaos_kind",
                "chaos_yaml",
                "recoverable",
            ):
                if meta.get(key) != case.get(key):
                    raise ValueError(
                        f"case metadata {folder}: {key} differs from frozen config: {meta.get(key)!r} != {case.get(key)!r}"
                    )
            paths = sorted((folder / "metric_data").glob(f"{case['case_id']}_metrics*.csv"))
            if len(paths) != 1:
                raise FileNotFoundError(
                    f"case {case['case_id']} requires exactly one original Metric CSV; found {paths}"
                )
            selected.append((paths, meta))
            raw_inputs.extend(paths)
            small_inputs.append(folder / "trace_data/case_metadata.json")
        # Small held-out Kubernetes event files and chaos plans are provenance only.
        # Original CSV content hashes in evidence are historical citations, not checked here.
        for source in evidence.source_path.dropna().unique():
            source = Path(source)
            if source not in raw_inputs:
                if not source.is_file():
                    raise FileNotFoundError(f"evidence source missing: {source}")
                # Only bounded metadata/config sources receive a second full read.
                # Metric/Log observations remain stat-only, regardless of fixture size.
                if (
                    source.suffix.lower() in {".yaml", ".yml", ".json"}
                    and source.stat().st_size <= 8 << 20
                ):
                    small_inputs.append(source)
                else:
                    raw_inputs.append(source)
        source_before = [_identify(p, small=True) for p in small_inputs]
        source_before += [_identify(p, small=False) for p in raw_inputs]
        feature_frames, quality_frames, case_reports = [], [], []
        for paths, meta in selected:
            case_start = time.monotonic()
            features, quality = build_metric_windows(paths, meta, protocol)
            case_reports.append(
                dict(
                    case_id=meta["case_id"],
                    feature_rows=len(features),
                    instance_quality_rows=len(quality),
                    elapsed_s=time.monotonic() - case_start,
                    metric_input_bytes=sum(p.stat().st_size for p in paths),
                    partial_windows=features.attrs.get("partial_windows"),
                    timestamp_semantics=features.attrs.get("timestamp_semantics"),
                )
            )
            feature_frames.append(features)
            quality_frames.append(quality)
        features = pd.concat(feature_frames, ignore_index=True)
        quality = pd.concat(quality_frames, ignore_index=True)
        index = make_indices(features, protocol)
        valid = features.set_index("sample_id").resource_valid
        reasons = features.set_index("sample_id").quality_reasons
        index["paired_resource_eligible"] = index.sample_id.map(valid).astype(bool)
        index["exclusion_reasons"] = [
            [] if eligible else list(reasons.loc[sample])
            for sample, eligible in zip(index.sample_id, index.paired_resource_eligible)
        ]
        # Half-open interval overlap establishes an evidence link, never a new label.
        index["evidence_ids"] = [
            evidence.loc[
                (evidence.case_id == row.case_id)
                & (evidence.canonical_service_id == row.canonical_service_id)
                & (evidence.start_ms < row.window_end_ms)
                & (evidence.end_ms > row.window_start_ms),
                "evidence_id",
            ].tolist()
            for row in index.itertuples()
        ]
        tables = dict(
            features=features,
            instance_quality=quality,
            index=index,
            events=_events(protocol, evidence),
            evidence=evidence,
        )
        for name, frame in tables.items():
            frame.to_parquet(out / f"{name}.parquet", index=False)
        coverage = (
            index.groupby(["case_id", "canonical_service_id", "partition", "phase"], observed=True)
            .paired_resource_eligible.agg(["size", "sum"])
            .reset_index()
        )
        coverage.columns = [
            "case_id",
            "canonical_service_id",
            "partition",
            "phase",
            "n_expected",
            "n_resource_valid",
        ]
        coverage["coverage"] = coverage.n_resource_valid / coverage.n_expected
        coverage.to_csv(out / "coverage.csv", index=False)
        # Small full-content snapshots also detect changes that preserve file stat.
        for identity in source_before:
            now = Path(identity["path"]).stat()
            content_changed = (
                "sha256" in identity
                and hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
                != identity["sha256"]
            )
            if content_changed or (now.st_size, now.st_mtime_ns) != (
                identity["size_bytes"],
                identity["mtime_ns"],
            ):
                raise RuntimeError(f"input changed during prepare: {identity['path']}")
        finished = datetime.now(timezone.utc).isoformat()
        import pyarrow
        import sklearn

        code_sources = [
            _identify(Path(__file__), small=True),
            _identify(REPO / "src/service/protocol.py", small=True),
            _identify(REPO / "src/service/metrics.py", small=True),
        ]
        environment = dict(
            python=sys.version,
            platform=platform.platform(),
            numpy=np.__version__,
            pandas=pd.__version__,
            pyarrow=pyarrow.__version__,
            sklearn=sklearn.__version__,
        )
        (out / "resolved_config.yaml").write_text(
            yaml.safe_dump(protocol.config, allow_unicode=True, sort_keys=False)
        )
        _json(
            out / "source_manifest.json",
            dict(
                started_at_utc=started,
                finished_at_utc=finished,
                sources=source_before,
                executed_code=code_sources,
                resolved_config_sha256=hashlib.sha256(
                    (out / "resolved_config.yaml").read_bytes()
                ).hexdigest(),
                raw_identification_note="large Metric/Log sources stat-only; no full raw content hash; no raw copies",
                small_evidence_hash_budget_bytes=8 << 20,
                source_snapshot_policy="bounded config/metadata full SHA256 before and after scan; large Metric/Log stat-only",
                evidence_hash_note="source_sha256 in evidence is frozen historical provenance; not recomputed for raw CSV",
                git_commit=_git(["rev-parse", "HEAD"]),
                git_status=_git(["status", "--short"]),
                environment=environment,
            ),
        )
        _json(
            out / "schema.json",
            dict(
                protocol_id=PROTOCOL_ID,
                schema=protocol.config["schema"],
                tables={n: {c: str(t) for c, t in f.dtypes.items()} for n, f in tables.items()},
            ),
        )
        _json(
            out / "quality_report.json",
            dict(
                cases=case_reports,
                feature_rows=len(features),
                resource_valid_rows=int(features.resource_valid.sum()),
                unknown_rows=int((~features.resource_valid).sum()),
                state_gold_note="Independent Service availability labels absent; Unknown remains unknown",
                cases_evidence_count=len(evidence),
                elapsed_s=time.monotonic() - start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                metric_input_bytes=sum(p.stat().st_size for paths, _ in selected for p in paths),
                artifacts={
                    n: dict(rows=len(f), bytes=(out / f"{n}.parquet").stat().st_size)
                    for n, f in tables.items()
                },
            ),
        )
        write_manifest(out, [*small_inputs, *raw_inputs], protocol)
        (out / "_SUCCESS").write_text(finished + "\n")
        LOG.info("Service prepare complete: %s", out)
        return out
    except Exception as error:
        (out / "_FAILURE").write_text(f"{type(error).__name__}: {error}\n")
        raise


# Formal prepare data assets; .gitignore is operational rather than data schema.
PREPARE_REQUIRED_ARTIFACTS = frozenset(
    {
        "features.parquet",
        "index.parquet",
        "instance_quality.parquet",
        "events.parquet",
        "evidence.parquet",
        "coverage.csv",
        "quality_report.json",
        "resolved_config.yaml",
        "schema.json",
        "source_manifest.json",
    }
)


def _load_score_manifest(data_dir: Path, required: frozenset = PREPARE_REQUIRED_ARTIFACTS) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate manifest key: {key}")
            result[key] = value
        return result

    manifest = json.loads((data_dir / "manifest.json").read_text(), object_pairs_hook=unique_object)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError("manifest and config must be objects")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("manifest outputs must be a nonempty object")
    missing = required - outputs.keys()
    if missing:
        raise ValueError(f"manifest missing required prepare artifacts: {sorted(missing)}")
    for name, expected in outputs.items():
        if not name or name in {".", ".."} or Path(name).parts != (name,) or "\\" in name:
            raise ValueError(f"unsafe manifest artifact name: {name!r}")
        path = data_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"manifest artifact must be a regular local file: {name}")
        if not isinstance(expected, dict):
            raise ValueError(f"manifest identity must be an object: {name}")
        digest, size = expected.get("sha256"), expected.get("size_bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"manifest SHA256 is invalid: {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"manifest size_bytes is invalid: {name}")
    return manifest


def score(
    config_path: Path,
    dataset_path: Path,
    data_dir: Path,
    out: Path,
    *,
    resolved_config: dict | None = None,
) -> Path:
    """Consume content-locked prepared data; fit each frozen reference once."""
    import tempfile

    import pyarrow
    import sklearn

    from src.service.baselines import fit_baselines, load_bundle, save_bundle, score_baselines

    start = datetime.now(timezone.utc).isoformat()
    if resolved_config is None:
        protocol = load_protocol(config_path, dataset_path)
    else:
        with tempfile.TemporaryDirectory(prefix="service-score-config-") as folder:
            path = Path(folder) / "protocol.yaml"
            path.write_text(yaml.safe_dump(resolved_config))
            protocol = load_protocol(path, dataset_path)
    data_dir = Path(data_dir).resolve()
    if not (data_dir / "_SUCCESS").is_file() or (data_dir / "_FAILURE").exists():
        raise ValueError("score input requires successful frozen prepare")
    manifest = _load_score_manifest(data_dir)
    # Recheck every output hash, including config/provenance, before any fit.
    input_identities = []
    for name, expected in manifest["outputs"].items():
        identity = _identify(data_dir / name, small=True)
        if (
            identity["sha256"] != expected["sha256"]
            or identity["size_bytes"] != expected["size_bytes"]
        ):
            raise ValueError(f"prepared input hash mismatch: {name}")
        input_identities.append(identity)
    for key, value in protocol.config.items():
        if key in {"command", "out", "data_dir", "run_dir", "evaluation_dir", "dataset"}:
            continue
        if manifest["config"].get(key) != value:
            raise ValueError(f"prepared protocol mismatch: {key}")
    features = pd.read_parquet(data_dir / "features.parquet")
    index = pd.read_parquet(data_dir / "index.parquet")
    protected = [REPO / "data", REPO.parent / "AnoMod", data_dir]
    protected += [Path(p) for p in protocol.config["dataset_resolved"]["roots"]]
    protected += [
        p for p in (REPO / "artifacts").glob("*") if p.name.startswith(("contract", "baseline"))
    ]
    out = reserve_output(out, protected)
    try:
        (out / ".gitignore").write_text("/*.parquet\n/*.joblib\n")
        bundle = fit_baselines(features, index, protocol)
        scores = score_baselines(features, index, bundle, protocol)
        # save_bundle refuses existing output; stage below newly reserved run.
        staged = out / "bundle"
        save_bundle(bundle, staged)
        reloaded = score_baselines(features, index, load_bundle(staged), protocol)
        for column in ("anomaly_score", "calibrated_score"):
            np.testing.assert_allclose(
                scores[column], reloaded[column], rtol=1e-10, atol=1e-12, equal_nan=True
            )
        pd.testing.assert_frame_equal(
            scores.drop(columns=["anomaly_score", "calibrated_score"]),
            reloaded.drop(columns=["anomaly_score", "calibrated_score"]),
        )
        for artifact in staged.iterdir():
            artifact.rename(out / artifact.name)
        staged.rmdir()
        scores.to_parquet(out / "scores.parquet", index=False)
        normal = scores[scores.partition.isin(["normal_fit", "normal_calibration", "normal_tail"])]
        normal.to_parquet(out / "normal_reference_scores.parquet", index=False)
        index[index.partition.isin(["normal_fit", "normal_calibration", "normal_tail"])][
            [
                "sample_id",
                "case_id",
                "canonical_service_id",
                "window_start_ms",
                "window_end_ms",
                "partition",
                "paired_resource_eligible",
            ]
        ].to_parquet(out / "normal_reference_ids.parquet", index=False)
        diagnostic = normal.groupby(
            ["canonical_service_id", "feature_group", "method", "partition"]
        ).agg(
            n_expected=("sample_id", "size"),
            n_scored=("anomaly_score", "count"),
            raw_min=("anomaly_score", "min"),
            raw_median=("anomaly_score", "median"),
            raw_max=("anomaly_score", "max"),
            calibrated_min=("calibrated_score", "min"),
            calibrated_median=("calibrated_score", "median"),
            calibrated_max=("calibrated_score", "max"),
        )
        diagnostic.to_csv(out / "normal_reference_summary.csv")
        # Equality includes Unknown/Untrained rows and every partition.
        counts = scores.groupby(["sample_id", "feature_group"]).score_status.first().unstack()
        if not counts["R3"].equals(counts["CPU-only"]):
            raise AssertionError("CPU-only and R3 scoring eligibility differs")
        _json(
            out / "quality_report.json",
            dict(
                score_rows=len(scores),
                expected_score_rows=len(features) * 8,
                status_counts=scores.score_status.value_counts().to_dict(),
                trained_service_groups=sum(
                    r["status"] == "Trained" for r in bundle.references.values()
                ),
                model_fit_count=len(bundle.models),
                shared_resource_ids_verified=True,
                roundtrip_score_match=True,
                reference_note="Simple baselines are research foundations; no independent Service state gold or method innovation claim",
            ),
        )
        (out / "resolved_config.yaml").write_text(
            yaml.safe_dump(protocol.config, allow_unicode=True, sort_keys=False)
        )
        code_paths = [
            Path(__file__),
            REPO / "src/service/baselines.py",
            REPO / "src/service/protocol.py",
            REPO / "scripts/benchmark_downstream.py",
        ]
        for identity in input_identities:
            if _identify(Path(identity["path"]), small=True)["sha256"] != identity["sha256"]:
                raise RuntimeError(f"input changed during score: {identity['path']}")
        _json(
            out / "source_manifest.json",
            dict(
                started_at_utc=start,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                command=sys.argv,
                sources=input_identities + [_identify(data_dir / "manifest.json", small=True)],
                executed_code=[_identify(p, small=True) for p in code_paths],
                config_sources=[
                    _identify(Path(config_path), small=True),
                    _identify(Path(dataset_path), small=True),
                ],
                git_commit=_git(["rev-parse", "HEAD"]),
                git_status=_git(["status", "--short"]),
                environment=dict(
                    python=sys.version,
                    numpy=np.__version__,
                    pandas=pd.__version__,
                    pyarrow=pyarrow.__version__,
                    sklearn=sklearn.__version__,
                    platform=platform.platform(),
                ),
                seed=42,
                fit_budget=100,
                score_direction="higher is more anomalous",
                calibration_note="empirical tail ranking with ties; no probability or conformal guarantee",
            ),
        )
        write_manifest(
            out,
            [data_dir / "manifest.json", *[Path(x["path"]) for x in input_identities]],
            protocol,
        )
        output_manifest = json.loads((out / "manifest.json").read_text())
        output_manifest["excluded_outputs"][
            "run_manifest.json"
        ] = "manifest alias; self-reference excluded"
        _json(out / "manifest.json", output_manifest)
        _json(out / "run_manifest.json", output_manifest)
        (out / "_SUCCESS").write_text(datetime.now(timezone.utc).isoformat() + "\n")
        return out
    except Exception as error:
        (out / "_FAILURE").write_text(f"{type(error).__name__}: {error}\n")
        raise


RUN_REQUIRED_ARTIFACTS = frozenset(
    {
        "scores.parquet",
        "normal_reference_scores.parquet",
        "normal_reference_ids.parquet",
        "normal_reference_summary.csv",
        "training_ids.parquet",
        "calibration_scores.parquet",
        "models.joblib",
        "preprocessing.json",
        "quality_report.json",
        "resolved_config.yaml",
        "source_manifest.json",
    }
)


def evaluate(
    config_path: Path,
    dataset_path: Path,
    data_dir: Path,
    run_dir: Path,
    out: Path,
    *,
    resolved_config: dict | None = None,
) -> Path:
    """Evaluate frozen scores without fitting, raw reads, or new labels."""
    import tempfile

    import pyarrow
    import sklearn

    from src.service.evaluation import aggregate_events, evaluate_s1, evaluate_s2
    from src.service.protocol import validate_scores

    started = datetime.now(timezone.utc).isoformat()
    if resolved_config is None:
        protocol = load_protocol(config_path, dataset_path)
    else:
        with tempfile.TemporaryDirectory(prefix="service-evaluate-config-") as folder:
            path = Path(folder) / "protocol.yaml"
            path.write_text(yaml.safe_dump(resolved_config))
            protocol = load_protocol(path, dataset_path)
    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    identities = []
    for folder, required in (
        (data_dir, PREPARE_REQUIRED_ARTIFACTS),
        (run_dir, RUN_REQUIRED_ARTIFACTS),
    ):
        if not (folder / "_SUCCESS").is_file() or (folder / "_FAILURE").exists():
            raise ValueError("evaluation requires successful frozen data/run")
        manifest = _load_score_manifest(folder, required)
        if folder == run_dir:
            alias = folder / "run_manifest.json"
            if alias.is_symlink() or not alias.is_file():
                raise ValueError("run manifest alias missing or not local regular file")
            if json.loads(alias.read_text()) != manifest:
                raise ValueError("run manifest alias differs from manifest")
            identities.append(_identify(alias, small=True))
        for name, expected in manifest["outputs"].items():
            identity = _identify(folder / name, small=True)
            if (
                identity["sha256"] != expected["sha256"]
                or identity["size_bytes"] != expected["size_bytes"]
            ):
                raise ValueError(f"evaluation input hash mismatch: {folder/name}")
            identities.append(identity)
        identities.append(_identify(folder / "manifest.json", small=True))
        for key, value in protocol.config.items():
            if (
                key not in {"command", "out", "data_dir", "run_dir", "evaluation_dir", "dataset"}
                and manifest["config"].get(key) != value
            ):
                raise ValueError(f"evaluation protocol mismatch: {key}")
    scores = pd.read_parquet(run_dir / "scores.parquet")
    index = pd.read_parquet(data_dir / "index.parquet")
    evidence = pd.read_parquet(data_dir / "evidence.parquet")
    events = pd.read_parquet(data_dir / "events.parquet")
    validate_scores(scores)
    evidence["depends_on_input_sources"] = evidence.depends_on_input_sources.map(list)
    validate_evidence(evidence)
    keys = ["case_id", "canonical_service_id", "window_start_ms", "window_end_ms", "partition"]
    if index.sample_id.duplicated().any() or set(scores.sample_id) != set(index.sample_id):
        raise ValueError("scores/index full sample identity mismatch")
    joined = scores.merge(
        index[["sample_id", *keys]],
        on="sample_id",
        suffixes=("", "__index"),
        validate="many_to_one",
    )
    for key in keys:
        if not joined[key].equals(joined[key + "__index"]):
            raise ValueError(f"scores/index identity mismatch: {key}")
    saved = (
        pd.read_parquet(run_dir / "normal_reference_ids.parquet")
        .set_index("sample_id")
        .sort_index()
    )
    normal = (
        index[index.partition.isin(["normal_fit", "normal_calibration", "normal_tail"])]
        .set_index("sample_id")
        .sort_index()
    )
    pd.testing.assert_frame_equal(saved, normal[saved.columns], check_dtype=False)
    reference = pd.read_parquet(run_dir / "normal_reference_scores.parquet")
    pd.testing.assert_frame_equal(
        reference.reset_index(drop=True),
        scores[
            scores.partition.isin(["normal_fit", "normal_calibration", "normal_tail"])
        ].reset_index(drop=True),
    )
    # Metadata intervals must equal frozen protocol, never adapted to scores.
    fault = {c["case_id"]: c for c in protocol.config["cases"] if c["role"] != "normal"}
    if set(events.case_id) != set(fault) or len(events) != 3:
        raise ValueError("evaluation requires all three frozen primary events")
    for event in events.itertuples():
        c = fault[event.case_id]
        if (
            event.event_id != c["case_id"]
            or event.configured_target_service != c["target_service"]
            or event.evaluation_start_ms != c["inject_start_ms"] + protocol.config["guard_ms"]
            or event.evaluation_end_ms
            != min(c["inject_end_ms"] - protocol.config["guard_ms"], c["window_end_ms"])
        ):
            raise ValueError("event differs from fixed metadata protocol")
    protected = [REPO / "data", REPO.parent / "AnoMod", data_dir, run_dir]
    protected += [Path(p) for p in protocol.config["dataset_resolved"]["roots"]]
    protected += [
        p for p in (REPO / "artifacts").glob("*") if p.name.startswith(("contract", "baseline"))
    ]
    out = reserve_output(out, protected)
    try:
        (out / ".gitignore").write_text("/*.parquet\n/*.joblib\n")
        event_scores = aggregate_events(scores, events, protocol.config["services"], protocol)
        comparison, metrics = evaluate_s2(event_scores, events, protocol)
        event_scores.to_parquet(out / "event_rankings.parquet", index=False)
        comparison.to_csv(out / "event_comparison.csv", index=False)
        _json(out / "s2_metrics.json", metrics)
        s1 = evaluate_s1(scores, evidence, index)
        _json(out / "s1_independent_metrics.json", s1["independent_s1"])
        _json(out / "s1_proxy_metrics.json", s1["s1_proxy_controlled_exposure"])
        pd.DataFrame(s1["evidence_conflicts"], columns=["evidence_id", "reasons"]).to_csv(
            out / "evidence_conflicts.csv", index=False
        )
        event_scores[
            [
                "event_id",
                "canonical_service_id",
                "feature_group",
                "method",
                "n_expected",
                "n_valid",
                "coverage",
                "status",
                "unknown_reasons",
            ]
        ].to_csv(out / "coverage.csv", index=False)
        tail = scores[scores.partition == "normal_tail"]
        summaries = []
        for (service, group, method), f in tail.groupby(
            ["canonical_service_id", "feature_group", "method"]
        ):
            row = dict(
                canonical_service_id=service,
                feature_group=group,
                method=method,
                n_expected=len(f),
                n_scored=int(f.score_status.eq("Scored").sum()),
                interpretation="descriptive Normal tail scores; not confirmed FPR",
            )
            for col in ["anomaly_score", "calibrated_score"]:
                for q in [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]:
                    row[f"{col}_q{q:g}"] = f[col].quantile(q)
            summaries.append(row)
        pd.DataFrame(summaries).to_csv(out / "normal_tail_summary.csv", index=False)
        (out / "resolved_config.yaml").write_text(
            yaml.safe_dump(protocol.config, allow_unicode=True, sort_keys=False)
        )
        for identity in identities:
            if _identify(Path(identity["path"]), small=True)["sha256"] != identity["sha256"]:
                raise RuntimeError(f"evaluation input changed: {identity['path']}")
        _json(
            out / "source_manifest.json",
            dict(
                started_at_utc=started,
                finished_at_utc=datetime.now(timezone.utc).isoformat(),
                command=sys.argv,
                sources=identities,
                executed_code=[
                    _identify(p, small=True)
                    for p in [
                        Path(__file__),
                        REPO / "src/service/evaluation.py",
                        REPO / "src/service/protocol.py",
                        REPO / "scripts/benchmark_downstream.py",
                    ]
                ],
                config_sources=[
                    _identify(Path(config_path), small=True),
                    _identify(Path(dataset_path), small=True),
                ],
                git_commit=_git(["rev-parse", "HEAD"]),
                git_status=_git(["status", "--short"]),
                environment=dict(
                    python=sys.version,
                    numpy=np.__version__,
                    pandas=pd.__version__,
                    pyarrow=pyarrow.__version__,
                    sklearn=sklearn.__version__,
                    platform=platform.platform(),
                ),
                note="Frozen artifacts only; no fitting or raw scan; S1 independent resource labels may be absent",
            ),
        )
        write_manifest(out, [Path(x["path"]) for x in identities], protocol)
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["excluded_outputs"][
            "evaluation_manifest.json"
        ] = "manifest alias; self-reference excluded"
        _json(out / "manifest.json", manifest)
        _json(out / "evaluation_manifest.json", manifest)
        (out / "_SUCCESS").write_text(datetime.now(timezone.utc).isoformat() + "\n")
        return out
    except Exception as error:
        (out / "_FAILURE").write_text(f"{type(error).__name__}: {error}\n")
        raise


@hydra.main(config_path="../configs/service", config_name="minimal_v1", version_base=None)
def main(cfg: DictConfig) -> None:
    command = cfg.command
    if command == "evaluate":
        evaluate(
            REPO / "configs/service/minimal_v1.yaml",
            Path(cfg.dataset),
            Path(cfg.data_dir),
            Path(cfg.run_dir),
            Path(cfg.out),
            resolved_config=OmegaConf.to_container(cfg, resolve=True),
        )
        return
    if command == "score":
        score(
            REPO / "configs/service/minimal_v1.yaml",
            Path(cfg.dataset),
            Path(cfg.data_dir),
            Path(cfg.out),
            resolved_config=OmegaConf.to_container(cfg, resolve=True),
        )
        return
    if command != "prepare":
        raise NotImplementedError(f"Service command {command!r} is not implemented")
    # Overrides are validated through load_protocol; original config stays identifiable.
    prepare(
        REPO / "configs/service/minimal_v1.yaml",
        Path(cfg.dataset),
        Path(cfg.out),
        resolved_config=OmegaConf.to_container(cfg, resolve=True),
    )


if __name__ == "__main__":
    main()
