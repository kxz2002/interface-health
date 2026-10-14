"""Evidence-separated state diagnostics and fixed controlled-target retrieval."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.benchmark_downstream import safe_auc, safe_auprc
from src.service.protocol import ServiceProtocol


def _metrics(frame: pd.DataFrame) -> dict[str, object]:
    known = frame.label.notna()
    finite = np.isfinite(frame.anomaly_score)
    valid = known & finite & frame.score_status.eq("Scored")
    y = frame.loc[valid, "label"].to_numpy(dtype=int)
    s = frame.loc[valid, "anomaly_score"].to_numpy(dtype=float)
    return dict(
        n_expected=len(frame),
        n_known=int(known.sum()),
        n_unknown=int((~known).sum()),
        n_unscored=int((known & ~valid).sum()),
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
        auroc=safe_auc(y, s),
        auprc=safe_auprc(y, s),
    )


def evaluate_s1(
    scores: pd.DataFrame, evidence: pd.DataFrame, index: pd.DataFrame
) -> dict[str, object]:
    """Resource WINDOW state only; instance/execution/availability are other tasks."""
    if index.sample_id.duplicated().any():
        raise ValueError("duplicate index sample IDs")
    conflicts = []
    admitted = []
    for row in evidence.to_dict("records"):
        reasons = []
        if row["unit"] != "WINDOW":
            reasons.append("unit_not_service_window")
        if row["state_class"] not in ("RESOURCE_CHANGE", "NO_OBSERVED_CHANGE"):
            reasons.append("state_not_resource")
        if row["evidence_grade"] != "INDEPENDENT_STATE":
            reasons.append("not_independent_state")
        if row["review_status"] != "confirmed":
            reasons.append("not_confirmed")
        if not row["heldout_from_model"]:
            reasons.append("not_heldout")
        if len(row["depends_on_input_sources"]):
            reasons.append("input_source_overlap")
        if pd.isna(row["independent_label"]):
            reasons.append("unknown_label")
        if (
            "KILL" in str(row.get("fact_type", "")).upper()
            or "TERMINATION" in str(row.get("fact_type", "")).upper()
        ):
            reasons.append("instance_termination_not_resource_gold")
        if reasons:
            conflicts.append(dict(evidence_id=row["evidence_id"], reasons=";".join(reasons)))
        else:
            admitted.append(row)
    independent = []
    for (case, service, group, method), frame in scores.groupby(
        ["case_id", "canonical_service_id", "feature_group", "method"]
    ):
        matching = [
            r for r in admitted if r["case_id"] == case and r["canonical_service_id"] == service
        ]
        for state in sorted({r["state_class"] for r in matching}):
            f = frame.copy()
            labels = []
            for sample in f.itertuples():
                if sample.partition in ("normal_fit", "normal_calibration"):
                    labels.append(None)
                    continue
                found = {
                    int(r["independent_label"])
                    for r in matching
                    if r["state_class"] == state
                    and r["start_ms"] <= sample.window_start_ms
                    and r["end_ms"] >= sample.window_end_ms
                }
                labels.append(next(iter(found)) if len(found) == 1 else None)
                if len(found) > 1:
                    conflicts.append(
                        dict(evidence_id=sample.sample_id, reasons="conflicting_window_labels")
                    )
            f["label"] = labels
            independent.append(
                dict(
                    case_id=case,
                    canonical_service_id=service,
                    state_class=state,
                    feature_group=group,
                    method=method,
                    **_metrics(f),
                )
            )
    proxy = scores.merge(
        index[["sample_id", "proxy_exposure_label"]], on="sample_id", validate="many_to_one"
    )
    proxy_rows = []
    for keys, frame in proxy.groupby(
        ["case_id", "canonical_service_id", "feature_group", "method"]
    ):
        if frame.proxy_exposure_label.notna().any():
            frame = frame.rename(columns={"proxy_exposure_label": "label"})
            proxy_rows.append(
                dict(
                    zip(["case_id", "canonical_service_id", "feature_group", "method"], keys),
                    **_metrics(frame),
                )
            )
    n_known = sum(r["n_known"] for r in independent)
    return dict(
        independent_s1=dict(
            n_known=n_known,
            auroc=None,
            auprc=None,
            rows=independent,
            reason=(
                "No matching independent resource WINDOW labels"
                if not n_known
                else "Metrics stratified by case/service/resource state; no pooled mixed-state metric"
            ),
        ),
        s1_proxy_controlled_exposure=dict(
            event_count=len({r["case_id"] for r in proxy_rows}),
            rows=proxy_rows,
            note="Controlled exposure association only; baseline is unexposed, not confirmed healthy; not independent state gold",
        ),
        evidence_conflicts=conflicts,
    )


def aggregate_events(
    scores: pd.DataFrame, events: pd.DataFrame, candidates: list[str], protocol: ServiceProtocol
) -> pd.DataFrame:
    if len(set(candidates)) != len(candidates) or events.event_id.duplicated().any():
        raise ValueError("duplicate candidates/events")
    rows = []
    width = protocol.config["window_ms"]
    for event in events.itertuples():
        start = math.ceil(event.evaluation_start_ms / width) * width
        expected = max(0, (event.evaluation_end_ms - start) // width)
        for service in candidates:
            for group in protocol.config["feature_groups"]:
                for method in protocol.config["methods"]:
                    f = scores[
                        (scores.case_id == event.case_id)
                        & (scores.canonical_service_id == service)
                        & (scores.feature_group == group)
                        & (scores.method == method)
                        & (scores.window_start_ms >= start)
                        & (scores.window_end_ms <= event.evaluation_end_ms)
                    ]
                    if f.sample_id.duplicated().any() or len(f) > expected:
                        raise ValueError("duplicate or excessive event windows")
                    valid = f[
                        f.score_status.eq("Scored")
                        & np.isfinite(f.calibrated_score)
                        & np.isfinite(f.anomaly_score)
                    ]
                    n = len(valid)
                    coverage = n / expected if expected else 0.0
                    k = math.ceil(protocol.config["event_top_fraction"] * n)
                    ok = (
                        expected > 0
                        and coverage >= protocol.config["event_min_coverage"]
                        and not f.score_status.eq("Untrained").any()
                    )
                    reasons = sorted({str(r) for values in f.quality_reasons for r in values})
                    if len(f) < expected:
                        reasons.append("missing_score_rows")
                    if coverage < protocol.config["event_min_coverage"]:
                        reasons.append("insufficient_coverage")
                    if f.score_status.eq("Untrained").any():
                        reasons.append("Untrained")
                    rows.append(
                        dict(
                            event_id=event.event_id,
                            case_id=event.case_id,
                            canonical_service_id=service,
                            feature_group=group,
                            method=method,
                            n_expected=expected,
                            n_valid=n,
                            coverage=coverage,
                            top_k=k,
                            event_score=(
                                float(valid.calibrated_score.nlargest(k).mean()) if ok else np.nan
                            ),
                            raw_median=float(valid.anomaly_score.median()) if n else np.nan,
                            raw_top20_mean=(
                                float(valid.anomaly_score.nlargest(k).mean()) if n else np.nan
                            ),
                            status="Scored" if ok else "Unscorable",
                            unknown_reasons=";".join(reasons),
                        )
                    )
    return pd.DataFrame(rows)


def evaluate_s2(
    event_scores: pd.DataFrame, events: pd.DataFrame, protocol: ServiceProtocol
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    arms = (
        event_scores[["feature_group", "method"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for group, method in arms:
        for event in events.itertuples():
            f = event_scores[
                (event_scores.event_id == event.event_id)
                & (event_scores.feature_group == group)
                & (event_scores.method == method)
            ]
            target = f[f.canonical_service_id == event.configured_target_service]
            if len(target) != 1:
                raise ValueError("event requires exactly one target candidate")
            t = target.iloc[0]
            others = f[f.canonical_service_id != event.configured_target_service]
            known = others[others.status.eq("Scored") & np.isfinite(others.event_score)]
            missing = len(others) - len(known)
            ok = t.status == "Scored" and np.isfinite(t.event_score)
            if ok:
                delta = known.event_score - t.event_score
                best = 1 + int((delta > 1e-12).sum())
                worst = best + int((delta.abs() <= 1e-12).sum()) + missing
            else:
                best = worst = None
            rows.append(
                dict(
                    event_id=event.event_id,
                    feature_group=group,
                    method=method,
                    target_service=event.configured_target_service,
                    target_status="Scored" if ok else "Unscorable",
                    target_score=float(t.event_score) if ok else None,
                    rank_best=best,
                    rank_worst=worst,
                    unknown_count=missing,
                    n_candidates=len(f),
                    hit_at_1=int(ok and worst <= 1),
                    hit_at_3=int(ok and worst <= 3),
                    rr=1 / worst if ok else 0.0,
                    target_coverage=float(t.get("coverage", 0)),
                )
            )
    table = pd.DataFrame(rows)
    summaries = []
    for (group, method), f in table.groupby(["feature_group", "method"]):
        valid = f[f.target_status == "Scored"]
        summaries.append(
            dict(
                feature_group=group,
                method=method,
                denominator=len(events),
                hit_at_1=float(f.hit_at_1.mean()),
                hit_at_1_numerator=int(f.hit_at_1.sum()),
                hit_at_3=float(f.hit_at_3.mean()),
                hit_at_3_numerator=int(f.hit_at_3.sum()),
                mrr=float(f.rr.mean()),
                rr_sum=float(f.rr.sum()),
                scorable_events=len(valid),
                scorable_subset_mrr=float(valid.rr.mean()) if len(valid) else None,
            )
        )
    return table, dict(
        task="S2 controlled injection target retrieval",
        arms=summaries,
        note="All events retained; ties and unknown competitors conservative; not full RCA",
    )


def render_report(data_dir: Path, run_dir: Path, evaluation_dir: Path, out: Path) -> None:
    """Build a standalone report using verified saved artifacts only."""
    import hashlib
    import json
    import platform
    import shutil
    import subprocess
    import sys
    from datetime import datetime, timezone

    import matplotlib
    import yaml

    from scripts.service_experiment import (
        PREPARE_REQUIRED_ARTIFACTS,
        RUN_REQUIRED_ARTIFACTS,
        _load_score_manifest,
    )
    from src.service.protocol import (
        reserve_output,
        validate_features,
        validate_scores,
        write_manifest,
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = datetime.now(timezone.utc).isoformat()
    repo = Path(__file__).resolve().parents[2]
    directories = [Path(p).resolve() for p in (data_dir, run_dir, evaluation_dir)]
    inventory = {}
    configs = []
    evaluation_required = frozenset(
        [
            "event_rankings.parquet",
            "event_comparison.csv",
            "coverage.csv",
            "s2_metrics.json",
            "s1_independent_metrics.json",
            "s1_proxy_metrics.json",
            "normal_tail_summary.csv",
            "evidence_conflicts.csv",
            "source_manifest.json",
            "resolved_config.yaml",
        ]
    )

    def remember(path, digest=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"report input must be a regular local file: {path}")
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"report input SHA256 is invalid: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest is not None and actual != digest:
            raise ValueError(f"report input hash mismatch: {path}")
        inventory[str(path)] = actual

    for directory, required in zip(
        directories, [PREPARE_REQUIRED_ARTIFACTS, RUN_REQUIRED_ARTIFACTS, evaluation_required]
    ):
        if not (directory / "_SUCCESS").is_file() or (directory / "_FAILURE").exists():
            raise ValueError("report input is not successful")
        manifest = _load_score_manifest(directory, required)
        configs.append(manifest["config"])
        for name, identity in manifest["outputs"].items():
            path = directory / name
            remember(path, identity["sha256"])
            if path.stat().st_size != identity["size_bytes"]:
                raise ValueError(f"report input size mismatch: {path}")
        remember(directory / "manifest.json")
        for alias in ["run_manifest.json", "evaluation_manifest.json"]:
            if (directory / alias).exists():
                if (directory / alias).is_symlink() or json.loads(
                    (directory / alias).read_text()
                ) != manifest:
                    raise ValueError("report manifest alias mismatch")
                remember(directory / alias)
    ignored = {"command", "out", "data_dir", "run_dir", "evaluation_dir", "dataset"}
    protocol_config = {k: v for k, v in configs[0].items() if k not in ignored}
    if any(
        {k: v for k, v in c.items() if k not in ignored} != protocol_config for c in configs[1:]
    ):
        raise ValueError("report protocol mismatch")
    config = configs[0]
    data_dir, run_dir, evaluation_dir = directories
    from src.service.protocol import validate_source_bindings

    validate_source_bindings(json.loads((run_dir / "source_manifest.json").read_text()), [data_dir])
    validate_source_bindings(
        json.loads((evaluation_dir / "source_manifest.json").read_text()), [data_dir, run_dir]
    )
    admission = evaluation_dir.parent / "admission"
    has_admission = (admission / "analysis_manifest.json").is_file()
    if has_admission:
        remember(admission / "analysis_manifest.json")
        analysis = json.loads((admission / "analysis_manifest.json").read_text())
        consumed = analysis["consumed_sha256"]
        required_consumed = {
            "data": {
                "events.parquet",
                "features.parquet",
                "index.parquet",
                "evidence.parquet",
                "quality_report.json",
            },
            "run": {"scores.parquet"},
            "evaluation": {
                "event_rankings.parquet",
                "event_comparison.csv",
                "s1_proxy_metrics.json",
                "s1_independent_metrics.json",
            },
        }
        current = dict(zip(["data", "run", "evaluation"], directories))
        expected_names = {
            f"{stage}/{name}" for stage, names in required_consumed.items() for name in names
        }
        if not isinstance(consumed, dict) or set(consumed) != expected_names:
            raise ValueError("admission binding missing or unexpected consumed identity")
        for name, digest in consumed.items():
            stage, filename = name.split("/")
            remember(current[stage] / filename, digest)
        for name, digest in analysis["output_sha256"].items():
            remember(admission / name, digest)
        remember(admission / "method_admission.md")
    protected = [
        repo / "data",
        repo.parent / "AnoMod",
        *directories,
        admission,
        *(Path(path).parent for path in inventory),
        *map(Path, config["dataset_resolved"]["roots"]),
    ]
    protected += [
        p for p in (repo / "artifacts").glob("*") if p.name.startswith(("contract", "baseline"))
    ]
    # Verify semantic alignment before reserving any output, without re-scoring.
    scores = pd.read_parquet(run_dir / "scores.parquet")
    features = pd.read_parquet(data_dir / "features.parquet")
    index = pd.read_parquet(data_dir / "index.parquet")
    events = pd.read_parquet(data_dir / "events.parquet")
    evidence = pd.read_parquet(data_dir / "evidence.parquet")
    comparison = pd.read_csv(evaluation_dir / "event_comparison.csv")
    rankings = pd.read_parquet(evaluation_dir / "event_rankings.parquet")
    validate_scores(scores)
    validate_features(features, ServiceProtocol(config))
    keys = ["case_id", "canonical_service_id", "window_start_ms", "window_end_ms", "partition"]
    if (
        index.sample_id.duplicated().any()
        or set(scores.sample_id) != set(index.sample_id)
        or set(features.sample_id) != set(index.sample_id)
    ):
        raise ValueError("report stable sample identity mismatch")
    for frame in [scores, features]:
        frame_keys = [key for key in keys if key in frame.columns]
        joined = frame.merge(
            index[["sample_id", *frame_keys]],
            on="sample_id",
            suffixes=("", "__index"),
            validate="many_to_one",
        )
        for key in frame_keys:
            if not joined[key].equals(joined[key + "__index"]):
                raise ValueError(f"report sample metadata mismatch: {key}")
    expected_events = {c["case_id"]: c for c in config["cases"] if c["role"] != "normal"}
    if events.event_id.duplicated().any() or set(events.event_id) != set(expected_events):
        raise ValueError("report event set mismatch")
    arms = {(g, m) for g in config["feature_groups"] for m in config["methods"]}
    for event in events.itertuples():
        c = expected_events[event.event_id]
        if (
            event.case_id != c["case_id"]
            or event.configured_target_service != c["target_service"]
            or event.evaluation_start_ms != c["inject_start_ms"] + config["guard_ms"]
            or event.evaluation_end_ms
            != min(c["inject_end_ms"] - config["guard_ms"], c["window_end_ms"])
        ):
            raise ValueError("report frozen event interval mismatch")
        rows = comparison[comparison.event_id == event.event_id]
        if (
            len(rows) != len(arms)
            or set(zip(rows.feature_group, rows.method)) != arms
            or not rows.target_service.eq(event.configured_target_service).all()
            or not rows.n_candidates.eq(len(config["services"])).all()
        ):
            raise ValueError("report event comparison linkage mismatch")
        for group, method in arms:
            candidates = rankings[
                (rankings.event_id == event.event_id)
                & (rankings.feature_group == group)
                & (rankings.method == method)
            ]
            if (
                len(candidates) != len(config["services"])
                or set(candidates.canonical_service_id) != set(config["services"])
                or not candidates.case_id.eq(event.case_id).all()
            ):
                raise ValueError("report event ranking candidate mismatch")
            target = candidates[
                candidates.canonical_service_id == event.configured_target_service
            ].iloc[0]
            compared = rows[(rows.feature_group == group) & (rows.method == method)].iloc[0]
            if target.status != compared.target_status or not np.isclose(
                target.coverage, compared.target_coverage
            ):
                raise ValueError("report target evaluation linkage mismatch")
    if len(comparison) != len(events) * len(arms) or len(rankings) != len(events) * len(arms) * len(
        config["services"]
    ):
        raise ValueError("report evaluation row completeness mismatch")
    out = reserve_output(Path(out), protected)
    try:
        for name in [
            "event_comparison.csv",
            "normal_tail_summary.csv",
            "coverage.csv",
            "evidence_conflicts.csv",
            "s1_independent_metrics.json",
            "s1_proxy_metrics.json",
            "s2_metrics.json",
        ]:
            shutil.copyfile(evaluation_dir / name, out / name)
        quality = (
            features.groupby(["canonical_service_id", "case_id"])
            .agg(n_windows=("sample_id", "size"), n_valid=("resource_valid", "sum"))
            .reset_index()
        )
        quality["coverage"] = quality.n_valid / quality.n_windows
        quality.to_csv(out / "service_coverage.csv", index=False)
        evidence.groupby(["evidence_grade", "state_class", "review_status"]).size().rename(
            "n_evidence"
        ).reset_index().to_csv(out / "evidence_grades.csv", index=False)
        if has_admission:
            for name in ["failure_modes.csv", "calibration_ties.csv", "method_admission.md"]:
                shutil.copyfile(admission / name, out / name)
        else:
            comparison[comparison.target_status.ne("Scored")].to_csv(
                out / "failure_modes.csv", index=False
            )
        images = []
        for i, event in enumerate(events.itertuples(), 1):
            selected = scores[
                (scores.case_id == event.case_id)
                & (scores.canonical_service_id == event.configured_target_service)
            ]
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            for axis, group in zip(axes[:2], config["feature_groups"]):
                for method, rows in selected[selected.feature_group == group].groupby("method"):
                    rows = rows.sort_values("window_end_ms")
                    axis.plot(
                        (rows.window_end_ms - event.configured_start_ms) / 60000,
                        rows.calibrated_score,
                        label=method,
                    )
                axis.set_ylabel(group + " calibrated")
                axis.legend(ncol=4)
            f = features[
                (features.case_id == event.case_id)
                & (features.canonical_service_id == event.configured_target_service)
            ].sort_values("window_end_ms")
            axes[2].step(
                (f.window_end_ms - event.configured_start_ms) / 60000,
                f.resource_valid.astype(int),
                where="post",
            )
            axes[2].set_ylabel("R3 valid (1/0)")
            axes[2].set_xlabel("Minutes from configured injection; offline closed windows")
            for axis in axes:
                axis.axvline(0, color="black", linestyle="--")
                axis.axvline(
                    (event.configured_end_ms - event.configured_start_ms) / 60000,
                    color="black",
                    linestyle="--",
                )
                if pd.notna(event.first_effect_evidence_ms):
                    axis.axvline(
                        (event.first_effect_evidence_ms - event.configured_start_ms) / 60000,
                        color="gray",
                        linestyle=":",
                    )
            fig.suptitle(event.case_id + " / " + event.configured_target_service)
            fig.tight_layout()
            for suffix in ["png", "svg"]:
                fig.savefig(out / f"event_{i}.{suffix}")
            plt.close(fig)
            images.append(f"![{event.case_id}](event_{i}.png)")
        s1 = json.loads((out / "s1_independent_metrics.json").read_text())
        text = "# Service 最小实验：2026-10-14 早期中期候选\n\n"
        text += "S1运行状态与S2受控注入目标检索分开；S3完整根因定位不属于本轮。25候选、3事件、8臂均保留。\n\n"
        text += f"独立S1 n_known={s1['n_known']}、AUROC={s1['auroc']}、AP={s1['auprc']}；人工读取同源输入不构成独立真值。\n\n"
        text += "Normal_tail仅描述分位，不是独立误报率；Unknown不代表健康。KILL不可评分保留失败和分母。不支持在线提前量、窗口独立重复、跨故障泛化或两个新方法已完成。\n\n"
        text += "[25 Service覆盖](service_coverage.csv) · [逐事件8臂排名/覆盖](event_comparison.csv) · [全部候选Unknown原因](coverage.csv) · [Normal尾段](normal_tail_summary.csv) · [证据等级](evidence_grades.csv) · [失败模式](failure_modes.csv)\n\n"
        text += comparison.to_markdown(index=False) + "\n\n" + "\n\n".join(images)
        text += "\n\n灰线为保存的首次效果证据区间起点，CPU并非精确生效瞬间；元数据关联仅作图表展示。\n\n"
        text += (
            (out / "method_admission.md").read_text()
            if has_admission
            else "方法准入未提供；不支持新方法成立。"
        )
        text += "\n\n10月26日冻结是未来计划节点。Endpoint公平9维RED/12维Response实验按原协议引用，不与expanded18/21维混表；KILL传播是此前审计证据，本轮没有重读raw。\n"
        (out / "report.md").write_text(text)
        (out / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        )
        for path, digest in inventory.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"report input changed: {path}")
        provenance = dict(
            started_at_utc=started,
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            command=sys.argv,
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            git_status=subprocess.check_output(
                ["git", "status", "--short"], cwd=repo, text=True
            ).strip(),
            environment=dict(
                python=sys.version,
                numpy=np.__version__,
                pandas=pd.__version__,
                matplotlib=matplotlib.__version__,
                platform=platform.platform(),
            ),
            source_sha256=inventory,
            code_sha256={
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    repo / "scripts/service_experiment.py",
                    repo / "src/service/protocol.py",
                ]
            },
            note="saved artifacts only; no fitting, raw parse, rescore or new labels",
        )
        (out / "source_manifest.json").write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
        )
        write_manifest(out, list(map(Path, inventory)), ServiceProtocol(config))
        (out / "report_manifest.json").write_text((out / "manifest.json").read_text())
        (out / "_SUCCESS").write_text(datetime.now(timezone.utc).isoformat() + "\n")
    except Exception as error:
        (out / "_FAILURE").write_text(f"{type(error).__name__}: {error}\n")
        raise
