"""Bounded optimization selected exclusively by held-out Normal Case loss."""

from __future__ import annotations

import copy
import itertools
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .availability import fit_availability
from .config import content_hash
from .data import assemble_prepared, build_sample_lists
from .logs import fit_log_statistics, fit_logs, prepare_logs, transform_log_statistics
from .numeric import fit_numeric, transform_numeric
from .temporal import score_temporal, temporal_inputs, train_temporal


def normal_case_split(outer):
    identity = outer.identity.set_index("sample_id")
    ids = set(outer.lists.fit)
    labels = outer.labels.set_index("sample_id")
    splits = outer.splits.set_index("sample_id")
    if (
        not ids
        or not labels.loc[sorted(ids), "label"].eq(0).all()
        or not splits.loc[sorted(ids), "split"].eq("fit").all()
        or not labels.loc[sorted(ids), "label_source"].eq("normal_run_reference").all()
    ):
        raise ValueError("Inner selection requires confirmed Normal outer-fit IDs")
    cases = sorted(set(identity.loc[sorted(ids), "case_id"]))
    if len(cases) != 4:
        raise ValueError("Inner selection requires exactly four Normal fit Cases")
    case_times = {
        case: int(identity.loc[identity.case_id.eq(case), "window_start_ms"].min())
        for case in cases
    }
    validation_case = max(cases, key=lambda case: (case_times[case], case))
    validation_ids = sorted(ids & set(identity.index[identity.case_id.eq(validation_case)]))
    fit_ids = sorted(ids - set(validation_ids))
    return dict(
        fit_cases=sorted(set(cases) - {validation_case}),
        validation_case=validation_case,
        case_start_ms=case_times,
        fit_ids=fit_ids,
        validation_ids=validation_ids,
        outer_fit_ids=sorted(ids),
    )


def prepare_inner(tables, outer, tokenizer):
    split = normal_case_split(outer)
    cases = set(split["fit_cases"]) | {split["validation_case"]}
    ids = set(tables.grid.loc[tables.grid.case_id.isin(cases), "sample_id"])
    if not set(split["outer_fit_ids"]) <= ids:
        raise ValueError("Raw inputs do not cover the Normal fit Cases")
    frames = {}
    for name in ["grid", "resource", "execution", "log", "labels", "splits", "evaluation"]:
        frame = getattr(tables, name)
        frames[name] = frame.loc[frame.sample_id.isin(ids)].copy()
    frame_ids = frames["splits"].sample_id
    frames["splits"]["split"] = np.where(frame_ids.isin(split["fit_ids"]), "fit", "test")
    inner_tables = replace(tables, **frames, feature_tables={})
    config = tables.config
    permitted = set(split["fit_ids"])
    numeric_state = fit_numeric(inner_tables, permitted, config)
    numeric = transform_numeric(inner_tables, numeric_state)
    valid = numeric["masks"].set_index("sample_id").any(axis=1)
    fit_ids = permitted & set(valid.index[valid])
    if not fit_ids:
        raise ValueError("No observed inner-fit windows")
    numeric_state = fit_numeric(inner_tables, fit_ids, config)
    numeric = transform_numeric(inner_tables, numeric_state)
    log_state = fit_logs(inner_tables, fit_ids, config)
    logs = prepare_logs(inner_tables, log_state, tokenizer, config)
    lists = build_sample_lists(inner_tables, numeric, pd.Series(logs["masks"]), "common")
    if set(lists.fit) != fit_ids:
        raise ValueError("Inner numeric fit membership changed after freezing")
    statistics_state = fit_log_statistics(inner_tables, logs, fit_ids, config)
    logs["statistics"] = transform_log_statistics(inner_tables, logs, statistics_state)
    provenance = dict(
        numeric_state_hash=numeric_state.state_hash,
        log_state_hash=log_state.state_hash,
        log_statistics_state_hash=statistics_state.state_hash,
    )
    prepared = assemble_prepared(inner_tables, numeric, logs, lists, provenance)
    states = dict(
        numeric=asdict(numeric_state),
        log={
            name: getattr(log_state, name)
            for name in [
                "templates",
                "config",
                "fit_ids",
                "fit_input_hash",
                "snapshot_hash",
                "state_hash",
            ]
        },
        log_statistics=asdict(statistics_state),
    )
    audit = dict(
        **split,
        actual_fit_ids=sorted(fit_ids),
        excluded_inner_fit_ids=sorted(permitted - fit_ids),
        states=states,
        preprocessor_hash=content_hash(states),
        feature_coverage={
            column: {
                "fit_observed": int(
                    numeric["masks"].set_index("sample_id").loc[sorted(fit_ids), column].sum()
                ),
                "validation_observed": int(
                    numeric["masks"]
                    .set_index("sample_id")
                    .loc[split["validation_ids"], column]
                    .sum()
                ),
            }
            for column in numeric["masks"].columns
            if column != "sample_id"
        },
        service_feature_coverage={
            str(service): {
                column: {
                    scope: int(
                        numeric["masks"]
                        .set_index("sample_id")
                        .loc[sorted(set(group.sample_id) & set(scope_ids)), column]
                        .sum()
                    )
                    for scope, scope_ids in [
                        ("fit_observed", fit_ids),
                        ("validation_observed", split["validation_ids"]),
                    ]
                }
                for column in numeric["masks"].columns
                if column != "sample_id"
            }
            for service, group in inner_tables.grid.groupby("canonical_service_id")
        },
    )
    return prepared, audit


def candidate_settings(base, search):
    if search != dict(lr=[0.0003, 0.001, 0.003], weight_decay=[0.0001, 0.001], epochs=[50, 100]):
        raise ValueError("Only the predeclared twelve-candidate search is supported")
    if any(
        base[name] != value
        for name, value in dict(
            seed=42, hidden_dim=32, batch_size=128, gradient_clip=1.0, checkpoint="last"
        ).items()
    ):
        raise ValueError("Tuning must preserve the fixed BERT GRU architecture and seed")
    return [
        {**base, "lr": lr, "weight_decay": weight_decay, "epochs": epochs}
        for lr, weight_decay, epochs in itertools.product(
            search["lr"], search["weight_decay"], search["epochs"]
        )
    ]


def select_candidate(results):
    if not results or any(not np.isfinite(row["validation_loss"]) for row in results):
        raise ValueError("Candidate validation losses must be finite")
    return min(range(len(results)), key=lambda index: (results[index]["validation_loss"], index))


def tune_temporal(prepared, embeddings, base_config, search, validation_ids, progress=None):
    validation_ids = sorted(validation_ids)
    labels = prepared.labels.set_index("sample_id").label
    if (
        not validation_ids
        or set(validation_ids) & set(prepared.lists.fit)
        or not labels.loc[validation_ids].eq(0).all()
    ):
        raise ValueError("Validation must be disjoint known-Normal windows")
    identity = prepared.identity.set_index("sample_id")
    if set(identity.loc[validation_ids, "case_id"]) & set(
        identity.loc[list(prepared.lists.fit), "case_id"]
    ):
        raise ValueError("Validation and fit must use disjoint Cases")
    prior = fit_availability(prepared, prepared.lists.fit)
    inputs = temporal_inputs(prepared, 4, availability_prior=prior, text_embeddings=embeddings)
    results = []
    for index, settings in enumerate(candidate_settings(base_config["training"], search)):
        model, history = train_temporal(prepared, inputs, settings)
        scores = score_temporal(model, inputs, validation_ids)
        row = dict(
            candidate_index=index,
            settings=settings,
            validation_loss=float(scores.score.mean()),
            validation_value_loss=float(scores.value_loss.mean()),
            validation_availability_loss=float(scores.availability_loss.mean()),
            validation_windows=len(scores),
            validation_scores=scores[
                ["sample_id", "score", "value_loss", "availability_loss"]
            ].to_dict("records"),
            training_history=history,
        )
        results.append(row)
        if progress is not None:
            progress(row)
    selected = select_candidate(results)
    best = copy.deepcopy(base_config)
    best["training"] = results[selected]["settings"]
    return dict(
        candidates=results,
        selected_index=selected,
        selected_config=best,
        selection_metric="normal_validation_mean_masked_mse_plus_bce",
        availability_prior=prior,
    )
