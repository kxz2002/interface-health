import copy
import hashlib
import json

import pytest
import torch

from scripts.audit_gru_architecture import check_checkpoint, check_contexts, frozen_config
from src.service.modeling.temporal import (
    build_temporal_model,
    temporal_inputs,
    temporal_model_metadata,
)
from tests.test_service_model_temporal import fixture


def checkpoint_fixture(candidate, seed, inner):
    prepared, _ = fixture()
    expected = frozen_config(candidate)
    inputs = temporal_inputs(prepared, expected["context_windows"])
    fit_ids = sorted(prepared.lists.fit)
    identity = prepared.identity.set_index("sample_id")
    vocabulary = sorted(set(identity.loc[fit_ids, "canonical_service_id"].astype(str)))
    settings = {**expected["training"], "seed": seed}
    model = build_temporal_model(inputs, settings, vocabulary)
    config = copy.deepcopy(expected)
    if not inner:
        config["training"]["seed"] = seed
        config["isolation_forest"]["random_state"] = seed
        config["method"] = "gru"
    checkpoint = dict(
        config=config,
        fit_ids=list(fit_ids),
        model_state=model.state_dict(),
        architecture_metadata=temporal_model_metadata(model),
    )
    if inner:
        checkpoint["settings"] = settings
    return checkpoint, expected, fit_ids, identity, inputs


@pytest.mark.parametrize("candidate", ["service_bias", "short_long"])
@pytest.mark.parametrize("seed", [1, 2, 42])
@pytest.mark.parametrize("inner", [True, False])
def test_frozen_checkpoint_restores_for_each_candidate_seed_and_scope(candidate, seed, inner):
    arguments = checkpoint_fixture(candidate, seed, inner)
    metadata = check_checkpoint(*arguments, seed=seed, inner=inner)
    assert metadata["architecture"] == arguments[1]["training"]["architecture"]


@pytest.mark.parametrize("candidate", ["service_bias", "short_long"])
@pytest.mark.parametrize("mutation", ["architecture", "learning_rate", "fit_ids", "nonfinite"])
def test_checkpoint_rejects_architecture_protocol_fit_and_parameter_corruption(candidate, mutation):
    checkpoint, *arguments = checkpoint_fixture(candidate, 1, False)
    if mutation == "architecture":
        checkpoint["architecture_metadata"]["architecture"] = "shared_gru"
    elif mutation == "learning_rate":
        checkpoint["config"]["training"]["lr"] *= 2
    elif mutation == "fit_ids":
        checkpoint["fit_ids"].append("test")
    else:
        next(iter(checkpoint["model_state"].values())).fill_(float("nan"))
    with pytest.raises(ValueError):
        check_checkpoint(checkpoint, *arguments, seed=1)


@pytest.mark.parametrize("candidate", ["service_bias", "short_long"])
def test_inner_checkpoint_rejects_wrong_actual_seed(candidate):
    checkpoint, *arguments = checkpoint_fixture(candidate, 1, True)
    checkpoint["settings"]["seed"] = 42
    with pytest.raises(ValueError, match="settings or seed"):
        check_checkpoint(checkpoint, *arguments, seed=1, inner=True)


def test_service_dictionary_cannot_include_a_test_only_service():
    checkpoint, expected, fit_ids, identity, inputs = checkpoint_fixture("service_bias", 1, False)
    identity.loc["test", "canonical_service_id"] = "test_only_service"
    checkpoint["architecture_metadata"]["service_vocabulary"].append("test_only_service")
    with pytest.raises(ValueError, match="fit-only services"):
        check_checkpoint(checkpoint, expected, fit_ids, identity, inputs, seed=1)


def write_contexts(root, histories):
    path = root / "contexts.json"
    path.write_text(json.dumps(histories))
    payload = path.read_bytes()
    return dict(
        artifacts=dict(
            context_sample_ids=dict(
                path=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
            )
        )
    )


@pytest.mark.parametrize("length", [4, 8])
def test_actual_context_artifact_passes_with_causal_contiguous_history(tmp_path, length):
    prepared, _ = fixture()
    histories = temporal_inputs(prepared, length)["history_ids"]
    manifest = write_contexts(tmp_path, histories)
    check_contexts(tmp_path, manifest, prepared, length)


@pytest.mark.parametrize(
    "boundary", ["case", "service", "split", "gap", "future", "missing_sample", "excluded_fit"]
)
def test_actual_context_artifact_rejects_boundaries_and_future_windows(tmp_path, boundary):
    prepared, _ = fixture()
    histories = temporal_inputs(prepared, 8)["history_ids"]
    if boundary == "future":
        histories["fit-003"] = ["fit-004"]
    elif boundary == "missing_sample":
        del histories["test"]
    elif boundary == "excluded_fit":
        prepared.lists.fit = tuple(
            sample_id for sample_id in prepared.lists.fit if sample_id != "fit-002"
        )
    elif boundary == "split":
        prepared.splits.loc[prepared.splits.sample_id.eq("fit-002"), "split"] = "test"
    else:
        column, value = {
            "case": ("case_id", "other_case"),
            "service": ("canonical_service_id", "other_service"),
            "gap": ("window_start_ms", 999999),
        }[boundary]
        prepared.identity.loc[prepared.identity.sample_id.eq("fit-002"), column] = value
    manifest = write_contexts(tmp_path, histories)
    with pytest.raises(ValueError):
        check_contexts(tmp_path, manifest, prepared, 8)
