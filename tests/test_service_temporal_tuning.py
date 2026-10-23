import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import service_temporal_model as cli
from src.service.modeling.config import DEFAULT_CONFIG, load_config
from src.service.modeling.temporal import temporal_inputs
from src.service.modeling.temporal_tuning import (
    candidate_settings,
    normal_case_split,
    prepare_inner,
    select_candidate,
    tune_temporal,
)
from tests.test_service_model_logs import FakeTokenizer
from tests.test_service_model_smoke import synthetic_tables


def inputs():
    tables = synthetic_tables(load_config(DEFAULT_CONFIG))
    frames = {}
    cases = ["z", "z", "a", "b", "c", "fault", "calibration"]
    for name in ["grid", "resource", "execution", "log", "labels", "splits", "evaluation"]:
        frame = getattr(tables, name).copy()
        frame["case_id"] = cases
        frames[name] = frame
    frames["labels"].loc[:4, "label"] = 0
    frames["labels"].loc[:4, "label_source"] = "normal_run_reference"
    frames["splits"].loc[:4, "split"] = "fit"
    tables = replace(tables, **frames)
    outer = SimpleNamespace(
        identity=tables.grid,
        labels=tables.labels,
        splits=tables.splits,
        lists=SimpleNamespace(fit=tuple(tables.grid.sample_id.iloc[:5])),
    )
    return tables, outer


def test_inner_case_split_uses_real_time_and_fit_preprocessors_exclude_validation():
    tables, outer = inputs()
    split = normal_case_split(outer)
    assert split["validation_case"] == "c"
    inner, audit = prepare_inner(tables, outer, FakeTokenizer())
    assert set(inner.identity.case_id) == {"z", "a", "b", "c"}
    for state in audit["states"].values():
        assert set(state["fit_ids"]) == set(audit["actual_fit_ids"])
        assert not set(state["fit_ids"]) & set(split["validation_ids"])
    assert not temporal_inputs(inner)["history_ids"][split["validation_ids"][0]]
    changed = copy.deepcopy(tables)
    changed.resource.loc[4:, "cpu_cores"] = 1e12
    changed.log.loc[4:, "messages_json"] = '[{"message":"ERROR unseen validation", "count":1}]'
    _, other = prepare_inner(changed, outer, FakeTokenizer())
    assert other["preprocessor_hash"] == audit["preprocessor_hash"]


def test_search_fixed_candidates_and_normal_loss_tie_break():
    config = json.loads(Path("configs/service/tt_temporal_prior_bert_tuning_v1.json").read_text())
    assert len(candidate_settings(config["base_config"]["training"], config["search"])) == 12
    assert (
        select_candidate(
            [
                dict(validation_loss=2, test_ap=1),
                dict(validation_loss=1, test_ap=0),
                dict(validation_loss=1, test_ap=1),
            ]
        )
        == 1
    )
    with pytest.raises(ValueError, match="finite"):
        select_candidate([dict(validation_loss=float("nan"))])
    config["base_config"]["training"]["hidden_dim"] = 64
    with pytest.raises(ValueError, match="fixed"):
        candidate_settings(config["base_config"]["training"], config["search"])


def test_tuning_is_reproducible_and_external_test_cannot_change_selection(monkeypatch):
    import torch

    import src.service.modeling.temporal_tuning as tuning

    torch.set_num_threads(1)
    tables, outer = inputs()
    config = json.loads(Path("configs/service/tt_temporal_prior_bert_gpu_b_v1.json").read_text())
    config["training"].update(device="cpu", epochs=1)
    monkeypatch.setattr(tuning, "candidate_settings", lambda base, search: [base])
    inner, audit = prepare_inner(tables, outer, FakeTokenizer())
    embeddings = np.zeros((len(inner.sample_ids), 8), dtype=np.float32)
    first = tune_temporal(inner, embeddings, config, {}, audit["validation_ids"])
    changed = copy.deepcopy(tables)
    changed.resource.loc[5:, "cpu_cores"] = 1e12
    changed.labels.loc[5:, "label"] = 0
    other, other_audit = prepare_inner(changed, outer, FakeTokenizer())
    second = tune_temporal(other, embeddings, config, {}, other_audit["validation_ids"])
    assert first["selected_index"] == second["selected_index"]
    assert first["candidates"][0]["validation_loss"] == second["candidates"][0]["validation_loss"]
    with pytest.raises(ValueError, match="disjoint"):
        tune_temporal(inner, embeddings, config, {}, inner.lists.fit)


def test_tuning_cli_publishes_selected_config_and_normal_audit(tmp_path, monkeypatch):
    import src.service.modeling.temporal_tuning as tuning
    from scripts import service_temporal_tune as command

    tables, outer = inputs()
    outer.versions = tables.versions
    tuning_config = json.loads(
        Path("configs/service/tt_temporal_prior_bert_tuning_v1.json").read_text()
    )
    config = tuning_config["base_config"]
    config["training"].update(device="cpu", epochs=1)
    tables.config["log"].update(
        model=config["text_embedding"]["model"],
        tokenizer_revision=config["text_embedding"]["revision"],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(tuning_config))
    monkeypatch.setattr(command, "read_prepared", lambda path: (tables, outer, {}))
    monkeypatch.setattr(command, "load_inputs", lambda *args: tables)
    monkeypatch.setattr(command, "_reference", lambda path: {"path": "synthetic"})
    monkeypatch.setattr(command.LocalRuntime, "tokenizer", lambda self, config: FakeTokenizer())
    monkeypatch.setattr(tuning, "candidate_settings", lambda base, search: [base])
    monkeypatch.setattr(
        command,
        "build_frozen_text_embeddings",
        lambda prepared, ids, **kwargs: np.zeros((len(ids), 8), dtype=np.float32),
    )
    output = tmp_path / "tuning"
    assert (
        command.main(
            [
                "--config",
                str(config_path),
                "--prepared",
                "synthetic",
                "--preparation",
                "synthetic",
                "--label-bundle",
                "synthetic",
                "--output",
                str(output),
                "--tracking-uri",
                f"sqlite:///{tmp_path / 'mlflow.db'}",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["stage"] == "temporal_normal_tuning"
    result = json.loads((output / "run/tuning_results.json").read_text())
    assert result["selected_index"] == 0
    assert not (output / "evaluation/report.json").exists()
    selected = json.loads((output / "run/selected_config.json").read_text())
    cli.validate_temporal_config(selected)


@pytest.mark.parametrize(
    "corrupt", [None, "ids", "sha", "parent", "dtype", "revision", "log_cache"]
)
def test_text_cache_checks_identity_metadata_and_raw_vectors(tmp_path, monkeypatch, corrupt):
    run = tmp_path / "cached"
    (run / "run").mkdir(parents=True)
    vectors = np.zeros((2, 256), dtype=np.float32)
    metadata = dict(
        sample_ids=["a", "b"],
        embedding_dim=256,
        config={"revision": "fixed"},
        prepared_log_cache_key="cache",
        vectors_sha256=hashlib.sha256(vectors.tobytes()).hexdigest(),
    )
    manifest = dict(
        stage="temporal_experiment",
        provenance=dict(parents=dict(prepared={"sha": "fixed"})),
        artifacts={
            name: dict(path="run/" + suffix)
            for name, suffix in [
                ("text_embeddings", "text_embeddings.npy"),
                ("text_embedding_metadata", "text_embedding_metadata.json"),
            ]
        },
    )
    if corrupt == "ids":
        metadata["sample_ids"] = ["b", "a"]
    elif corrupt == "sha":
        metadata["vectors_sha256"] = "wrong"
    elif corrupt == "parent":
        manifest["provenance"]["parents"]["prepared"] = {}
    elif corrupt == "dtype":
        vectors = vectors.astype(np.float64)
    elif corrupt == "revision":
        metadata["config"] = {"revision": "wrong"}
    elif corrupt == "log_cache":
        metadata["prepared_log_cache_key"] = "wrong"
    manifest["provenance"]["text_embedding"] = {"revision": "fixed"}
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "run/config.json").write_text(json.dumps(dict(text_embedding={"revision": "fixed"})))
    (run / "run/text_embedding_metadata.json").write_text(json.dumps(metadata))
    np.save(run / "run/text_embeddings.npy", vectors)
    checked = []
    monkeypatch.setattr(cli, "check_run", lambda root: checked.append(root))
    monkeypatch.setattr(cli, "_reference", lambda root: {"sha": "fixed"})
    prepared = SimpleNamespace(sample_ids=["a", "b"], logs={"cache": {"cache_key": "cache"}})
    if corrupt:
        with pytest.raises(ValueError, match="cache"):
            cli.load_text_cache(run, tmp_path, prepared, {"revision": "fixed"})
    else:
        restored, _ = cli.load_text_cache(run, tmp_path, prepared, {"revision": "fixed"})
        np.testing.assert_array_equal(restored, vectors)
    assert checked == [run]
