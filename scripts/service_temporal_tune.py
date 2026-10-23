"""Select optimization settings on a held-out Normal fit Case only."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.service_model import LocalRuntime, _json, _publish, _record, _reference, read_prepared
from scripts.service_temporal_model import validate_temporal_config
from src.service.modeling.data import load_inputs
from src.service.modeling.temporal_tuning import candidate_settings, prepare_inner, tune_temporal
from src.service.modeling.text_embeddings import build_frozen_text_embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "prepared", "preparation", "label-bundle", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    tuning = json.loads(args.config.read_text())
    if (
        set(tuning) != {"schema_version", "base_config", "search"}
        or tuning["schema_version"] != "service-temporal-normal-tuning-v1"
    ):
        raise ValueError("Unknown Normal-only tuning protocol")
    config = tuning["base_config"]
    validate_temporal_config(config)
    if config["schema_version"] != "service-temporal-prior-bert-v1":
        raise ValueError("Tuning requires the fixed prior BERT GRU architecture")
    candidate_settings(config["training"], tuning["search"])
    outer_tables, outer, _ = read_prepared(args.prepared)
    tables = load_inputs(args.preparation, args.label_bundle, outer_tables.config)
    if tables.hashes != outer_tables.hashes or tables.versions != outer_tables.versions:
        raise ValueError("Raw tuning input identity differs from outer prepared input")
    log_config = tables.config["log"]
    if (
        log_config["model"] != config["text_embedding"]["model"]
        or log_config["tokenizer_revision"] != config["text_embedding"]["revision"]
    ):
        raise ValueError("Tuning text model/tokenizer revision mismatch")
    inner, audit = prepare_inner(tables, outer, LocalRuntime().tokenizer(tables.config))
    provenance = dict(
        stage="temporal_normal_tuning",
        exploratory=True,
        parents=dict(prepared=_reference(args.prepared)),
        versions=outer.versions,
        selection_scope="latest_whole_normal_fit_case_only",
        preprocessor_hash=audit["preprocessor_hash"],
    )
    with _record(args.output, tuning, args, provenance) as recorder:
        embeddings = build_frozen_text_embeddings(
            inner,
            sorted(inner.sample_ids),
            model_name=config["text_embedding"]["model"],
            revision=config["text_embedding"]["revision"],
            device=config["training"]["device"],
            batch_size=config["text_embedding"]["batch_size"],
        )
        np.save(recorder.output / "run/inner_text_embeddings.npy", embeddings)
        result = tune_temporal(
            inner,
            embeddings,
            config,
            tuning["search"],
            audit["validation_ids"],
            progress=lambda row: (
                _json(recorder.output / "run/candidate_progress.json", row),
                recorder.log_epoch(
                    {"normal_validation_loss": row["validation_loss"]},
                    step=row["candidate_index"],
                ),
            ),
        )
        selected_path = recorder.output / "run/selected_config.json"
        _json(selected_path, result["selected_config"])
        _publish(
            recorder,
            dict(
                tuning_results=result,
                selected_config=selected_path,
                inner_split_and_preprocessing=audit,
                inner_text_embeddings=recorder.output / "run/inner_text_embeddings.npy",
                inner_text_embedding_metadata=dict(
                    sample_ids=sorted(inner.sample_ids),
                    config=config["text_embedding"],
                    prepared_log_cache_key=inner.logs["cache"]["cache_key"],
                    vectors_sha256=hashlib.sha256(embeddings.tobytes(order="C")).hexdigest(),
                ),
            ),
        )
    print(
        json.dumps(
            dict(selected_config=str(selected_path), selected_index=result["selected_index"])
        )
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
