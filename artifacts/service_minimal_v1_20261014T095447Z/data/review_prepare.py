"""Reproduce bounded T2 review from immutable prepared tables, without raw reads."""

import hashlib
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent
features = pd.read_parquet(DATA / "features.parquet")
index = pd.read_parquet(DATA / "index.parquet")
quality = pd.read_parquet(DATA / "instance_quality.parquet")
evidence = pd.read_parquet(DATA / "evidence.parquet")
source = json.loads((DATA / "source_manifest.json").read_text())
assert (DATA / "_SUCCESS").exists() and not (DATA / "_FAILURE").exists()
assert features.sample_id.is_unique and index.sample_id.is_unique
assert set(features.sample_id) == set(index.sample_id)
assert features.groupby(["case_id", "window_start_ms"]).size().eq(25).all()
assert index.groupby(["case_id", "window_start_ms"]).partition.nunique().eq(1).all()
assert index.paired_resource_eligible.sum() == features.resource_valid.sum()
assert index.loc[index.paired_resource_eligible, "exclusion_reasons"].map(len).eq(0).all()
assert evidence.independent_label.dtype == "Int8"
assert evidence.proxy_label.dtype == "Int8"
for code in source["executed_code"]:
    assert hashlib.sha256(Path(code["path"]).read_bytes()).hexdigest() == code["sha256"]
merged = features.merge(
    index[["sample_id", "phase", "partition"]], on="sample_id", validate="one_to_one"
)
selected = merged[
    merged.canonical_service_id.isin(
        ["ts-preserve-service", "ts-order-service", "ts-auth-service", "ts-gateway-service"]
    )
]
summary = (
    selected.groupby(["case_id", "canonical_service_id", "phase"])
    .agg(
        n_expected=("sample_id", "size"),
        n_resource_valid=("resource_valid", "sum"),
        cpu_min=("cpu_cores", "min"),
        cpu_max=("cpu_cores", "max"),
        cpu_median=("cpu_cores", "median"),
        ambiguous=("identity_ambiguity", "sum"),
    )
    .reset_index()
)
ids = (
    quality.groupby(["case_id", "canonical_service_id"])
    .container_id.nunique()
    .reset_index(name="container_ids")
)
result = dict(
    assertions="PASS",
    source_code_sha_match=True,
    complete_grid_rows=len(features),
    resource_valid=int(features.resource_valid.sum()),
    unknown=int((~features.resource_valid).sum()),
    selected_service_phase_summary=json.loads(summary.to_json(orient="records")),
    observed_id_counts=json.loads(ids.to_json(orient="records")),
    normal_shared_partition_counts=index[index.partition.str.startswith("normal_")]
    .groupby("partition")
    .window_start_ms.nunique()
    .to_dict(),
    evidence_note="Killing is instance termination only; CPU weak proxy is input-dependent; no availability gold or fitted model",
)
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
