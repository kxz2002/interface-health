import json
from pathlib import Path

import pandas as pd

from src.data.contract_dataloader import ContractDataset


def _write_fixture(tmp_path: Path, with_endpoint_id: bool) -> tuple[Path, Path]:
    cols = {
        "sample_id": ["s1", "s2"],
        "endpoint_key": ["epA", "epB"],
        "phase": ["normal", "normal"],
        "is_anomaly": [False, False],
        "f__x": [1.0, 2.0],
    }
    if with_endpoint_id:
        cols["endpoint_id"] = [0, 1]
    df = pd.DataFrame(cols)
    pq_path = tmp_path / "data.parquet"
    df.to_parquet(pq_path, index=False)
    schema = {"feature_groups": {"f": {"columns": ["f__x"]}}}
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(schema))
    return pq_path, schema_path


def test_row_to_sample_includes_endpoint_id_when_present(tmp_path):
    pq_path, schema_path = _write_fixture(tmp_path, with_endpoint_id=True)
    ds = ContractDataset(parquet_path=pq_path, schema_path=schema_path, nan_strategy="zero")
    sample = ds[0]
    assert sample["meta"]["endpoint_id"] == 0


def test_row_to_sample_omits_endpoint_id_when_absent(tmp_path):
    """v0 数据没有 endpoint_id 列时，meta 里不应出现这个 key（而不是填 None 之类的
    占位符）——下游 _collate 靠这个 key 是否存在判断要不要堆叠传给 fusion。"""
    pq_path, schema_path = _write_fixture(tmp_path, with_endpoint_id=False)
    ds = ContractDataset(parquet_path=pq_path, schema_path=schema_path, nan_strategy="zero")
    sample = ds[0]
    assert "endpoint_id" not in sample["meta"]
