from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class ContractDataset(Dataset):
    """从 contract parquet 加载数据的 PyTorch Dataset。"""

    def __init__(
        self,
        parquet_path: str | Path,
        schema_path: str | Path,
        mode: str = "point",
        nan_strategy: str = "zero",
        sequence_length: int = 8,
    ):
        self._mode = mode
        self._seq_len = sequence_length

        schema = json.loads(Path(schema_path).read_text())
        self._groups: dict[str, list[str]] = {
            name: grp["columns"] for name, grp in schema["feature_groups"].items()
        }

        self._df = pd.read_parquet(parquet_path)

        feature_cols = [c for cols in self._groups.values() for c in cols]
        if nan_strategy == "mean":
            means = self._df[feature_cols].mean()
            self._df[feature_cols] = self._df[feature_cols].fillna(means)
        else:
            self._df[feature_cols] = self._df[feature_cols].fillna(0.0)

        if mode == "point":
            self._indices: list = list(range(len(self._df)))
        elif mode == "sequence":
            self._indices = self._build_sequence_indices()
        else:
            raise ValueError(f"未知 mode: {mode}")

    def _build_sequence_indices(self) -> list[tuple[str, str, int]]:
        """构建 (case_id, endpoint_key, start_row_idx) 滑窗索引。不跨 case/endpoint 边界。"""
        indices = []
        for (case_id, ep_key), group in self._df.groupby(["case_id", "endpoint_key"], sort=False):
            g_idx = group.index.tolist()
            if len(g_idx) >= self._seq_len:
                for i in range(len(g_idx) - self._seq_len + 1):
                    indices.append((case_id, ep_key, g_idx[i]))
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "point":
            row = self._df.iloc[idx]
            return self._row_to_sample(row)
        case_id, ep_key, start_idx = self._indices[idx]
        loc = self._df.index.get_loc(start_idx)
        rows = self._df.iloc[loc : loc + self._seq_len]
        return self._rows_to_sequence_sample(rows)

    def _row_to_sample(self, row: pd.Series) -> dict:
        tensors = {
            name: torch.tensor(row[cols].values.astype(float), dtype=torch.float32)
            for name, cols in self._groups.items()
        }
        label = {
            "phase": row["phase"],
            "is_anomaly": bool(row["is_anomaly"]),
        }
        meta = {"sample_id": row["sample_id"], "endpoint_key": row["endpoint_key"]}
        return {**tensors, "label": label, "meta": meta}

    def _rows_to_sequence_sample(self, rows: pd.DataFrame) -> dict:
        tensors = {
            name: torch.tensor(rows[cols].values.astype(float), dtype=torch.float32)
            for name, cols in self._groups.items()
        }
        label = {
            "phase": rows["phase"].tolist(),
            "is_anomaly": rows["is_anomaly"].tolist(),
        }
        meta = {
            "case_id_per_step": rows["case_id"].tolist(),
            "endpoint_key_per_step": rows["endpoint_key"].tolist(),
        }
        return {**tensors, "label": label, "meta": meta}
