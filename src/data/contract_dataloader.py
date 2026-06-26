from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class ContractDataset(Dataset):
    """从 contract parquet 加载数据的 PyTorch Dataset。"""

    def __init__(
        self,
        parquet_path: str | Path,
        schema_path: str | Path,
        mode: str = "point",
        nan_strategy: str = "zero",
        sequence_length: int = 8,
        fit_on_parquet: str | Path | None = None,
    ):
        self._mode = mode
        self._seq_len = sequence_length
        if sequence_length < 1:
            raise ValueError(f"sequence_length 必须 >= 1，got {sequence_length}")

        schema = json.loads(Path(schema_path).read_text())
        self._groups: dict[str, list[str]] = {
            name: grp["columns"] for name, grp in schema["feature_groups"].items()
        }

        self._df = pd.read_parquet(parquet_path)

        feature_cols = [c for cols in self._groups.values() for c in cols]
        if nan_strategy not in ("mean", "zero"):
            raise ValueError(f"未知 nan_strategy: {nan_strategy!r}，支持 'mean' 或 'zero'")
        if nan_strategy == "mean":
            # 默认用自身均值；传入 fit_on_parquet 时改用外部（训练集）均值，
            # 避免 eval 用自身均值填补——eval 自填会引入统计量偏差，应使用训练集均值保持一致性。
            fit_src = pd.read_parquet(fit_on_parquet) if fit_on_parquet else self._df
            means = fit_src[feature_cols].mean()
            all_nan_cols = means[means.isna()].index.tolist()
            if all_nan_cols:
                raise ValueError(f"fit 源数据以下特征列全为 NaN，无法均值填补：{all_nan_cols}")
            self._df[feature_cols] = self._df[feature_cols].fillna(means)
        else:
            all_nan_cols = [c for c in feature_cols if self._df[c].isna().all()]
            if all_nan_cols:
                logger.warning(
                    "nan_strategy='zero': 以下特征列全为 NaN，将填 0"
                    "（可能表示某模态数据缺失）：%s",
                    all_nan_cols,
                )
            self._df[feature_cols] = self._df[feature_cols].fillna(0.0)

        if mode == "point":
            self._indices: list = list(range(len(self._df)))
        elif mode == "sequence":
            self._indices = self._build_sequence_indices()
        else:
            raise ValueError(f"未知 mode: {mode}")

    def _build_sequence_indices(self) -> list[list]:
        """构建滑窗索引，每项为一个窗口的 DataFrame index 列表。不跨 case/endpoint 边界。

        存整窗 index 列表（而非起始物理行号），按 DataFrame index 显式切片，
        不依赖同一 group 的行在 parquet 中物理连续。
        """
        indices = []
        for _, group in self._df.groupby(["case_id", "endpoint_key"], sort=False):
            g_idx = group.index.tolist()
            if len(g_idx) >= self._seq_len:
                for i in range(len(g_idx) - self._seq_len + 1):
                    indices.append(g_idx[i : i + self._seq_len])
        return indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        if self._mode == "point":
            row = self._df.iloc[idx]
            return self._row_to_sample(row)
        idx_list = self._indices[idx]
        rows = self._df.loc[idx_list]
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
