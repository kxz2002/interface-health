"""数据集组成配置加载：把"数据集由哪些 root 组成"沉淀成受版本控制的 config。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    roots: list[Path]
    normal_source: Path
    fused_window: str


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text())

    name = raw.get("name")
    if not name:
        raise ValueError(f"dataset config {path} 缺少 name 字段")

    roots_raw = raw.get("roots")
    if not roots_raw:
        raise ValueError(f"dataset config {path} 缺少 roots 字段（至少一个数据源目录）")
    roots = [Path(r) for r in roots_raw]

    normal_raw = raw.get("normal_source")
    if not normal_raw:
        raise ValueError(f"dataset config {path} 缺少 normal_source 字段")
    normal_source = Path(normal_raw)
    if normal_source not in roots:
        raise ValueError(
            f"dataset config {path} 的 normal_source={normal_source} 不在 roots {roots} 中"
        )

    return DatasetConfig(
        name=name,
        roots=roots,
        normal_source=normal_source,
        fused_window=raw.get("fused_window", "15s"),
    )
