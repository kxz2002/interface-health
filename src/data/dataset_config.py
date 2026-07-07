"""数据集组成配置加载：把"数据集由哪些 root 组成"沉淀成受版本控制的 config。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    roots: tuple[Path, ...]
    normal_source: Path
    fused_window: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DatasetConfig 缺少 name 字段")
        if not self.roots:
            raise ValueError("DatasetConfig 缺少 roots 字段（至少一个数据源目录）")
        if self.normal_source not in self.roots:
            raise ValueError(
                f"DatasetConfig 的 normal_source={self.normal_source} 不在 roots {self.roots} 中"
            )


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"dataset config {path} 解析结果不是合法的 YAML mapping（可能为空文件或格式错误）"
        )

    name = raw.get("name")

    roots_raw = raw.get("roots")
    if not isinstance(roots_raw, list):
        raise ValueError(
            f"dataset config {path} 的 roots 字段必须是列表，实际为 {type(roots_raw).__name__}: {roots_raw!r}"
        )
    roots = tuple(Path(r) for r in roots_raw)

    normal_raw = raw.get("normal_source")
    if not normal_raw:
        raise ValueError(f"dataset config {path} 缺少 normal_source 字段")
    normal_source = Path(normal_raw)

    return DatasetConfig(
        name=name,
        roots=roots,
        normal_source=normal_source,
        fused_window=raw.get("fused_window", "15s"),
    )
