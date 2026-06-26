from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class ModalityPreprocessor(ABC):
    version: str = "v0"

    @abstractmethod
    def fit(self, raw_paths: list[Path]) -> None: ...

    @abstractmethod
    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame: ...

    @abstractmethod
    def get_feature_columns(self) -> list[str]: ...
