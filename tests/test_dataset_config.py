from pathlib import Path

import pytest

from src.data.dataset_config import DatasetConfig, load_dataset_config


def test_load_merged_config(tmp_path):
    cfg_path = tmp_path / "merged.yaml"
    cfg_path.write_text(
        "name: merged_v1\n"
        "roots:\n"
        "  - data/anomod_v1\n"
        "  - data/endpoint_raw2\n"
        "normal_source: data/anomod_v1\n"
        "fused_window: 15s\n"
    )
    cfg = load_dataset_config(cfg_path)
    assert isinstance(cfg, DatasetConfig)
    assert cfg.name == "merged_v1"
    assert cfg.roots == [Path("data/anomod_v1"), Path("data/endpoint_raw2")]
    assert cfg.normal_source == Path("data/anomod_v1")
    assert cfg.fused_window == "15s"


def test_missing_roots_raises(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("name: x\nnormal_source: data/anomod_v1\n")
    with pytest.raises(ValueError, match="roots"):
        load_dataset_config(cfg_path)


def test_normal_source_must_be_in_roots(tmp_path):
    cfg_path = tmp_path / "bad2.yaml"
    cfg_path.write_text("name: x\nroots:\n  - data/anomod_v1\nnormal_source: data/not_listed\n")
    with pytest.raises(ValueError, match="normal_source"):
        load_dataset_config(cfg_path)
