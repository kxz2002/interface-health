from pathlib import Path

import pytest

from scripts.build_contract import _enumerate_cases_multi


def _make_case(root: Path, name: str):
    (root / name / "_pipeline_out").mkdir(parents=True)


def test_enumerate_cases_multi_root_merges_and_sorts(tmp_path):
    root_a = tmp_path / "ds_a"
    root_b = tmp_path / "ds_b"
    _make_case(root_a, "Normal")
    _make_case(root_a, "Lv_S_HTTPABORT_preserve")
    _make_case(root_b, "Lv_E_HTTPABORT_assurance")

    cases = _enumerate_cases_multi([root_a, root_b])

    names = [c.name for c in cases]
    assert names == ["Lv_E_HTTPABORT_assurance", "Lv_S_HTTPABORT_preserve", "Normal"]
    assert len(names) == len(set(names))  # case 名无重复


def test_enumerate_cases_multi_root_nonexistent_raises(tmp_path):
    # 不存在的 root（如配置里的路径打错）现在应显式 raise，而不是静默返回空
    with pytest.raises(RuntimeError, match="empty"):
        _enumerate_cases_multi([tmp_path / "empty"])


def test_enumerate_cases_multi_root_existing_but_no_cases_warns_not_raises(tmp_path):
    # root 存在但没有任何 _pipeline_out/ case：这是极可能的误配置，但只告警不 raise
    empty_root = tmp_path / "empty_but_real"
    empty_root.mkdir()

    assert _enumerate_cases_multi([empty_root]) == []
