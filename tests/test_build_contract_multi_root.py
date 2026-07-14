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
    with pytest.raises(RuntimeError, match="不存在或不是目录"):
        _enumerate_cases_multi([tmp_path / "empty"])


def test_enumerate_cases_multi_root_existing_but_no_cases_warns_not_raises(tmp_path):
    # root 存在但没有任何 _pipeline_out/ case：这是极可能的误配置，但只告警不 raise
    empty_root = tmp_path / "empty_but_real"
    empty_root.mkdir()

    assert _enumerate_cases_multi([empty_root]) == []


def test_enumerate_cases_multi_root_archived_case_not_scanned(tmp_path):
    # 归档：把 case 从 root 直接子目录挪到 root/_archive/ 下多一层，
    # 验证 _enumerate_cases_multi（root.glob("*/_pipeline_out")，只扫一层）不会扫到它。
    root = tmp_path / "anomod_like"
    _make_case(root, "Lv_P_DISKIO_preserve")
    (root / "_archive" / "Normal_old" / "_pipeline_out").mkdir(parents=True)

    cases = _enumerate_cases_multi([root])

    names = [c.name for c in cases]
    assert names == ["Lv_P_DISKIO_preserve"]
    assert "Normal_old" not in names


def test_enumerate_cases_multi_root_archiving_does_not_exclude_other_siblings(tmp_path):
    # 归档只应排除被多套一层目录的那个 case，同一 root 下其它一层深度的有效
    # case 必须原样保留——防止"按深度排除"机制误伤同级兄弟 case。
    root = tmp_path / "anomod_like"
    _make_case(root, "Lv_P_DISKIO_preserve")
    _make_case(root, "Lv_S_HTTPABORT_preserve")
    _make_case(root, "Lv_D_CPU_assurance")
    (root / "_archive" / "Normal_old" / "_pipeline_out").mkdir(parents=True)

    cases = _enumerate_cases_multi([root])

    names = [c.name for c in cases]
    assert names == ["Lv_D_CPU_assurance", "Lv_P_DISKIO_preserve", "Lv_S_HTTPABORT_preserve"]
    assert "Normal_old" not in names
