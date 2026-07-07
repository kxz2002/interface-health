from pathlib import Path

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
    assert len(cases) == len(set(cases))  # 无重复


def test_enumerate_cases_multi_root_empty_raises_upstream(tmp_path):
    # 空 root 返回空列表（由 main() 负责 raise，此处只验证枚举本身不抛）
    assert _enumerate_cases_multi([tmp_path / "empty"]) == []
