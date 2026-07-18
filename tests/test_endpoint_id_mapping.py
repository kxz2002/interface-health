"""endpoint_id_mapping 唯一权威来源的单元测试。

锁住的核心不变量：build_contract.py 的 endpoint_id_map() 与
train_baseline_v0.py 的 id_to_endpoint_key() 必须互为精确逆映射——
否则整数 id 在两处指向不同 endpoint_key，ReliabilityGatedFusion 静默
查错 baseline 统计量。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts.endpoint_id_mapping import endpoint_id_map, id_to_endpoint_key

REPO_ROOT = Path(__file__).parents[1]

# 故意用非 sorted 顺序插入 key，验证映射结果只取决于 key 本身的排序，
# 不受 dict 插入顺序影响（dict 迭代顺序不保证跨 Python 版本/运行稳定）。
_SAMPLE_EP_TO_SVC = {
    "POST:/api/v1/z": "svc-z",
    "GET:/api/v1/a": "svc-a",
    "POST:/api/v1/m": "svc-m",
    "GET:/api/v1/b": "svc-b",
}


def test_id_to_endpoint_key_is_exact_inverse_of_endpoint_id_map():
    forward = endpoint_id_map(_SAMPLE_EP_TO_SVC)
    backward = id_to_endpoint_key(_SAMPLE_EP_TO_SVC)

    assert {v: k for k, v in forward.items()} == backward
    assert {v: k for k, v in backward.items()} == forward


def test_deterministic_across_calls():
    assert endpoint_id_map(_SAMPLE_EP_TO_SVC) == endpoint_id_map(_SAMPLE_EP_TO_SVC)
    assert id_to_endpoint_key(_SAMPLE_EP_TO_SVC) == id_to_endpoint_key(_SAMPLE_EP_TO_SVC)


def test_real_endpoint_to_service_yaml_covers_all_keys_with_no_gaps():
    ep_to_svc = yaml.safe_load(
        (REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text()
    )
    forward = endpoint_id_map(ep_to_svc)
    backward = id_to_endpoint_key(ep_to_svc)

    assert set(forward.keys()) == set(ep_to_svc.keys())
    # id 是 0..N-1 的连续整数，无空洞无重复
    assert sorted(forward.values()) == list(range(len(ep_to_svc)))
    assert set(backward.keys()) == set(range(len(ep_to_svc)))
    assert {v: k for k, v in forward.items()} == backward
