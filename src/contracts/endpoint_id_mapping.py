"""endpoint_key 字符串 ↔ 整数 id 映射的唯一权威来源。

build_contract.py（写入 endpoint_id 列）与 train_baseline_v0.py（构造
ReliabilityGatedFusion 的 id_to_endpoint_key 反查表）曾各自 inline 一份
`sorted(ep_to_svc.keys())` 派生逻辑——两份实现必须永远一致，否则整数 id
在两处指向不同的 endpoint_key，ReliabilityGatedFusion 会静默查错 baseline
统计量：不会报错，只会让门控结果失真，且很难在结果层面察觉。这里把派生
逻辑收敛到一处，两个公开函数都基于同一次 sorted() 调用，任何一处需要改变
排序规则时另一处自动同步。
"""

from __future__ import annotations


def _sorted_endpoint_keys(ep_to_svc: dict[str, str]) -> list[str]:
    return sorted(ep_to_svc.keys())


def endpoint_id_map(ep_to_svc: dict[str, str]) -> dict[str, int]:
    """endpoint_key -> 整数 id（sorted 顺序，跨运行/跨环境稳定，不依赖 dict 迭代顺序）。"""
    return {ep: i for i, ep in enumerate(_sorted_endpoint_keys(ep_to_svc))}


def id_to_endpoint_key(ep_to_svc: dict[str, str]) -> dict[int, str]:
    """整数 id -> endpoint_key，是 endpoint_id_map() 的精确逆映射。"""
    return {i: ep for i, ep in enumerate(_sorted_endpoint_keys(ep_to_svc))}
