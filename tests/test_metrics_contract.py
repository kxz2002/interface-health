"""tests/test_metrics_contract.py — metrics contract v0 校验器测试。

与 scores contract 测试同理：每条 case 锁定 metrics_v0.py 文档里的一条规则。
重点覆盖 has_labels 与 auroc/auprc 的一致性约束——这是 v0 最容易写错的地方。
"""

from __future__ import annotations

import pytest

from src.contracts.metrics_v0 import MetricsContractError, validate_metrics_dict


def _valid_with_labels() -> dict:
    """有标签场景：auroc/auprc 必须是 [0,1] 数值。"""
    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": 700,
        "has_labels": True,
        "auroc": 0.95,
        "auprc": 0.88,
    }


def _valid_without_labels() -> dict:
    """无标签场景：auroc/auprc 必须为 None。"""
    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": 700,
        "has_labels": False,
        "auroc": None,
        "auprc": None,
    }


# ---- 正向 cases ----


def test_valid_dict_with_labels_passes():
    validate_metrics_dict(_valid_with_labels())


def test_valid_dict_without_labels_passes():
    validate_metrics_dict(_valid_without_labels())


def test_boundary_auroc_values_pass():
    # 0.0 和 1.0 是合法边界
    d = _valid_with_labels()
    d["auroc"] = 0.0
    d["auprc"] = 1.0
    validate_metrics_dict(d)


def test_with_extra_fields_passes():
    d = _valid_with_labels()
    d["score_stats"] = {"mean": 1.2, "std": 0.3}
    d["meta"] = {"git_commit": "abc123", "timestamp_utc": "2026-06-23T06:55:47+00:00"}
    validate_metrics_dict(d)


# ---- 结构性失败 ----


def test_not_dict_raises():
    with pytest.raises(MetricsContractError, match="dict"):
        validate_metrics_dict([("protocol_version", "v0")])


@pytest.mark.parametrize(
    "missing_field",
    ["protocol_version", "higher_is_more_anomalous", "n_samples", "has_labels", "auroc", "auprc"],
)
def test_missing_required_field_raises(missing_field):
    d = _valid_with_labels()
    del d[missing_field]
    with pytest.raises(MetricsContractError, match="missing required fields"):
        validate_metrics_dict(d)


# ---- 字段取值失败 ----


def test_wrong_protocol_version_raises():
    d = _valid_with_labels()
    d["protocol_version"] = "v1"
    with pytest.raises(MetricsContractError, match="protocol_version"):
        validate_metrics_dict(d)


def test_higher_is_more_anomalous_false_raises():
    # v0 固定 True
    d = _valid_with_labels()
    d["higher_is_more_anomalous"] = False
    with pytest.raises(MetricsContractError, match="higher_is_more_anomalous"):
        validate_metrics_dict(d)


def test_n_samples_zero_raises():
    d = _valid_with_labels()
    d["n_samples"] = 0
    with pytest.raises(MetricsContractError, match="n_samples"):
        validate_metrics_dict(d)


def test_n_samples_negative_raises():
    d = _valid_with_labels()
    d["n_samples"] = -5
    with pytest.raises(MetricsContractError, match="n_samples"):
        validate_metrics_dict(d)


def test_n_samples_bool_raises():
    # bool 是 int 子类，但语义上不是合法样本数
    d = _valid_with_labels()
    d["n_samples"] = True
    with pytest.raises(MetricsContractError, match="n_samples"):
        validate_metrics_dict(d)


# ---- has_labels 与 auroc/auprc 一致性 ----


def test_auroc_out_of_range_raises():
    d = _valid_with_labels()
    d["auroc"] = 1.5
    with pytest.raises(MetricsContractError, match="auroc"):
        validate_metrics_dict(d)


def test_auprc_negative_raises():
    d = _valid_with_labels()
    d["auprc"] = -0.1
    with pytest.raises(MetricsContractError, match="auprc"):
        validate_metrics_dict(d)


def test_auroc_none_with_labels_raises():
    # has_labels=True 时 auroc 不能是 None
    d = _valid_with_labels()
    d["auroc"] = None
    with pytest.raises(MetricsContractError, match="auroc"):
        validate_metrics_dict(d)


def test_auroc_bool_with_labels_raises():
    # bool 不应被当成合法 [0,1] 数值
    d = _valid_with_labels()
    d["auroc"] = True
    with pytest.raises(MetricsContractError, match="auroc"):
        validate_metrics_dict(d)


def test_auroc_not_none_without_labels_raises():
    # has_labels=False 时 auroc 必须 None
    d = _valid_without_labels()
    d["auroc"] = 0.5
    with pytest.raises(MetricsContractError, match="auroc"):
        validate_metrics_dict(d)


def test_multiple_violations_reported_together():
    """所有违规应一次性聚合报告，而不是遇到第一个就 raise。"""
    d = {
        "protocol_version": "v1",  # 错误：应为 v0
        "higher_is_more_anomalous": True,
        "n_samples": -5,  # 错误：必须正整数
        "has_labels": True,
        "auroc": 1.5,  # 错误：超出 [0,1]
        "auprc": 0.88,
    }
    with pytest.raises(MetricsContractError) as exc_info:
        validate_metrics_dict(d)
    msg = str(exc_info.value)
    assert "protocol_version" in msg
    assert "n_samples" in msg
    assert "auroc" in msg
