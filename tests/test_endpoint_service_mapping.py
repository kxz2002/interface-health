"""验证 8 个外部 endpoint 都有 service 映射，且 service 名符合 Train-Ticket 命名。"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[1]

EXPECTED_ENDPOINTS = {
    "GET:/api/v1/assuranceservice/assurances/types",
    "GET:/api/v1/contactservice/contacts/account/{uuid}",
    "POST:/api/v1/inside_pay_service/inside_payment",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/users/login",
}


@pytest.fixture(scope="module")
def mapping():
    return yaml.safe_load((REPO_ROOT / "configs/contract/endpoint_to_service.yaml").read_text())


def test_endpoint_mapping_covers_all_v0_endpoints(mapping):
    assert set(mapping.keys()) == EXPECTED_ENDPOINTS


def test_endpoint_mapping_services_follow_ts_naming(mapping):
    for endpoint, service in mapping.items():
        assert service.startswith("ts-"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"
        assert service.endswith("-service"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"


def test_endpoint_mapping_matches_dataset_guide(mapping):
    """回归防线：endpoint_to_service.yaml 的 key 必须与 dataset-guide §6 的 8 个真实
    client 入口完全一致。任何一方漂移都会在此断裂，防止配置与真实数据口径悄悄背离。"""
    guide = (REPO_ROOT / "docs/agent-docs/dataset-guide.md").read_text()
    section = guide.split("## 六、Endpoint 清单")[1].split("## 七")[0]
    guide_endpoints = {
        line.strip()
        for line in section.splitlines()
        if line.strip().startswith(("GET:", "POST:", "PUT:", "DELETE:", "PATCH:"))
    }
    assert set(mapping.keys()) == guide_endpoints
