"""验证 8 个外部 endpoint 都有 service 映射，且 service 名符合 Train-Ticket 命名。"""

from pathlib import Path

import yaml

EXPECTED_ENDPOINTS = {
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelplanservice/travelPlan/cheapest",
    "POST:/api/v1/travelplanservice/travelPlan/minStation",
    "POST:/api/v1/travelplanservice/travelPlan/quickest",
    "GET:/api/v1/routeservice/routes",
}


def test_endpoint_mapping_covers_all_v0_endpoints():
    mapping = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    assert set(mapping.keys()) == EXPECTED_ENDPOINTS


def test_endpoint_mapping_services_follow_ts_naming():
    mapping = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    for endpoint, service in mapping.items():
        assert service.startswith("ts-"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"
        assert service.endswith("-service"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"
