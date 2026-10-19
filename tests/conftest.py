"""Shared temporary Service artifacts, built once without external datasets."""

import pytest

from tests.service_fixtures import build_frozen_service_chain


@pytest.fixture(scope="session")
def frozen_service_chain(tmp_path_factory):
    return build_frozen_service_chain(tmp_path_factory.mktemp("service-frozen-chain"))
