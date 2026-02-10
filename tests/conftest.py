from __future__ import annotations as _annotations

import os

import pytest


@pytest.fixture(scope='session')
def zai_api_key() -> str:
    return os.getenv('ZAI_API_KEY', 'mock-api-key')


@pytest.fixture(scope='session')
def anyio_backend():
    return 'asyncio'
