from __future__ import annotations as _annotations

import re

import httpx
import pytest
from openai import AsyncOpenAI

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai_zai import ZaiModel
from pydantic_ai_zai.profile import zai_model_profile
from pydantic_ai_zai.provider import ZaiProvider


def test_zai_provider():
    provider = ZaiProvider(api_key='api-key')
    assert provider.name == 'zai'
    assert provider.base_url == 'https://api.z.ai/api/paas/v4'
    assert isinstance(provider.client, AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_zai_provider_need_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ZAI_API_KEY', raising=False)
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `ZAI_API_KEY` environment variable or pass it via `ZaiProvider(api_key=...)` '
            'to use the Z.AI provider.'
        ),
    ):
        ZaiProvider()


def test_zai_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = ZaiProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_zai_provider_pass_openai_client() -> None:
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = ZaiProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_zai_provider_model_profile(mocker: pytest.MonkeyPatch):
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = ZaiProvider(openai_client=openai_client)

    profile = provider.model_profile('glm-4.7')
    assert profile is not None
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert profile.openai_chat_thinking_field == 'reasoning_content'
    assert profile.openai_chat_send_back_thinking_parts == 'field'

    profile_air = provider.model_profile('glm-4.5-air')
    assert profile_air is not None
    assert isinstance(profile_air, OpenAIModelProfile)
    assert profile_air.openai_chat_thinking_field == 'reasoning_content'
    assert profile_air.openai_chat_send_back_thinking_parts == 'field'


def test_zai_model_default_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('ZAI_API_KEY', 'test-key')
    model = ZaiModel('glm-4.7')
    assert model.model_name == 'glm-4.7'


def test_zai_profile_thinking_models():
    assert zai_model_profile('glm-4.7') is not None
    assert zai_model_profile('glm-4.6') is not None
    assert zai_model_profile('glm-4.6v') is not None
    assert zai_model_profile('GLM-4.7') is not None
    assert zai_model_profile('glm-4.5-air') is None
    assert zai_model_profile('glm-4.5-air-250723') is None
