from __future__ import annotations as _annotations

import os
from typing import overload

import httpx
from openai import AsyncOpenAI

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.providers import Provider

from .profile import zai_model_profile


class ZaiProvider(Provider[AsyncOpenAI]):
    @property
    def name(self) -> str:
        return 'zai'

    @property
    def base_url(self) -> str:
        return 'https://api.z.ai/api/paas/v4'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        profile = zai_model_profile(model_name)

        return OpenAIModelProfile(
            json_schema_transformer=OpenAIJsonSchemaTransformer,
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        ).update(profile)

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, *, api_key: str) -> None: ...

    @overload
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, http_client: httpx.AsyncClient) -> None: ...

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...

    def __init__(
        self,
        *,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        api_key = api_key or os.getenv('ZAI_API_KEY')
        if not api_key and openai_client is None:
            raise UserError(
                'Set the `ZAI_API_KEY` environment variable or pass it via `ZaiProvider(api_key=...)` '
                'to use the Z.AI provider.'
            )

        if openai_client is not None:
            self._client = openai_client
        elif http_client is not None:
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
        else:
            http_client = cached_async_http_client(provider='zai')
            self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
