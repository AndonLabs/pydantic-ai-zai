from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, cast

from typing_extensions import override

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles import ModelProfileSpec
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings

from .provider import ZaiProvider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Z.AI model, '
        'you can use — `pip install "pydantic-ai-zai"`'
    ) from _import_error

__all__ = ('ZaiModel', 'ZaiModelName', 'ZaiModelSettings')

LatestZaiModelNames = str

ZaiModelName = str


class ZaiModelSettings(ModelSettings, total=False):
    zai_thinking: bool
    zai_clear_thinking: bool


@dataclass(init=False)
class ZaiModel(OpenAIChatModel):
    def __init__(
        self,
        model_name: ZaiModelName,
        *,
        provider: Provider[AsyncOpenAI] | None = None,
        profile: ModelProfileSpec | None = None,
        settings: ZaiModelSettings | None = None,
    ):
        if provider is None:
            provider = ZaiProvider()
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        merged_settings, customized_parameters = super().prepare_request(model_settings, model_request_parameters)
        new_settings = _zai_settings_to_openai_settings(cast(ZaiModelSettings, merged_settings or {}))
        return new_settings, customized_parameters


def _zai_settings_to_openai_settings(model_settings: ZaiModelSettings) -> OpenAIChatModelSettings:
    extra_body = dict(cast(dict[str, Any], model_settings.get('extra_body', {})))

    thinking_enabled = model_settings.get('zai_thinking')
    clear_thinking = model_settings.get('zai_clear_thinking')

    if thinking_enabled is not None:
        thinking: dict[str, Any] = {
            'type': 'enabled' if thinking_enabled else 'disabled',
        }
        if clear_thinking is not None:
            thinking['clear_thinking'] = clear_thinking
        extra_body['thinking'] = thinking

    filtered = {k: v for k, v in model_settings.items() if not k.startswith('zai_')}
    if extra_body:
        filtered['extra_body'] = extra_body

    return OpenAIChatModelSettings(**filtered)  # type: ignore[reportCallIssue]
