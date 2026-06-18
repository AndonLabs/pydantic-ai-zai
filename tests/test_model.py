from __future__ import annotations as _annotations

import pytest

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai_zai import ZaiModel, ZaiModelSettings
from pydantic_ai_zai.provider import ZaiProvider

pytestmark = pytest.mark.anyio


def _model(model_name: str = 'glm-4.7') -> ZaiModel:
    return ZaiModel(model_name, provider=ZaiProvider(api_key='mock-api-key'))


def _prepared_settings(model: ZaiModel, settings: ZaiModelSettings) -> object:
    merged_settings, _ = model.prepare_request(settings, ModelRequestParameters())
    return merged_settings


async def test_thinking_enabled():
    assert _prepared_settings(_model(), ZaiModelSettings(thinking=True)) == {
        'extra_body': {'thinking': {'type': 'enabled'}}
    }


async def test_thinking_disabled():
    assert _prepared_settings(_model(), ZaiModelSettings(thinking=False)) == {
        'extra_body': {'thinking': {'type': 'disabled'}}
    }


async def test_thinking_omitted():
    assert _prepared_settings(_model(), ZaiModelSettings()) == {}


async def test_thinking_stripped_on_non_thinking_model():
    assert _prepared_settings(_model('glm-4-32b-0414-128k'), ZaiModelSettings(thinking=True)) == {}


@pytest.mark.parametrize('effort', ['minimal', 'low', 'medium', 'high', 'xhigh'])
async def test_thinking_effort_collapses_to_enabled_on_older_model(effort: str):
    assert _prepared_settings(_model('glm-4.7'), ZaiModelSettings(thinking=effort)) == {  # type: ignore[typeddict-item]
        'extra_body': {'thinking': {'type': 'enabled'}}
    }


@pytest.mark.parametrize('effort', ['minimal', 'low', 'medium', 'high', 'xhigh'])
async def test_thinking_effort_sets_reasoning_effort_on_glm_5_2(effort: str):
    assert _prepared_settings(_model('glm-5.2'), ZaiModelSettings(thinking=effort)) == {  # type: ignore[typeddict-item]
        'extra_body': {'thinking': {'type': 'enabled'}, 'reasoning_effort': effort}
    }


async def test_thinking_true_omits_reasoning_effort_on_glm_5_2():
    assert _prepared_settings(_model('glm-5.2'), ZaiModelSettings(thinking=True)) == {
        'extra_body': {'thinking': {'type': 'enabled'}}
    }


async def test_preserved_thinking():
    assert _prepared_settings(_model(), ZaiModelSettings(thinking=True, zai_clear_thinking=False)) == {
        'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': False}}
    }


async def test_clear_thinking():
    assert _prepared_settings(_model(), ZaiModelSettings(thinking=True, zai_clear_thinking=True)) == {
        'extra_body': {'thinking': {'type': 'enabled', 'clear_thinking': True}}
    }


async def test_preserves_existing_extra_body():
    settings = ZaiModelSettings(thinking=True, extra_body={'custom_key': 'value'})
    assert _prepared_settings(_model(), settings) == {
        'extra_body': {'custom_key': 'value', 'thinking': {'type': 'enabled'}}
    }
