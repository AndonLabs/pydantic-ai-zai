from __future__ import annotations as _annotations

from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.openai import OpenAIModelProfile


def zai_model_profile(model_name: str) -> ModelProfile | None:
    model_lower = model_name.lower()
    if 'glm-4.7' in model_lower or 'glm-4.6' in model_lower:
        return OpenAIModelProfile(
            openai_chat_thinking_field='reasoning_content',
            openai_chat_send_back_thinking_parts='field',
        )
    return None
