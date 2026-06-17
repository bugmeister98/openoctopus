import os
from .base import LLMProvider
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_provider() -> LLMProvider:
    name = os.getenv("LLM_PROVIDER", "anthropic").lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Choose from: {list(_PROVIDERS)}")
    return cls()
