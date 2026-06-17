from typing import Generator
import anthropic
from .base import LLMProvider

MODELS = [
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
        ) as s:
            yield from s.text_stream

    def list_models(self) -> list[str]:
        return MODELS
