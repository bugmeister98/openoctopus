from typing import Generator
import openai
from .base import LLMProvider

MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3",
    "o4-mini",
]


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = openai.OpenAI()

    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Generator[str, None, None]:
        with self.client.chat.completions.stream(
            model=model,
            max_completion_tokens=max_tokens,
            messages=messages,
        ) as s:
            for event in s:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta

    def list_models(self) -> list[str]:
        return MODELS
