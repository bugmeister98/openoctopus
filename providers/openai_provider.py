from __future__ import annotations
import json
from typing import AsyncGenerator, Generator

import openai

from .base import LLMProvider
from core.models import (
    AuthenticationError,
    ContextWindowExceededError,
    ModelInfo,
    ProviderError,
    ProviderResponse,
    ProviderUnavailableError,
    RateLimitError,
    StopReason,
    ToolCall,
    ToolParam,
)

MODELS: list[ModelInfo] = [
    ModelInfo(id="gpt-4o",      context_window=128_000, max_output_tokens=16_384, supports_tools=True),
    ModelInfo(id="gpt-4o-mini", context_window=128_000, max_output_tokens=16_384, supports_tools=True),
    ModelInfo(id="o3",          context_window=200_000, max_output_tokens=100_000, supports_tools=True),
    ModelInfo(id="o4-mini",     context_window=200_000, max_output_tokens=100_000, supports_tools=True),
]

_STOP_REASON_MAP = {
    "stop":       StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "length":     StopReason.MAX_TOKENS,
}


def _translate_stop_reason(raw: str | None) -> StopReason:
    return _STOP_REASON_MAP.get(raw or "", StopReason.END_TURN)


def _wrap_error(exc: openai.APIError) -> ProviderError:
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, openai.BadRequestError) and "maximum context length" in str(exc).lower():
        return ContextWindowExceededError(str(exc))
    if isinstance(exc, openai.AuthenticationError):
        return AuthenticationError(str(exc))
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
        return ProviderUnavailableError(str(exc))
    return ProviderUnavailableError(str(exc))


class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = openai.OpenAI()
        self.async_client = openai.AsyncOpenAI()

    def stream(
        self, messages: list[dict], model: str, max_tokens: int
    ) -> Generator[str, None, None]:
        with self.client.chat.completions.stream(
            model=model, max_completion_tokens=max_tokens, messages=messages
        ) as s:
            for event in s:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta

    async def complete(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        tools: list[ToolParam] | None = None,
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        kwargs: dict = dict(model=model, max_completion_tokens=max_tokens, messages=messages)
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }}
                for t in tools
            ]
            kwargs["tool_choice"] = tool_choice

        try:
            resp = await self.async_client.chat.completions.create(**kwargs)
        except openai.APIError as exc:
            raise _wrap_error(exc) from exc

        choice = resp.choices[0]
        calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return ProviderResponse(
            content=choice.message.content or "",
            tool_calls=calls,
            stop_reason=_translate_stop_reason(choice.finish_reason),
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=resp.model,
        )

    async def stream_async(
        self, messages: list[dict], model: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        async with self.async_client.chat.completions.stream(
            model=model, max_completion_tokens=max_tokens, messages=messages
        ) as s:
            async for event in s:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta

    def list_models(self) -> list[ModelInfo]:
        return MODELS

    async def count_tokens(
        self,
        messages: list[dict],
        model: str,
        tools: list[ToolParam] | None = None,
    ) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model)
            total = 0
            for msg in messages:
                total += 4  # per-message overhead
                for v in msg.values():
                    total += len(enc.encode(str(v)))
            return total
        except ImportError:
            raise NotImplementedError(
                "Install tiktoken for token counting with OpenAI models: pip install tiktoken"
            )
