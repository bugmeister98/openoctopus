from __future__ import annotations
from typing import AsyncGenerator, Generator

import anthropic

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
    ModelInfo(
        id="claude-opus-4-8",
        context_window=200_000,
        max_output_tokens=32_000,
        supports_tools=True,
        input_cost_per_mtok=15.0,
        output_cost_per_mtok=75.0,
    ),
    ModelInfo(
        id="claude-sonnet-4-6",
        context_window=200_000,
        max_output_tokens=16_000,
        supports_tools=True,
        input_cost_per_mtok=3.0,
        output_cost_per_mtok=15.0,
    ),
    ModelInfo(
        id="claude-haiku-4-5-20251001",
        context_window=200_000,
        max_output_tokens=8_096,
        supports_tools=True,
        input_cost_per_mtok=0.8,
        output_cost_per_mtok=4.0,
    ),
]

_STOP_REASON_MAP = {
    "end_turn":   StopReason.END_TURN,
    "tool_use":   StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
}


def _translate_stop_reason(raw: str | None) -> StopReason:
    return _STOP_REASON_MAP.get(raw or "", StopReason.END_TURN)


def _wrap_error(exc: anthropic.APIError) -> ProviderError:
    if isinstance(exc, anthropic.RateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, anthropic.BadRequestError) and "prompt is too long" in str(exc).lower():
        return ContextWindowExceededError(str(exc))
    if isinstance(exc, anthropic.AuthenticationError):
        return AuthenticationError(str(exc))
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.InternalServerError)):
        return ProviderUnavailableError(str(exc))
    return ProviderUnavailableError(str(exc))


class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()

    def stream(
        self, messages: list[dict], model: str, max_tokens: int
    ) -> Generator[str, None, None]:
        with self.client.messages.stream(
            model=model, max_tokens=max_tokens, messages=messages
        ) as s:
            yield from s.text_stream

    async def complete(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        tools: list[ToolParam] | None = None,
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
            kwargs["tool_choice"] = {"type": tool_choice}

        try:
            resp = await self.async_client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise _wrap_error(exc) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return ProviderResponse(
            content="".join(text_parts),
            tool_calls=calls,
            stop_reason=_translate_stop_reason(resp.stop_reason),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
        )

    async def stream_async(
        self, messages: list[dict], model: str, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        async with self.async_client.messages.stream(
            model=model, max_tokens=max_tokens, messages=messages
        ) as s:
            async for chunk in s.text_stream:
                yield chunk

    def list_models(self) -> list[ModelInfo]:
        return MODELS

    async def count_tokens(
        self,
        messages: list[dict],
        model: str,
        tools: list[ToolParam] | None = None,
    ) -> int:
        kwargs: dict = dict(model=model, messages=messages)
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        resp = await self.async_client.messages.count_tokens(**kwargs)
        return resp.input_tokens
