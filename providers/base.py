from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Generator

from core.models import ModelInfo, ProviderResponse, ToolParam


class LLMProvider(ABC):
    # ------------------------------------------------------------------
    # Sync streaming — kept for backward compat with api/chat.py
    # ------------------------------------------------------------------

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> Generator[str, None, None]:
        ...

    # ------------------------------------------------------------------
    # Async interface — used by agents and orchestrator
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        tools: list[ToolParam] | None = None,
        tool_choice: str = "auto",
    ) -> ProviderResponse:
        """
        Non-streaming structured completion.

        tool_choice: "auto" | "any" | "<tool-name>"
        Providers map this to their own internal format.

        Raises: RateLimitError, ContextWindowExceededError,
                AuthenticationError, ProviderUnavailableError
        """
        ...

    @abstractmethod
    async def stream_async(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> AsyncGenerator[str, None]:
        """Async streaming variant — for agents that need live output."""
        ...

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        """Returns metadata for all models this provider exposes."""
        ...

    @abstractmethod
    async def count_tokens(
        self,
        messages: list[dict],
        model: str,
        tools: list[ToolParam] | None = None,
    ) -> int:
        """
        Count tokens for the prompt without sending a completion request.
        May raise NotImplementedError — callers must handle with a fallback.
        """
        ...
