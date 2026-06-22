from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass
class ToolParam:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


class StopReason(str, Enum):
    END_TURN   = "end_turn"
    TOOL_USE   = "tool_use"
    MAX_TOKENS = "max_tokens"
    ERROR      = "error"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@dataclass
class ModelInfo:
    id: str
    context_window: int
    max_output_tokens: int
    supports_tools: bool = True
    supports_vision: bool = False
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0


# ---------------------------------------------------------------------------
# Exception hierarchy — maps provider-specific errors to a common contract
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base for all provider errors."""


class RateLimitError(ProviderError):
    def __init__(self, message: str = "", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ContextWindowExceededError(ProviderError):
    def __init__(self, message: str = "", limit: int = 0, used: int = 0):
        super().__init__(message)
        self.limit = limit
        self.used = used


class AuthenticationError(ProviderError):
    """API key missing, invalid, or expired."""


class ProviderUnavailableError(ProviderError):
    """Transient network failure or provider 5xx. Safe to retry."""
