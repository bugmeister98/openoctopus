from abc import ABC, abstractmethod
from typing import Generator


class LLMProvider(ABC):
    @abstractmethod
    def stream(self, messages: list[dict], model: str, max_tokens: int) -> Generator[str, None, None]:
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        ...
