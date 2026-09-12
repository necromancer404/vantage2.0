from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMError(RuntimeError):
    pass


class LLMProvider:
    provider_name = "base"

    def complete(self, *, system: str, user: str) -> LLMResponse:
        raise NotImplementedError
