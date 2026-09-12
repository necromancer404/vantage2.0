from __future__ import annotations

from typing import Any

import httpx

from evalsys.config import LLMSettings
from evalsys.llm.base import LLMError, LLMProvider, LLMResponse


def _client_timeout(settings: LLMSettings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=30.0,
        read=settings.timeout_seconds,
        write=30.0,
        pool=30.0,
    )


def _resolve_base_url(settings: LLMSettings) -> str:
    if settings.base_url_env:
        import os

        env_url = os.getenv(settings.base_url_env)
        if env_url:
            return env_url.rstrip("/")

    if settings.base_url:
        return settings.base_url.rstrip("/")

    return "http://localhost:11434"


def extract_ollama_text(data: dict[str, Any]) -> str:
    try:
        message = data["message"]
        content = message["content"]
    except (KeyError, TypeError) as exc:
        raise LLMError(f"Unexpected Ollama payload: {data}") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMError(f"Empty Ollama response: {data}")

    return content.strip()


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.base_url = _resolve_base_url(settings)

    def complete(self, *, system: str, user: str) -> LLMResponse:
        url = f"{self.base_url}/api/chat"

        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }

        if self.settings.extra_body:
            body.update(self.settings.extra_body)

        try:
            with httpx.Client(timeout=_client_timeout(self.settings)) as client:
                response = client.post(
                    url,
                    json=body,
                    headers=self.settings.extra_headers,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPError as exc:
            raise LLMError(
                f"ollama request failed: {exc}"
            ) from exc

        text = extract_ollama_text(data)

        return LLMResponse(
            text=text,
            model=self.settings.model,
            provider=self.provider_name,
        )