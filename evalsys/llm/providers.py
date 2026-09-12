from __future__ import annotations

import os
from typing import Any

import httpx

from evalsys.config import LLMSettings
from evalsys.llm.base import LLMError, LLMProvider, LLMResponse


def _resolve_base_url(settings: LLMSettings, fallback: str) -> str:
    if settings.base_url_env:
        env_url = os.getenv(settings.base_url_env)
        if env_url:
            return env_url.rstrip("/")
    if settings.base_url:
        return settings.base_url.rstrip("/")
    return fallback.rstrip("/")


def _api_key(settings: LLMSettings) -> str:
    if not settings.api_key_env:
        return ""
    return os.getenv(settings.api_key_env, "")


def _client_timeout(settings: LLMSettings) -> httpx.Timeout:
    return httpx.Timeout(
        connect=30.0,
        read=settings.timeout_seconds,
        write=30.0,
        pool=30.0,
    )


def _http_error(prefix: str, exc: httpx.HTTPError) -> LLMError:
    detail = str(exc)
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        body = (exc.response.text or "")[:800]
        detail = f"{exc.response.status_code} {body or exc}"
    return LLMError(f"{prefix}: {detail}")


def extract_chat_text(data: dict[str, Any]) -> str:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected openai_compat payload: {data}") from exc

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                chunks.append(str(part.get("text") or part.get("content") or ""))
        joined = "".join(chunks).strip()
        if joined:
            return joined
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if isinstance(content, str):
        return content
    raise LLMError(f"Empty openai_compat message content: {data}")


def extract_gemini_text(data: dict[str, Any]) -> str:
    try:
        parts = data["candidates"][0]["content"].get("parts") or []
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"Unexpected gemini payload: {data}") from exc
    text = "".join(str(part.get("text") or "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise LLMError(f"Empty gemini payload: {data}")
    return text


class OpenAICompatProvider(LLMProvider):
    provider_name = "openai_compat"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.provider_name = settings.provider
        self.base_url = _resolve_base_url(settings, "https://api.openai.com/v1")

    def complete(self, *, system: str, user: str) -> LLMResponse:
        key = _api_key(self.settings)
        if not key:
            raise LLMError(f"Missing API key in {self.settings.api_key_env}")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            **self.settings.extra_headers,
        }
        body: dict[str, Any] = {
            "model": self.settings.model,
            self.settings.token_field: self.settings.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.settings.send_temperature:
            body["temperature"] = self.settings.temperature
        if self.settings.extra_body:
            body.update(self.settings.extra_body)
        try:
            with httpx.Client(timeout=_client_timeout(self.settings)) as client:
                print("[DEBUG] REQUEST BODY:")
                print(body)
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise _http_error("openai_compat request failed", exc) from exc
        text = extract_chat_text(data)
        return LLMResponse(text=text, model=self.settings.model, provider=self.provider_name)


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.base_url = _resolve_base_url(settings, "https://api.anthropic.com")

    def complete(self, *, system: str, user: str) -> LLMResponse:
        key = _api_key(self.settings)
        if not key:
            raise LLMError(f"Missing API key in {self.settings.api_key_env}")
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            **self.settings.extra_headers,
        }
        body = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.settings.send_temperature:
            body["temperature"] = self.settings.temperature
        if self.settings.extra_body:
            body.update(self.settings.extra_body)
        try:
            with httpx.Client(timeout=_client_timeout(self.settings)) as client:
                response = client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise _http_error("anthropic request failed", exc) from exc

        parts = data.get("content") or []
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not text:
            raise LLMError(f"Unexpected anthropic payload: {data}")
        return LLMResponse(text=text, model=self.settings.model, provider=self.provider_name)


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.base_url = _resolve_base_url(settings, "https://generativelanguage.googleapis.com/v1beta")

    def complete(self, *, system: str, user: str) -> LLMResponse:
        key = _api_key(self.settings)
        if not key:
            raise LLMError(f"Missing API key in {self.settings.api_key_env}")
        url = f"{self.base_url}/models/{self.settings.model}:generateContent"
        params = {"key": key}
        generation: dict[str, Any] = {
            "maxOutputTokens": self.settings.max_tokens,
            "responseMimeType": "application/json",
        }
        if self.settings.send_temperature:
            generation["temperature"] = self.settings.temperature
        body: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        if self.settings.extra_body:
            body.update(self.settings.extra_body)
        try:
            with httpx.Client(timeout=_client_timeout(self.settings)) as client:
                response = client.post(url, params=params, json=body, headers=self.settings.extra_headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:
            raise _http_error("gemini request failed", exc) from exc
        text = extract_gemini_text(data)
        return LLMResponse(text=text, model=self.settings.model, provider=self.provider_name)


class MockProvider(LLMProvider):
    provider_name = "mock"

    def __init__(self, settings: LLMSettings, canned: str | None = None) -> None:
        self.settings = settings
        self.canned = canned

    def complete(self, *, system: str, user: str) -> LLMResponse:
        if self.canned:
            text = self.canned
        else:
            score = 40 if "[EMPTY SUBMISSION]" in user else 72
            text = (
                '{"score": %s, "confidence": 0.8, "grade": "B", '
                '"summary": "Mock evaluation.", "findings": ["mock"]}' % score
            )
        return LLMResponse(text=text, model=self.settings.model, provider=self.provider_name)
