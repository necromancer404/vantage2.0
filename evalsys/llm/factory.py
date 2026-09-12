from __future__ import annotations

from evalsys.config import LLMSettings
from evalsys.llm.base import LLMProvider
from evalsys.llm.providers import AnthropicProvider, GeminiProvider, MockProvider, OpenAICompatProvider

_PROVIDERS = {
    "openai": OpenAICompatProvider,
    "openai_compat": OpenAICompatProvider,
    "groq": OpenAICompatProvider,
    "openrouter": OpenAICompatProvider,
    "together": OpenAICompatProvider,
    "ollama": OpenAICompatProvider,
    "deepseek": OpenAICompatProvider,
    "nvidia": OpenAICompatProvider,
    "nim": OpenAICompatProvider,
    "cerebras": OpenAICompatProvider,
    "moonshot": OpenAICompatProvider,
    "kimi": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
    "mock": MockProvider,
}


def build_provider(settings: LLMSettings) -> LLMProvider:
    name = settings.provider.strip().lower()
    cls = _PROVIDERS.get(name)
    if cls is None:
        known = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"Unknown LLM provider '{settings.provider}'. Known: {known}")
    return cls(settings)
