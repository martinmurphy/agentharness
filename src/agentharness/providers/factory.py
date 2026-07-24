"""Construct a Provider for a given name + config."""

from __future__ import annotations

from agentharness.config import Config
from agentharness.providers.anthropic import AnthropicProvider
from agentharness.providers.base import Provider
from agentharness.providers.gemini import GeminiProvider
from agentharness.providers.openai_compatible import OpenAICompatibleProvider

_KNOWN = ("anthropic", "openai", "gemini")


def build_provider(name: str, model: str, config: Config) -> Provider:
    """Build a provider instance bound to ``model``.

    ``name`` selects the adapter; per-provider settings (e.g. base_url) come
    from ``config.providers[name]``. API keys are read by the SDKs from their
    own environment variables and are never passed through here.
    """
    options = config.provider_options(name)
    if name == "anthropic":
        return AnthropicProvider(
            model,
            effort=config.effort,
            show_thinking=config.show_thinking,
            **options,
        )
    if name == "openai":
        return OpenAICompatibleProvider(model, **options)
    if name == "gemini":
        return GeminiProvider(model, show_thinking=config.show_thinking, **options)
    raise ValueError(f"unknown provider {name!r}; known providers: {', '.join(_KNOWN)}")
