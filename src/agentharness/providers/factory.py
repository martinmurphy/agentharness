"""Construct a Provider for a given name + config, and report provider status."""

from __future__ import annotations

import os
from dataclasses import dataclass

from agentharness.config import Config
from agentharness.providers.anthropic import AnthropicProvider
from agentharness.providers.base import Provider
from agentharness.providers.gemini import GeminiProvider
from agentharness.providers.openai_compatible import OpenAICompatibleProvider

# Single source of truth: provider name -> the env var its SDK reads its key
# from. Keys are never handled by this code; we only report whether they exist.
_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_KNOWN = tuple(_ENV_VARS)


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    env_var: str
    available: bool  # env_var is set in the environment


def provider_status() -> list[ProviderInfo]:
    """Report each supported provider and whether its API key is set.

    State-free: it does not know the active state. Callers that want to mark the
    active provider do so themselves. ``available`` is a key-set heuristic — an
    OpenAI-compatible provider aimed at a local base_url needs no key, so
    ``available=False`` there does not imply "unusable".
    """
    return [
        ProviderInfo(name=n, env_var=v, available=bool(os.environ.get(v)))
        for n, v in _ENV_VARS.items()
    ]


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
            thinking_budget=config.thinking_budget,
            **options,
        )
    if name == "openai":
        return OpenAICompatibleProvider(model, **options)
    if name == "gemini":
        return GeminiProvider(model, show_thinking=config.show_thinking, **options)
    raise ValueError(f"unknown provider {name!r}; known providers: {', '.join(_KNOWN)}")
