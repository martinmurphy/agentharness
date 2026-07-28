"""Construct a Provider for a given name + config, and report provider status.

A provider *name* is either one of the three built-ins (``anthropic``,
``openai``, ``gemini``) or an alias declared in the config's ``providers:``
block with a ``type:`` naming the adapter to use:

    providers:
      lmstudio:
        type: openai
        base_url: http://host.containers.internal:1234/v1
      together:
        type: openai
        base_url: https://api.together.xyz/v1
        api_key_env: TOGETHER_API_KEY

Aliases are what make a *second* endpoint of the same protocol reachable: the
built-in ``openai`` name has exactly one ``base_url``, and a state binds to a
provider name.

One adapter type has no built-in name: ``vertex`` reaches Anthropic's models
through Google Cloud Vertex AI, and cannot work without a project and region to
name, so it only exists as an alias:

    providers:
      vertex:
        type: vertex
        project_id: my-gcp-project
        region: global

Keys are still never written in config. A built-in's SDK reads its own env var;
an alias names the env var to read via ``api_key_env`` and we resolve it here.
Vertex is the exception that proves the rule: it has no API key at all, and
authenticates with Google Application Default Credentials instead.
An endpoint with a ``base_url`` and no key resolvable either way is treated as a
keyless local server and gets a placeholder, because the OpenAI SDK refuses to
construct a client without one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from agentharness.config import Config
from agentharness.providers.anthropic import AnthropicProvider
from agentharness.providers.base import Provider
from agentharness.providers.gemini import GeminiProvider
from agentharness.providers.openai_compatible import OpenAICompatibleProvider

# Built-in provider name -> the env var its SDK reads its key from. Keys are
# never handled by this code for these; we only report whether they exist.
_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
_KNOWN = tuple(_ENV_VARS)

# Adapter types an alias may declare. The built-in names, because a built-in
# name is just the default alias for its own adapter — plus `vertex`, which is
# the Anthropic adapter pointed at Google's endpoint. It has no built-in name
# of its own because it cannot work without a project and region to name.
_ADAPTERS = (*_KNOWN, "vertex")

# Adapters that authenticate with something other than an API key, so the
# key-resolution step below is skipped entirely for them. Vertex uses Google
# Application Default Credentials.
_KEYLESS_ADAPTERS = ("vertex",)

# Sent to a keyless local server: the OpenAI SDK refuses to construct a client
# with no key at all, and such servers ignore whatever they are given.
_PLACEHOLDER_KEY = "not-needed"

# Config keys the harness consumes itself rather than passing to an adapter.
_HARNESS_KEYS = ("type", "api_key_env")


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    env_var: str  # "" when the provider needs no key (keyless local endpoint)
    available: bool  # ready to use: its key is set, or it needs none

    @property
    def keyless(self) -> bool:
        return not self.env_var


def _aliases(config: Config | None) -> dict[str, dict[str, Any]]:
    """Configured provider names that are not built-ins, in config order."""
    if config is None:
        return {}
    return {
        name: dict(block)
        for name, block in config.providers.items()
        if name not in _ENV_VARS and isinstance(block, dict)
    }


def provider_status(config: Config | None = None) -> list[ProviderInfo]:
    """Report each usable provider and whether its key is set.

    State-free: it does not know the active state. Callers that want to mark the
    active provider do so themselves. Without ``config`` only the built-ins are
    reported, since aliases are a config-only concept.

    For a built-in, ``available`` means its env var is set — a heuristic, since
    an OpenAI-compatible provider aimed at a local ``base_url`` needs no key. An
    alias reports the truth directly: keyed aliases track their ``api_key_env``,
    keyless ones are always available.
    """
    infos = [
        ProviderInfo(name=n, env_var=v, available=bool(os.environ.get(v)))
        for n, v in _ENV_VARS.items()
    ]
    for name, block in _aliases(config).items():
        env_var = block.get("api_key_env") or ""
        infos.append(
            ProviderInfo(
                name=name,
                env_var=env_var,
                available=bool(os.environ.get(env_var)) if env_var else True,
            )
        )
    return infos


def known_providers(config: Config | None = None) -> list[str]:
    """Every provider name that can be selected: built-ins plus configured aliases."""
    return [p.name for p in provider_status(config)]


def _adapter_for(name: str, options: dict[str, Any], config: Config | None) -> str:
    """The adapter type backing a provider name."""
    if name in _ENV_VARS:
        return name
    declared = options.get("type")
    if not declared:
        known = ", ".join(known_providers(config))
        raise ValueError(
            f"unknown provider {name!r}; known providers: {known}. "
            f"To add it, declare providers.{name}.type (one of {', '.join(_ADAPTERS)}) "
            f"in the config."
        )
    if declared not in _ADAPTERS:
        raise ValueError(
            f"provider {name!r} declares unknown type {declared!r}; "
            f"choose one of {', '.join(_ADAPTERS)}"
        )
    return declared


def _resolve_api_key(name: str, options: dict[str, Any]) -> None:
    """Put an ``api_key`` into ``options`` if one is needed and resolvable.

    Order: an explicit literal ``api_key`` wins (it exists for placeholder use);
    then ``api_key_env``, read from the environment so the secret never lives in
    the config file; then, for an endpoint with a ``base_url`` whose SDK would
    find no key of its own, the keyless-local placeholder.
    """
    env_var = options.pop("api_key_env", None)
    if options.get("api_key"):
        return
    if env_var:
        value = os.environ.get(env_var)
        if not value:
            raise ValueError(
                f"provider {name!r} reads its key from {env_var}, which is not set"
            )
        options["api_key"] = value
        return
    builtin_env = _ENV_VARS.get(name)
    if builtin_env and os.environ.get(builtin_env):
        return  # the SDK will find its own key
    if options.get("base_url"):
        options["api_key"] = _PLACEHOLDER_KEY


def build_provider(name: str, model: str, config: Config) -> Provider:
    """Build a provider instance bound to ``model``.

    ``name`` selects a built-in adapter or a configured alias; per-provider
    settings (e.g. base_url) come from ``config.providers[name]``.
    """
    options = config.provider_options(name)
    adapter = _adapter_for(name, options, config)
    if adapter not in _KEYLESS_ADAPTERS:
        _resolve_api_key(name, options)
    options = {k: v for k, v in options.items() if k not in _HARNESS_KEYS}

    if adapter in ("anthropic", "vertex"):
        return AnthropicProvider(
            model,
            effort=config.effort,
            show_thinking=config.show_thinking,
            thinking_budget=config.thinking_budget,
            vertex=adapter == "vertex",
            **options,
        )
    if adapter == "openai":
        return OpenAICompatibleProvider(model, **options)
    return GeminiProvider(model, show_thinking=config.show_thinking, **options)
