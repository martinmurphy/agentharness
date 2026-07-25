"""Configuration: a dataclass loaded from YAML with environment overrides.

Resolution order for the config file:
    $AGENTHARNESS_CONFIG  ->  /config/config.yaml  ->  ./config.yaml

Any key may be overridden by an ``AGENTHARNESS_<KEY>`` environment variable.
API keys are read from the provider SDKs' own env vars (ANTHROPIC_API_KEY,
OPENAI_API_KEY) and are deliberately never stored here or in the config file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running inside the agentharness CLI. "
    "You have tools available, including tools to load Agent Skills on demand. "
    "When a task matches a skill's description, use read_skill to load its full "
    "instructions before proceeding."
)

_CONFIG_SEARCH = ("/config/config.yaml", "./config.yaml")


@dataclass
class Config:
    provider: str = "anthropic"
    model: str = "claude-opus-4-8"
    max_tokens: int = 16000
    effort: str = "high"
    # Fixed thinking budget (tokens) for Anthropic models that don't support
    # adaptive thinking (Haiku 4.5, Sonnet 4.5, …). None = no thinking on those
    # models. Ignored on current models, which always use adaptive thinking.
    thinking_budget: int | None = None
    show_thinking: bool = False
    max_tool_iterations: int = 10
    skills_dir: str = "./skills"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Per-provider settings, e.g. {"openai": {"base_url": "http://localhost:11434/v1"}}
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)

    def provider_options(self, name: str) -> dict[str, Any]:
        """Return the settings block for a named provider (empty dict if none)."""
        return dict(self.providers.get(name, {}))


def _config_path() -> Path | None:
    env = os.environ.get("AGENTHARNESS_CONFIG")
    if env:
        return Path(env)
    for candidate in _CONFIG_SEARCH:
        p = Path(candidate)
        if p.is_file():
            return p
    return None


# Fields that accept a scalar env override, with their coercion function.
_SCALAR_FIELDS: dict[str, Any] = {
    "provider": str,
    "model": str,
    "max_tokens": int,
    "effort": str,
    "thinking_budget": int,
    "show_thinking": lambda v: str(v).strip().lower() in ("1", "true", "yes", "on"),
    "max_tool_iterations": int,
    "skills_dir": str,
    "system_prompt": str,
}


def load_config() -> Config:
    """Load config from the resolved YAML file, then apply env overrides."""
    data: dict[str, Any] = {}
    path = _config_path()
    if path is not None and path.is_file():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {path} must contain a YAML mapping")
        data = loaded

    kwargs: dict[str, Any] = {}
    for name, coerce in _SCALAR_FIELDS.items():
        env_key = f"AGENTHARNESS_{name.upper()}"
        if env_key in os.environ:
            kwargs[name] = coerce(os.environ[env_key])
        elif name in data:
            kwargs[name] = data[name]

    if "providers" in data and isinstance(data["providers"], dict):
        kwargs["providers"] = data["providers"]

    return Config(**kwargs)
