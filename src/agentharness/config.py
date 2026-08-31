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
    # Print a per-turn token line after each turn (dim, one line).
    show_usage: bool = True
    max_tool_iterations: int = 10
    # Tool calls from one assistant message that may run at once. 1 is the old
    # sequential behaviour.
    max_concurrency: int = 8
    # Turns that may run in the background (a prompt ending in `&`) at once.
    max_jobs: int = 4
    skills_dir: str = "./skills"
    # Read/write scratch directory the filesystem tools are confined to. Set
    # workspace_writable false to register only the read tools.
    workspace_dir: str = "./workspace"
    workspace_writable: bool = True
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Per-provider settings, e.g. {"openai": {"base_url": "http://localhost:11434/v1"}}
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # MCP servers to connect to, name -> settings block. Held raw here for the
    # same reason as ``providers``: this module knows nothing about what the
    # blocks mean. agentharness.mcp.config parses and validates them.
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Skill script execution. Held raw for the same reason as ``mcp_servers``:
    # this module knows nothing about what the block means.
    # agentharness.skills.runner parses and validates it.
    skill_scripts: dict[str, Any] = field(default_factory=dict)

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


def as_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# Fields that accept a scalar env override, with their coercion function.
_SCALAR_FIELDS: dict[str, Any] = {
    "provider": str,
    "model": str,
    "max_tokens": int,
    "effort": str,
    "thinking_budget": int,
    "show_thinking": as_bool,
    "show_usage": as_bool,
    "max_tool_iterations": int,
    "max_concurrency": int,
    "max_jobs": int,
    "skills_dir": str,
    "workspace_dir": str,
    "workspace_writable": as_bool,
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

    for block in ("providers", "mcp_servers", "skill_scripts"):
        if block in data and isinstance(data[block], dict):
            kwargs[block] = data[block]

    return Config(**kwargs)
