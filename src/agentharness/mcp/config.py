"""Parse and validate the ``mcp_servers:`` config block.

The block is a registry of named servers, the same shape ``providers:`` uses:

    mcp_servers:
      fs:
        type: stdio
        command: uvx
        args: [mcp-server-filesystem, /workspace]
        env_pass: [HOME]
      github:
        type: http
        url: https://api.githubcopilot.com/mcp/
        token_env: GITHUB_MCP_TOKEN

Structure is validated here, at load, so a typo is a clear error naming the
server and the key rather than a failure inside an event-loop thread later.

Secrets are not. ``token_env`` and ``${VAR}`` in ``headers`` are resolved at
*connect* time (``resolve_headers``), because a missing environment variable
should take down one server with a useful message, not the whole REPL — and
because a resolved token would otherwise sit in the Config object for the life
of the process. Keys never live in the config file; this is the same rule the
provider factory documents.

Security note: a ``stdio`` server entry is arbitrary local command execution
declared in a config file. That is inherent to the transport — the harness runs
what the config names, with the environment the config names.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from agentharness.config import Config

Transport = Literal["stdio", "http"]
AuthMode = Literal["none", "bearer", "oauth"]

_TRANSPORTS: tuple[Transport, ...] = ("stdio", "http")
_AUTH_MODES: tuple[AuthMode, ...] = ("none", "bearer", "oauth")

DEFAULT_TIMEOUT = 15.0

# A person has to read a consent screen and click. Configurable per server via
# oauth_timeout, because five minutes is generous for a returning user and
# short for one who has to find a password manager first.
DEFAULT_OAUTH_TIMEOUT = 300.0

# Keys accepted in a server block, by transport. Unknown keys are an error
# rather than ignored: there is no adapter downstream to absorb them, so a typo
# would otherwise be a silently missing setting.
_COMMON_KEYS = frozenset({"type", "enabled", "timeout", "tools"})
_STDIO_KEYS = frozenset({"command", "args", "env_pass", "cwd"})
_HTTP_KEYS = frozenset(
    {"url", "headers", "token_env", "auth", "callback_port", "oauth_timeout"}
)

# ${VAR} in a header value: an environment lookup, never a literal secret.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class McpConfigError(ValueError):
    """A malformed ``mcp_servers`` entry. Raised at config load."""


@dataclass(frozen=True)
class McpServerConfig:
    """One validated server entry.

    Holds the *declaration*, not resolved credentials: ``headers`` may still
    contain ``${VAR}`` references and ``token_env`` is a variable name. Call
    ``resolve_headers`` when connecting.
    """

    name: str
    transport: Transport
    enabled: bool = True
    timeout: float = DEFAULT_TIMEOUT
    # None = advertise every tool the server offers; a tuple = allowlist.
    tools: tuple[str, ...] | None = None
    # stdio
    command: str = ""
    args: tuple[str, ...] = ()
    env_pass: tuple[str, ...] = ()
    cwd: str = ""
    # http
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    token_env: str = ""
    auth: AuthMode = "none"
    # Loopback port the OAuth redirect comes back on. 0 = pick a free one,
    # which is fine unless the provider requires a pre-registered redirect URI.
    callback_port: int = 0
    # How long the person at the keyboard has to complete the browser flow.
    oauth_timeout: float = DEFAULT_OAUTH_TIMEOUT

    @property
    def is_stdio(self) -> bool:
        return self.transport == "stdio"

    @property
    def handshake_timeout(self) -> float:
        """How long connecting may take, which is not how long a call may take.

        An OAuth server's first connect waits for a person to read a consent
        screen and click, so ``timeout`` — sized for a machine answering a
        machine — would abort the flow mid-way. The two budgets add: the human
        one, plus the ordinary network time either side of it.
        """
        if self.auth != "oauth":
            return self.timeout
        return self.oauth_timeout + self.timeout

    def allows(self, tool_name: str) -> bool:
        """Whether a tool the server offers should be advertised to the model."""
        return self.tools is None or tool_name in self.tools

    def resolve_headers(self) -> dict[str, str]:
        """Headers with ``${VAR}`` expanded and any bearer token attached.

        Raises ValueError naming the server and the variable when one is unset,
        so the failure is attributable to a single server.
        """
        resolved: dict[str, str] = {}
        for key, value in self.headers.items():
            resolved[key] = _ENV_REF.sub(lambda m: self._env(m.group(1)), value)
        if self.token_env:
            resolved.setdefault("Authorization", f"Bearer {self._env(self.token_env)}")
        return resolved

    def extra_env(self) -> dict[str, str]:
        """The named variables to add to a stdio server's environment.

        These are *additions* to the small safe set the transport inherits by
        default (PATH, HOME, SHELL, …) — a server launched as ``uvx …`` needs
        PATH to exist at all, so an allowlist that replaced the environment
        outright would break every real server while looking strict.

        What ``env_pass`` controls is therefore the interesting part: nothing
        beyond that safe set reaches the subprocess unless it is named here, so
        a server is handed the credentials it was granted and no others. Unset
        names are skipped rather than fatal.
        """
        return {n: os.environ[n] for n in self.env_pass if n in os.environ}

    def _env(self, var: str) -> str:
        value = os.environ.get(var)
        if not value:
            raise ValueError(
                f"mcp server {self.name!r} reads {var}, which is not set"
            )
        return value


def parse_servers(config: Config) -> list[McpServerConfig]:
    """Validate every entry in ``mcp_servers``, in config order.

    Disabled servers are returned too — ``/mcp`` should be able to show that a
    server exists but is switched off.
    """
    return [_parse_one(name, block) for name, block in config.mcp_servers.items()]


def _parse_one(name: str, block: Any) -> McpServerConfig:
    if not isinstance(block, dict):
        raise McpConfigError(f"mcp server {name!r}: expected a mapping of settings")

    transport = block.get("type")
    if not transport:
        raise McpConfigError(
            f"mcp server {name!r}: 'type' is required (one of {', '.join(_TRANSPORTS)})"
        )
    if transport not in _TRANSPORTS:
        raise McpConfigError(
            f"mcp server {name!r}: unknown type {transport!r}; "
            f"choose one of {', '.join(_TRANSPORTS)}"
        )

    allowed = _COMMON_KEYS | (_STDIO_KEYS if transport == "stdio" else _HTTP_KEYS)
    for key in block:
        if key not in allowed:
            raise McpConfigError(
                f"mcp server {name!r}: unknown key {key!r} for a {transport} server; "
                f"accepted: {', '.join(sorted(allowed))}"
            )

    common = {
        "name": name,
        "transport": transport,
        "enabled": _bool(name, block, "enabled", default=True),
        "timeout": _timeout(name, block),
        # Key absent = no allowlist. An explicit empty list is an empty
        # allowlist (advertise nothing), which is not the same thing.
        "tools": _str_tuple(name, block, "tools") if "tools" in block else None,
    }
    if transport == "stdio":
        return McpServerConfig(**common, **_stdio_fields(name, block))
    return McpServerConfig(**common, **_http_fields(name, block))


def _stdio_fields(name: str, block: dict[str, Any]) -> dict[str, Any]:
    command = block.get("command")
    if not isinstance(command, str) or not command.strip():
        raise McpConfigError(
            f"mcp server {name!r}: 'command' is required for a stdio server"
        )
    cwd = block.get("cwd", "")
    if not isinstance(cwd, str):
        raise McpConfigError(f"mcp server {name!r}: 'cwd' must be a string")
    return {
        "command": command,
        "args": _str_tuple(name, block, "args"),
        "env_pass": _str_tuple(name, block, "env_pass"),
        "cwd": cwd,
    }


def _http_fields(name: str, block: dict[str, Any]) -> dict[str, Any]:
    url = block.get("url")
    if not isinstance(url, str) or not url.strip():
        raise McpConfigError(f"mcp server {name!r}: 'url' is required for an http server")
    scheme = urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise McpConfigError(
            f"mcp server {name!r}: url must be http or https, got {scheme or 'no scheme'!r}"
        )

    headers = block.get("headers", {})
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise McpConfigError(f"mcp server {name!r}: 'headers' must be a mapping of strings")

    token_env = block.get("token_env", "")
    if not isinstance(token_env, str):
        raise McpConfigError(f"mcp server {name!r}: 'token_env' must be a variable name")

    # Default the auth mode from what was configured: a token_env means bearer.
    auth = block.get("auth") or ("bearer" if token_env else "none")
    if auth not in _AUTH_MODES:
        raise McpConfigError(
            f"mcp server {name!r}: unknown auth {auth!r}; choose one of {', '.join(_AUTH_MODES)}"
        )
    if auth == "bearer" and not token_env:
        raise McpConfigError(f"mcp server {name!r}: auth 'bearer' needs 'token_env'")

    oauth_timeout = block.get("oauth_timeout", DEFAULT_OAUTH_TIMEOUT)
    if (
        isinstance(oauth_timeout, bool)
        or not isinstance(oauth_timeout, (int, float))
        or oauth_timeout <= 0
    ):
        raise McpConfigError(f"mcp server {name!r}: 'oauth_timeout' must be a positive number")

    port = block.get("callback_port", 0)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise McpConfigError(f"mcp server {name!r}: 'callback_port' must be a port number")

    return {
        "url": url,
        "headers": dict(headers),
        "token_env": token_env,
        "auth": auth,
        "callback_port": port,
        "oauth_timeout": float(oauth_timeout),
    }


def _bool(name: str, block: dict[str, Any], key: str, *, default: bool) -> bool:
    value = block.get(key, default)
    if not isinstance(value, bool):
        raise McpConfigError(f"mcp server {name!r}: {key!r} must be true or false")
    return value


def _timeout(name: str, block: dict[str, Any]) -> float:
    value = block.get("timeout", DEFAULT_TIMEOUT)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise McpConfigError(f"mcp server {name!r}: 'timeout' must be a positive number")
    return float(value)


def _str_tuple(name: str, block: dict[str, Any], key: str) -> tuple[str, ...]:
    value = block.get(key, ())
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise McpConfigError(f"mcp server {name!r}: {key!r} must be a list of strings")
    if not all(isinstance(v, str) for v in value):
        raise McpConfigError(f"mcp server {name!r}: {key!r} must be a list of strings")
    return tuple(value)
