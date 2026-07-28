"""OAuth 2.1 for MCP servers — host only.

The SDK owns the protocol: discovery, dynamic client registration, PKCE, the
code exchange and refresh all live in ``OAuthClientProvider``, which is an
``httpx.Auth`` and so attaches to the transport's HTTP client. What it does not
own is the two ends of the flow that are specific to a program's shape, and
that is what this module supplies:

* **where tokens live** — a file under the user's config directory, written
  ``0600`` in a ``0700`` directory, keyed by server URL so a renamed server
  keeps its grant;
* **how the user authorises** — the URL is printed and a browser opened, while
  a one-shot listener on ``127.0.0.1`` catches the redirect.

Both of those assume a desktop. The container image is headless, runs as an
unprivileged user with no config directory, and mounts ``/config`` read-only, so
``auth: oauth`` is refused there with a message pointing at ``token_env``
instead. Half-working would be worse than an honest boundary.
"""

from __future__ import annotations

import http.server
import json
import logging
import os
import socket
import urllib.parse
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import httpx
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

from agentharness.mcp.config import DEFAULT_OAUTH_TIMEOUT, McpServerConfig

_log = logging.getLogger(__name__)

CLIENT_NAME = "agentharness"
TOKEN_FILE = "mcp-tokens.json"
_STORE_VERSION = 1

# Files that exist only inside a container runtime.
_CONTAINER_MARKERS = ("/run/.containerenv", "/.dockerenv")


class OAuthUnavailable(RuntimeError):
    """OAuth cannot be run here (no browser, nowhere to keep tokens)."""


def in_container() -> bool:
    return any(Path(marker).exists() for marker in _CONTAINER_MARKERS)


def token_dir() -> Path:
    """The directory holding the token cache, honouring XDG."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "agentharness"


def require_available(server: McpServerConfig) -> None:
    """Refuse OAuth where it cannot honestly work, naming the alternative."""
    if in_container():
        raise OAuthUnavailable(
            f"mcp server {server.name!r} uses auth: oauth, which needs a browser and a "
            f"writable config directory — neither exists in the container. Use "
            f"token_env with a token issued outside it, or run the harness on the host."
        )


# ---- token storage -----------------------------------------------------------


@dataclass
class FileTokenStorage:
    """Per-server OAuth state, persisted for the next run of the harness.

    Implements the SDK's ``TokenStorage`` protocol. One file holds every
    server's entry, keyed by URL; a corrupt or unreadable file is treated as
    empty rather than fatal, since the worst case is authorising again.
    """

    server_url: str
    path: Path

    @classmethod
    def for_server(cls, server: McpServerConfig) -> FileTokenStorage:
        return cls(server_url=server.url, path=token_dir() / TOKEN_FILE)

    async def get_tokens(self) -> OAuthToken | None:
        return self._read_model("tokens", OAuthToken)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._write("tokens", tokens.model_dump(mode="json", exclude_none=True))

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._read_model("client_info", OAuthClientInformationFull)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._write("client_info", client_info.model_dump(mode="json", exclude_none=True))

    # ---- the file itself ----

    def _read_model(self, key: str, model: type) -> Any | None:
        raw = self._load().get(self.server_url, {}).get(key)
        if not raw:
            return None
        try:
            return model.model_validate(raw)
        except Exception:  # noqa: BLE001 - a stale shape just means re-authorising
            return None

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        servers = data.get("servers") if isinstance(data, dict) else None
        return servers if isinstance(servers, dict) else {}

    def _write(self, key: str, value: Any) -> None:
        servers = self._load()
        servers.setdefault(self.server_url, {})[key] = value
        self._save(servers)

    def _save(self, servers: dict[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps({"version": _STORE_VERSION, "servers": servers}, indent=2)
        # Write-then-rename so an interrupted save cannot truncate the store,
        # and create the temp file already private — never world-readable, not
        # even briefly.
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, self.path)
        os.chmod(self.path, 0o600)

    def clear(self) -> None:
        """Forget this server's grant, so the next connect authorises afresh."""
        servers = self._load()
        if servers.pop(self.server_url, None) is not None:
            self._save(servers)


# ---- the browser leg ---------------------------------------------------------


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures ``?code=…&state=…`` from the authorisation redirect."""

    result: tuple[str, str | None] | None = None
    error: str | None = None

    def do_GET(self) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        code = query.get("code", [""])[0]
        state = query.get("state", [None])[0]
        if code:
            type(self).result = (code, state)
            body = b"<html><body><h3>agentharness: authorised.</h3>You can close this tab.</body></html>"
            status = 200
        else:
            type(self).error = query.get("error", ["no code in the redirect"])[0]
            body = b"<html><body><h3>agentharness: authorisation failed.</h3></body></html>"
            status = 400
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr logging; it would land in the REPL."""


@dataclass
class LoopbackCallback:
    """A one-shot listener on 127.0.0.1 that the authorisation server redirects to."""

    port: int
    timeout: float = DEFAULT_OAUTH_TIMEOUT

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}/callback"

    def wait_for_code(self) -> tuple[str, str | None]:
        """Serve exactly one request and return its (code, state). Blocking."""
        _CallbackHandler.result = None
        _CallbackHandler.error = None
        try:
            server = http.server.HTTPServer(("127.0.0.1", self.port), _CallbackHandler)
        except OSError as exc:
            raise OAuthUnavailable(
                f"cannot listen on {self.redirect_uri} for the OAuth redirect: {exc}. "
                f"Set callback_port to a free port."
            ) from exc
        with server:
            server.timeout = self.timeout
            server.handle_request()
        if _CallbackHandler.result is None:
            raise OAuthUnavailable(
                f"no authorisation callback received: {_CallbackHandler.error or 'timed out'}"
            )
        return _CallbackHandler.result


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# ---- assembly ----------------------------------------------------------------


def build_auth(server: McpServerConfig) -> httpx.Auth:
    """The httpx.Auth that drives one server's OAuth flow.

    Raises OAuthUnavailable where the flow cannot be completed, which the
    runtime turns into that server's connection error.
    """
    require_available(server)
    callback = LoopbackCallback(
        port=server.callback_port or free_port(), timeout=server.oauth_timeout
    )
    storage = FileTokenStorage.for_server(server)

    async def redirect_handler(authorization_url: str) -> None:
        print(f"\n[mcp:{server.name}] authorise this harness at:\n  {authorization_url}\n")
        # A machine with no browser still gets the URL above, which is why the
        # print comes first and the failure to open one is not fatal.
        try:
            webbrowser.open(authorization_url)
        except Exception as exc:  # noqa: BLE001 - the printed URL is the fallback
            _log.debug("could not open a browser for %s: %s", server.name, exc)

    async def callback_handler() -> tuple[str, str | None]:
        # The listener blocks a thread, not the event loop, which still has a
        # live session on it.
        return await anyio.to_thread.run_sync(callback.wait_for_code)

    metadata = OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[callback.redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    return OAuthClientProvider(
        server_url=server.url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=server.oauth_timeout,
    )


def forget(server: McpServerConfig) -> None:
    """Drop a server's cached grant (used by ``/mcp login``)."""
    FileTokenStorage.for_server(server).clear()
