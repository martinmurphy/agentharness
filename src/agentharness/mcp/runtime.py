"""The sync/async bridge to MCP servers.

Everything above this module is synchronous — tool handlers are
``Callable[[dict], str]`` and the agent loop is a plain generator — while the
MCP SDK is async only. So one background thread owns one event loop for the
whole process, and this module's entire public surface is blocking calls that
hand work to it.

The shape that matters:

* **One supervisor task per server.** It enters the transport and
  ``ClientSession`` context managers, publishes the session, then parks on a
  stop event until asked to leave. It must be this way round: the SDK's
  transports use anyio cancel scopes, which have to be exited by the same task
  that entered them, so the ``AsyncExitStack`` is owned by that one task and is
  never touched from anywhere else.
* **Tool calls are submitted from outside it.** That is safe — a call is just
  message passing over streams that are already open — via
  ``run_coroutine_threadsafe``, with the caller blocking on the result.

Failures are values, not exceptions escaping into the REPL: connecting returns
a result the caller inspects. The one thing this module guarantees on the way
out is that ``shutdown`` leaves no live thread and no orphaned subprocess.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import tempfile
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import IO, Any

import httpx
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client

from agentharness.mcp.config import McpServerConfig
from agentharness.mcp.oauth import build_auth

# How long to wait for a supervisor task to unwind before giving up on it and
# letting the loop close anyway. Generous enough for a subprocess to die.
_SHUTDOWN_GRACE = 5.0

# How much of a failing server's stderr to quote back in the error.
_STDERR_TAIL_LINES = 5
_STDERR_TAIL_CHARS = 4000

# An HTTP server holds its response stream open to push messages, so a read
# that takes a while is normal operation, not a stall. The server's configured
# timeout bounds connecting and the handshake instead.
_HTTP_READ_TIMEOUT = 300.0


_log = logging.getLogger(__name__)


def _log_teardown(name: str, exc: BaseException) -> None:
    """Note a supervisor that died on the way up. Debug level on purpose: the
    REPL has already shown the user the same failure, via connect()."""
    _log.debug("mcp server %r supervisor exited: %s: %s", name, type(exc).__name__, exc)


# tools/list is paginated. The cap is a guard against a server that keeps
# handing back cursors — we would rather advertise a truncated list than spin.
_MAX_TOOL_PAGES = 20


class McpConnectionError(RuntimeError):
    """A server could not be connected, or a call to it failed."""


def _leaves(exc: BaseException) -> list[BaseException]:
    """Flatten nested ExceptionGroups to the actual failures.

    The SDK's transports run inside anyio task groups, so what surfaces from a
    failed connect is usually an ExceptionGroup wrapping the one thing that
    went wrong. Reporting the group would tell the user nothing.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


# What the SDK reports when a POST to the message endpoint comes back 404
# (mcp/client/streamable_http.py — it is the only thing that raises this). The
# string says nothing about a status code or a URL, which makes the single most
# common misconfiguration look like a session problem, so translate it.
_SESSION_TERMINATED = "session terminated"


def _describe(cause: BaseException, server: McpServerConfig) -> str:
    """Say what failed in terms of the endpoint, not just the exception type.

    An SDK message like "Session terminated" or "Connection error" is accurate
    and useless: it names neither the status nor the host. For an HTTP server
    the URL is the thing being diagnosed, so it goes in the message.
    """
    status = getattr(getattr(cause, "response", None), "status_code", None)
    if status is not None:
        return f"HTTP {status} from {server.url or 'the server'}"

    text = f"{type(cause).__name__}: {cause}"
    if server.is_stdio:
        return text
    if _SESSION_TERMINATED in str(cause).lower():
        return (
            f"{text} — which is how the SDK reports HTTP 404 from {server.url}. "
            f"Check the path: this client speaks streamable HTTP, so a legacy "
            f"'/sse' endpoint answers the auth challenge and then 404s."
        )
    # A connection failure names an errno, never the host it was aimed at.
    return text if server.url in text else f"{text} ({server.url})"


def _connect_error(server: McpServerConfig, exc: BaseException) -> McpConnectionError:
    """Turn whatever a failed connect raised into one attributable message."""
    leaves = _leaves(exc)
    if any(isinstance(leaf, TimeoutError) for leaf in leaves):
        return McpConnectionError(
            f"mcp server {server.name!r} did not respond within {server.handshake_timeout:g}s"
        )
    cause = leaves[0] if leaves else exc
    return McpConnectionError(f"mcp server {server.name!r}: {_describe(cause, server)}")


async def _list_all_tools(session: ClientSession) -> list[types.Tool]:
    tools: list[types.Tool] = []
    cursor: str | None = None
    for _ in range(_MAX_TOOL_PAGES):
        listed = await session.list_tools(cursor)
        tools.extend(listed.tools)
        cursor = listed.nextCursor
        if not cursor:
            break
    return tools


@dataclass
class _Connection:
    """One server's live state, shared between the caller and its supervisor."""

    server: McpServerConfig
    stop: asyncio.Event
    # Handshake outcome: the tools the server offers, or the failure. Resolved
    # by the supervisor, awaited by the connecting thread.
    ready: concurrent.futures.Future[list[types.Tool]] = field(
        default_factory=concurrent.futures.Future
    )
    session: ClientSession | None = None
    supervisor: concurrent.futures.Future[None] | None = None
    tools: list[types.Tool] = field(default_factory=list)
    # A stdio server's stderr. Captured to a temp file rather than inherited:
    # servers log there routinely, and letting that land on the terminal would
    # scribble over the REPL. It is also where a server that fails to start
    # says why, so the tail goes into the connection error.
    errlog: IO[str] | None = None

    def stderr_tail(self, lines: int = _STDERR_TAIL_LINES) -> str:
        if self.errlog is None:
            return ""
        try:
            self.errlog.seek(0)
            captured = self.errlog.read()[-_STDERR_TAIL_CHARS:]
        except (OSError, ValueError):  # closed, or not seekable
            return ""
        kept = [ln for ln in captured.splitlines() if ln.strip()][-lines:]
        return "\n".join(kept)


class McpRuntime:
    """Owns the event-loop thread and the set of live sessions."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connections: dict[str, _Connection] = {}
        self._lock = threading.Lock()

    # ---- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the loop thread. Idempotent, so callers need not track it."""
        with self._lock:
            if self._thread is not None:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=self._run_loop, args=(loop,), name="mcp-runtime", daemon=True
            )
            self._loop, self._thread = loop, thread
            thread.start()

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def shutdown(self) -> None:
        """Disconnect every server, stop the loop, join the thread.

        Safe to call more than once, and safe to call when nothing was ever
        connected — the REPL's exit path should not have to know either way.
        """
        for name in list(self._connections):
            with contextlib.suppress(Exception):
                self.disconnect(name)
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        if loop is None or thread is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=_SHUTDOWN_GRACE)
        loop.close()

    # ---- connections --------------------------------------------------------

    def connect(self, server: McpServerConfig) -> list[types.Tool]:
        """Connect, handshake, and return the tools the server offers.

        Blocks for at most ``server.timeout``. Raises McpConnectionError on any
        failure — including a timeout, which also tears the attempt down rather
        than leaving a half-open session behind.
        """
        self.start()
        assert self._loop is not None

        conn = _Connection(server=server, stop=asyncio.Event())
        if server.is_stdio:
            # Not a context manager (SIM115) on purpose: its lifetime is the
            # connection's, and _teardown closing it is what deletes it.
            conn.errlog = tempfile.TemporaryFile(mode="w+", errors="replace")  # noqa: SIM115
        # Claim the name atomically: connect_all runs several of these at once,
        # so checking and inserting have to be one step.
        with self._lock:
            taken = server.name in self._connections
            if not taken:
                self._connections[server.name] = conn
        if taken:
            if conn.errlog is not None:
                conn.errlog.close()
            raise McpConnectionError(f"mcp server {server.name!r} is already connected")
        conn.supervisor = asyncio.run_coroutine_threadsafe(self._serve(conn), self._loop)

        # The supervisor bounds the handshake itself; this wait is only a
        # backstop for a loop thread that has stopped answering at all.
        try:
            conn.tools = conn.ready.result(timeout=server.handshake_timeout + _SHUTDOWN_GRACE)
        except concurrent.futures.TimeoutError as exc:
            raise self._failed(
                conn, f"mcp server {server.name!r} did not respond within {server.handshake_timeout:g}s"
            ) from exc
        except McpConnectionError as exc:
            raise self._failed(conn, str(exc)) from exc
        except Exception as exc:
            raise self._failed(
                conn, f"mcp server {server.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return conn.tools

    def _failed(self, conn: _Connection, message: str) -> McpConnectionError:
        """Build the error for a failed connect, then tear the attempt down.

        Order matters: a stdio server usually explains itself on stderr, and
        teardown closes the capture — so the tail is read first.
        """
        tail = conn.stderr_tail()
        self._teardown(conn)
        return McpConnectionError(f"{message}\n{tail}" if tail else message)

    def disconnect(self, name: str) -> None:
        """Close one server's session and reap its subprocess."""
        with self._lock:
            conn = self._connections.pop(name, None)
        if conn is not None:
            self._teardown(conn)

    def _teardown(self, conn: _Connection) -> None:
        """Ask the supervisor to leave its context managers, and wait for it.

        Signalling beats cancelling: the supervisor unwinds its own exit stack
        in its own task, which is the only place anyio's cancel scopes may be
        exited.
        """
        with self._lock:
            self._connections.pop(conn.server.name, None)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(conn.stop.set)
        if conn.supervisor is not None:
            try:
                conn.supervisor.result(timeout=_SHUTDOWN_GRACE)
            except concurrent.futures.TimeoutError:
                conn.supervisor.cancel()  # last resort; the loop is going away
            except Exception as exc:  # noqa: BLE001 - already reported via `ready`
                # A supervisor that died on the way up has nothing left to tell
                # us: connect() raised with this same failure attached.
                _log_teardown(conn.server.name, exc)
        conn.session = None
        if conn.errlog is not None:
            with contextlib.suppress(OSError, ValueError):
                conn.errlog.close()  # a TemporaryFile; closing deletes it
            conn.errlog = None

    def is_connected(self, name: str) -> bool:
        return name in self._connections

    def names(self) -> list[str]:
        return list(self._connections)

    def tools(self, name: str) -> list[types.Tool]:
        conn = self._connections.get(name)
        return list(conn.tools) if conn else []

    # ---- calls --------------------------------------------------------------

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        """Call a tool on a connected server and block for its result."""
        conn = self._connections.get(server_name)
        if conn is None or conn.session is None:
            raise McpConnectionError(f"mcp server {server_name!r} is not connected")
        loop = self._loop
        if loop is None:
            raise McpConnectionError("mcp runtime is not running")

        future = asyncio.run_coroutine_threadsafe(
            conn.session.call_tool(tool_name, arguments), loop
        )
        try:
            return future.result(timeout=conn.server.timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise McpConnectionError(
                f"{server_name}/{tool_name} timed out after {conn.server.timeout:g}s"
            ) from None

    # ---- the supervisor -----------------------------------------------------

    async def _serve(self, conn: _Connection) -> None:
        """Hold one server's session open until asked to stop.

        Everything that must be entered and exited by the same task lives here.

        The handshake is bounded from *inside* the loop, not just by the caller
        blocking outside it. A caller that merely gives up would leave this task
        parked in ``initialize()`` forever, holding an unreachable subprocess —
        the timeout has to be raised in here so that the exit stack unwinds and
        the SDK terminates the child.
        """
        server = conn.server
        try:
            async with AsyncExitStack() as stack:
                async with asyncio.timeout(server.handshake_timeout):
                    read, write = await self._open_transport(stack, conn)
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    tools = await _list_all_tools(session)
                conn.session = session
                if conn.ready.done():
                    return  # the caller gave up; unwind quietly
                conn.ready.set_result(tools)
                await conn.stop.wait()
        except Exception as exc:  # noqa: BLE001 - reported to the connecting thread
            self._fail(conn, _connect_error(server, exc))
        finally:
            conn.session = None

    @staticmethod
    def _fail(conn: _Connection, exc: BaseException) -> None:
        if not conn.ready.done():
            with contextlib.suppress(Exception):
                conn.ready.set_exception(exc)

    async def _open_transport(self, stack: AsyncExitStack, conn: _Connection):
        """Enter the configured transport and return its (read, write) streams."""
        server = conn.server
        if server.is_stdio:
            # The SDK's default is a small set of variables safe to inherit
            # (PATH, HOME, …). env_pass adds to it rather than replacing it: a
            # server launched as `uvx …` cannot start without PATH.
            params = StdioServerParameters(
                command=server.command,
                args=list(server.args),
                env=get_default_environment() | server.extra_env(),
                cwd=server.cwd or None,
            )
            read, write = await stack.enter_async_context(
                stdio_client(params, errlog=conn.errlog)
            )
            return read, write
        # The transport takes a whole httpx client, which is also where auth
        # attaches: static headers, or the OAuth flow as an httpx.Auth.
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                headers=server.resolve_headers(),
                auth=build_auth(server) if server.auth == "oauth" else None,
                follow_redirects=True,
                # Reads are long-lived (the response stream stays open for
                # server-sent messages), so only the connect side is bounded by
                # the server's timeout.
                timeout=httpx.Timeout(server.timeout, read=_HTTP_READ_TIMEOUT),
            )
        )
        read, write, _get_session_id = await stack.enter_async_context(
            streamable_http_client(server.url, http_client=client)
        )
        return read, write
