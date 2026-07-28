"""A streamable-HTTP MCP server on localhost, used as a real peer in tests.

Run as ``python http_mcp_server.py <port> [expected-token]``. When a token is
given, every request must carry ``Authorization: Bearer <token>`` or it is
refused with 401 — which is what lets the tests assert that configured
credentials actually reach the wire.
"""

from __future__ import annotations

import os
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

PORT = int(sys.argv[1])
EXPECTED_TOKEN = sys.argv[2] if len(sys.argv) > 2 else ""

# MCP servers validate the Host header against DNS rebinding, so reaching this
# one under any name other than 127.0.0.1 (from a container, say) needs that
# name allowed. Set MCP_ALLOWED_HOSTS to a comma-separated list.
_extra_hosts = [h for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h]

server = FastMCP(
    "remote",
    host="127.0.0.1",
    port=PORT,
    stateless_http=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[f"127.0.0.1:{PORT}", f"localhost:{PORT}", *_extra_hosts]
    ),
)


@server.tool()
def ping() -> str:
    """Return a fixed string, to prove the round trip works."""
    return "pong"


@server.tool()
def whoami(header: str = "x-tenant") -> str:
    """Echo one request header back, so header plumbing can be checked."""
    from mcp.server.fastmcp.server import Context  # noqa: F401  (documented import path)

    request = server.get_context().request_context.request
    return request.headers.get(header, "<absent>") if request else "<no request>"


async def _require_token(scope, receive, send, app):
    headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
    if headers.get("authorization") != f"Bearer {EXPECTED_TOKEN}":
        response = JSONResponse({"error": "unauthorized"}, status_code=401)
        await response(scope, receive, send)
        return
    await app(scope, receive, send)


if __name__ == "__main__":
    app = server.streamable_http_app()
    if EXPECTED_TOKEN:
        inner = app

        async def app(scope, receive, send):  # deliberately shadows the app above
            if scope["type"] != "http":
                await inner(scope, receive, send)
                return
            await _require_token(scope, receive, send, inner)

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")
