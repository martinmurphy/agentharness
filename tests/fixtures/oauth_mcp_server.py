"""An MCP server behind OAuth 2.1, plus the authorisation server, in one process.

Enough of RFC 8414 / 9728 / 7591 to drive the SDK's client through a real flow:
metadata discovery, dynamic client registration, an authorisation code with
PKCE, and a bearer-guarded MCP endpoint. It verifies the S256 challenge, so a
client that got PKCE wrong would fail here rather than pass silently.

Run as ``python oauth_mcp_server.py <port>``.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import sys

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route

PORT = int(sys.argv[1])
BASE = f"http://127.0.0.1:{PORT}"

server = FastMCP("secured", host="127.0.0.1", port=PORT, stateless_http=True)


@server.tool()
def secret() -> str:
    """Return a string only an authorised caller can see."""
    return "the vault is open"


# Issued state, in memory: one registered client, one code, one token.
CLIENTS: dict[str, dict] = {}
CODES: dict[str, dict] = {}
TOKENS: set[str] = set()


def _metadata(_request: Request) -> JSONResponse:
    return JSONResponse(
        {
            "issuer": BASE,
            "authorization_endpoint": f"{BASE}/authorize",
            "token_endpoint": f"{BASE}/token",
            "registration_endpoint": f"{BASE}/register",
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
        }
    )


def _protected_resource(_request: Request) -> JSONResponse:
    return JSONResponse(
        {"resource": f"{BASE}/mcp", "authorization_servers": [BASE]}
    )


async def _register(request: Request) -> JSONResponse:
    body = await request.json()
    client_id = f"client-{secrets.token_hex(4)}"
    CLIENTS[client_id] = body
    return JSONResponse(
        {**body, "client_id": client_id, "client_id_issued_at": 0}, status_code=201
    )


def _authorize(request: Request) -> Response:
    params = request.query_params
    redirect_uri = params.get("redirect_uri", "")
    if not redirect_uri:
        return JSONResponse({"error": "invalid_request"}, status_code=400)
    code = f"code-{secrets.token_hex(8)}"
    CODES[code] = {
        "challenge": params.get("code_challenge", ""),
        "redirect_uri": redirect_uri,
    }
    # Stand in for a human clicking "allow".
    state = params.get("state", "")
    return RedirectResponse(f"{redirect_uri}?code={code}&state={state}", status_code=302)


async def _token(request: Request) -> JSONResponse:
    form = await request.form()
    grant = form.get("grant_type")
    if grant == "refresh_token":
        token = f"token-{secrets.token_hex(8)}"
        TOKENS.add(token)
        return JSONResponse(
            {"access_token": token, "token_type": "Bearer", "expires_in": 3600}
        )

    issued = CODES.pop(str(form.get("code", "")), None)
    if issued is None:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)

    verifier = str(form.get("code_verifier", ""))
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    if issued["challenge"] and expected != issued["challenge"]:
        return JSONResponse({"error": "invalid_grant", "detail": "pkce"}, status_code=400)

    token = f"token-{secrets.token_hex(8)}"
    TOKENS.add(token)
    return JSONResponse(
        {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": f"refresh-{secrets.token_hex(8)}",
        }
    )


def _guard(app):
    """Refuse unauthenticated MCP requests, pointing at the resource metadata."""

    async def wrapped(scope, receive, send):
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        presented = headers.get("authorization", "").removeprefix("Bearer ")
        if presented not in TOKENS:
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        'Bearer resource_metadata='
                        f'"{BASE}/.well-known/oauth-protected-resource/mcp"'
                    )
                },
            )
            await response(scope, receive, send)
            return
        await app(scope, receive, send)

    return wrapped


mcp_app = server.streamable_http_app()

app = Starlette(
    # Mounting FastMCP inside another app does not carry its lifespan across,
    # and its session manager is started there — without this every MCP request
    # is a 500.
    lifespan=lambda _app: mcp_app.router.lifespan_context(mcp_app),
    routes=[
        Route("/.well-known/oauth-authorization-server", _metadata),
        Route("/.well-known/oauth-authorization-server/mcp", _metadata),
        Route("/.well-known/openid-configuration", _metadata),
        Route("/.well-known/oauth-protected-resource", _protected_resource),
        Route("/.well-known/oauth-protected-resource/mcp", _protected_resource),
        Route("/register", _register, methods=["POST"]),
        Route("/authorize", _authorize),
        Route("/token", _token, methods=["POST"]),
        Mount("/", app=_guard(mcp_app)),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="error")
