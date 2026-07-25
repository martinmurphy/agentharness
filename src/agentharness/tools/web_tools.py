"""A web request tool: fetch internet data over HTTP(S) via GET or POST.

Built on the standard library (``urllib``) so it needs no extra dependency and
works unchanged in the container.

Security note: this lets the model make arbitrary outbound HTTP requests, which
is an SSRF surface (it can reach localhost, cloud metadata endpoints, and other
internal services). We restrict the scheme to http/https, cap the response size,
and set a timeout, but we do NOT block private/loopback addresses — that would
break legitimate local-API testing in a dev harness. Host allowlisting is noted
as future work (docs/future-work.md).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from agentharness.tools.registry import Tool

_TIMEOUT_SECONDS = 30
_MAX_BYTES = 100_000  # cap the body we read into the model's context
_USER_AGENT = "agentharness/0.1"


def _fetch(args: dict) -> str:
    url = args.get("url")
    if not isinstance(url, str) or not url.strip():
        raise ValueError("'url' is required and must be a non-empty string")

    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme {scheme!r}; only http and https are allowed")

    method = str(args.get("method") or "GET").upper()
    if method not in ("GET", "POST"):
        raise ValueError("method must be 'GET' or 'POST'")

    headers = {"User-Agent": _USER_AGENT}
    accept = args.get("accept")
    if accept:
        headers["Accept"] = str(accept)  # the desired response content type

    data = None
    if method == "POST":
        body = args.get("body")
        if body is not None:
            # Accept a string body as-is, or serialise a JSON object for convenience.
            body_text = body if isinstance(body, str) else json.dumps(body)
            data = body_text.encode("utf-8")
            headers["Content-Type"] = str(args.get("content_type") or "application/json")

    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        # scheme is restricted to http/https above
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
            return _format(resp.status, resp.reason, resp.headers, resp.read(_MAX_BYTES + 1))
    except urllib.error.HTTPError as exc:
        # 4xx/5xx responses still carry a body worth returning to the model.
        body = exc.read(_MAX_BYTES + 1) if hasattr(exc, "read") else b""
        return _format(exc.code, exc.reason, exc.headers, body)
    except urllib.error.URLError as exc:
        raise ValueError(f"request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ValueError(f"request timed out after {_TIMEOUT_SECONDS}s") from exc


def _format(status, reason, resp_headers, raw: bytes) -> str:
    content_type = resp_headers.get("Content-Type", "") if resp_headers else ""
    truncated = len(raw) > _MAX_BYTES
    raw = raw[:_MAX_BYTES]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = f"<{len(raw)} bytes of non-text data (Content-Type: {content_type or 'unknown'})>"
    note = "\n… (response truncated)" if truncated else ""
    header = f"HTTP {status} {reason}"
    if content_type:
        header += f"\nContent-Type: {content_type}"
    return f"{header}\n\n{text}{note}"


def web_fetch_tool() -> Tool:
    return Tool(
        name="web_fetch",
        description=(
            "Fetch data from an http/https URL. Supports GET and POST. For POST, "
            "pass 'body' and optionally 'content_type' to declare the input type "
            "(e.g. application/json, application/x-www-form-urlencoded). Use "
            "'accept' to request a response type (sets the Accept header). Returns "
            "the status line and the response body as text (truncated if large)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The http/https URL to request."},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method (default GET).",
                },
                "body": {
                    "type": "string",
                    "description": "Request body for POST (e.g. a JSON or form-encoded string).",
                },
                "content_type": {
                    "type": "string",
                    "description": (
                        "Content-Type of the request body (the input type). "
                        "Defaults to application/json when a body is given."
                    ),
                },
                "accept": {
                    "type": "string",
                    "description": "Desired response content type; sets the Accept header.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        handler=_fetch,
    )
