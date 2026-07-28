# Add MCP server support

## Context

Every tool the harness had was compiled in. `build_default_registry`
(`tools/registry.py`) names them one by one, and adding a capability meant writing a
Python module here. Meanwhile the ecosystem publishes capabilities as MCP servers —
GitHub, Sentry, Linear, Postgres, filesystem — that any MCP client can consume with
no code at all.

This adds that client, so tools can come from configuration. It reuses the shape the
repo already chose for providers: a config block that is a registry of *named
entries*, credentials resolved from named environment variables and never written in
config, and failures that degrade to a reported error rather than a dead REPL.

Decisions taken up front:

| Decision | Choice |
|---|---|
| Protocol | Official `mcp` SDK (PyPI `mcp`), added to `dependencies` |
| Transports | `stdio` and streamable HTTP. No legacy HTTP+SSE |
| MCP surface | **Tools only.** Resources and prompts → `docs/future-work.md` |
| Static auth | `token_env` (bearer), `${VAR}` in `headers`, `env_pass` for stdio |
| OAuth 2.1 | Full flow — **host only**, refused in the container |
| Tool naming | `mcp__<server>__<tool>`, sanitised, capped at 64 chars |

The SDK is async-only and the harness is entirely synchronous, so a bridge was
unavoidable. That bridge is the only genuinely delicate part of this change; the rest
is ordinary plumbing.

## Design

### The bridge — `mcp/runtime.py`

One background thread owns one event loop. Per server, one **supervisor task** enters
the transport and `ClientSession` context managers on an `AsyncExitStack`, publishes
the session, then parks on a stop event:

```python
async with AsyncExitStack() as stack:
    async with asyncio.timeout(server.handshake_timeout):
        read, write = await self._open_transport(stack, conn)
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        tools = await _list_all_tools(session)
    conn.ready.set_result(tools)
    await conn.stop.wait()          # …until asked to leave
```

It has to be this way round. The SDK's transports use anyio cancel scopes, which must
be exited by the task that entered them, so the exit stack is owned by that one task
and touched from nowhere else. Tool calls are submitted from outside via
`run_coroutine_threadsafe` — safe, because a call is only message passing over
streams that are already open.

Three things were learned by getting them wrong first:

- **The handshake must be bounded from inside the loop.** Bounding it only in the
  calling thread means a caller that gives up leaves the supervisor parked in
  `initialize()` forever, holding a subprocess nobody can reach. Verified by a test
  that checks the child PID is gone after a failed connect, not merely that our
  bookkeeping forgot it.
- **What surfaces from a failed connect is an `ExceptionGroup`.** anyio task groups
  wrap the real failure; reporting the group tells the user nothing, so `_leaves`
  flattens it and `_connect_error` reports the actual cause.
- **A stdio server's stderr must be captured, not inherited.** Servers log there
  routinely and it would scribble over the REPL — but it is also where a server that
  fails to start says why, so it goes to a temp file whose tail is quoted into the
  connection error.

### Identity and results — `mcp/manager.py`

The manager is to MCP what `SkillSet` is to skills: a live collection carrying both
what loaded and what failed. Servers connect **concurrently** (the REPL cannot prompt
until the tool list is known) but results are applied in **config order**, so which
tool wins a name collision is not a race.

Tool identity is decided here:

- `mcp__<server>__<tool>` — namespaced, so a server cannot shadow `write_file` by
  naming one of its own that way;
- sanitised to `[A-Za-z0-9_-]` and capped at 64 with a hash suffix, because all three
  provider APIs constrain function names and an unsanitised server name is a 400 at
  call time;
- collision-checked, so `ToolRegistry.register`'s duplicate guard is never what
  discovers the problem.

A server-reported failure is **raised**, because `ToolRegistry.dispatch` already turns
an exception into `ToolResult(is_error=True)` — the model sees the failure and can
recover. No change to the agent loop was needed anywhere in this work.

### Auth — `mcp/config.py` and `mcp/oauth.py`

Structure is validated at load; credentials are resolved at *connect*, so a missing
variable takes down one server with a message naming it rather than the whole REPL.

OAuth 2.1 is the SDK's `OAuthClientProvider` (an `httpx.Auth`, attached to the
transport's client) plus the two ends specific to this program: a token store under
`$XDG_CONFIG_HOME/agentharness/` written `0600` in a `0700` directory and keyed by
server URL, and a one-shot loopback listener that catches the redirect while the
authorisation URL is printed and opened.

`timeout` and `oauth_timeout` are deliberately separate. One is how long a machine may
take to answer a machine; the other is how long a person may take to read a consent
screen. Collapsing them means either aborting real users or waiting five minutes for
a dead host.

In the container OAuth is **refused**, not attempted: no browser, no writable config
directory, `/config` mounted read-only. The error names `token_env` as the supported
alternative. Half-working would be worse than an honest boundary.

## Verification

`make test` (272 tests) and `make lint`. No test touches the network — stdio servers
are local fixture scripts, HTTP and OAuth servers are `FastMCP` + Starlette on
ephemeral localhost ports.

The OAuth tests deserve a specific note: they run the **real** flow against a real
authorisation server (`tests/fixtures/oauth_mcp_server.py` implements enough of RFC
8414/9728/7591 to be a genuine peer, and verifies the S256 challenge, so a client
that got PKCE wrong would fail there). The only substitution is the human:
`webbrowser.open` is replaced by an HTTP GET of the same URL. Discovery, dynamic
client registration, PKCE, the code exchange through our own listener, token caching,
and reuse of a cached grant are all exercised.

Leak checks are assertions, not eyeballing: after `shutdown()` the loop thread is
joined and every stdio child is confirmed gone by PID.
