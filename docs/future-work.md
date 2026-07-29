# Future work

Deferred improvements, kept here so they aren't lost. Each entry notes the
problem, why it's deferred, and a sketch of the fix.

## Surface the adaptive-thinking fallback (currently silent)

**Problem.** When an Anthropic model rejects `thinking={"type":"adaptive"}` +
`output_config.effort` (older models — Haiku 4.5, Sonnet 4.5), the adapter
catches the 400, drops those parameters, and retries — either with no thinking
or, if `thinking_budget` is set, with fixed-budget extended thinking. This all
happens **silently**: no log line, no on-screen notice. The user gets no signal
that the model doesn't support adaptive thinking or that a different thinking
mode was substituted; the only observable effect is a slightly slower first turn
(one extra round trip) on such a model.

See `AnthropicProvider.chat` / `_create` in `src/agentharness/providers/anthropic.py`.

**Why deferred.** The provider layer is intentionally decoupled from rendering
(it returns a `ProviderResponse`; the REPL owns all terminal output via the
event stream), so surfacing this cleanly is a small design choice rather than a
one-liner, and it isn't required for correctness.

**Sketch of the fix.** Either (or both):
- A `logging.info` in the adapter on fallback (e.g. "model X rejected adaptive
  thinking; retrying with budget_tokens=N" / "…with no thinking"). No coupling;
  off by default in the REPL, visible via log level / `ANTHROPIC_LOG`.
- A one-time dim REPL notice the first time a state falls back — thread the fact
  from the provider to the REPL (a small "last fallback" flag the REPL checks
  after a turn, or a dedicated event the loop renders).

## SSRF guardrails for the web_fetch tool

**Problem.** The `web_fetch` tool (`src/agentharness/tools/web_tools.py`) lets
the model make arbitrary outbound HTTP(S) requests. It restricts the scheme to
http/https, caps the response body, and sets a timeout, but it does **not**
block requests to private/loopback addresses or cloud metadata endpoints
(169.254.169.254, `localhost`, RFC 1918 ranges, `.internal`, …). A model that is
prompt-injected by fetched content could be steered into probing internal
services or exfiltrating instance credentials.

Two things have raised this since it was first written, and it is now the item
worth doing first:

- **The model can persist what it fetches.** `write_file` lands in a
  host-mounted directory, so data pulled from a metadata endpoint or a LAN
  service needs no outbound exfiltration channel — it can simply be written
  somewhere that gets synced or committed.
- **A container-to-host route is now routine.** Reaching a local model server
  means `host.containers.internal` resolves and works, so running in a container
  is not a boundary.

Both tools are in the base registry, so subagents have them too, and a
subagent's turn is far less visible than the main conversation.

**Why deferred.** Blocking loopback/private ranges outright would break a
legitimate dev-harness use — hitting a local API or `localhost` test server. The
right design is a *configurable* policy (default deny internal, opt-in
allowlist), which is more than a one-liner and wasn't in the tool's initial ask.

**Sketch of the fix.** Default-deny by *category of the resolved IP*, using
stdlib `ipaddress`: `is_loopback`, `is_private`, `is_link_local`, `is_reserved`,
`is_unspecified`, `is_multicast`. This needs no knowledge of the local topology.
Measured on a macOS podman setup it correctly denies `127.0.0.1`,
`192.168.127.254` (`host.containers.internal` under gvproxy) and the container's
own `10.88.0.0/16` subnet, while leaving public hosts alone.

Re-permit through an allowlist keyed on **host and port**:

```yaml
web_fetch:
  allow_private: false
  allowed_hosts: [host.containers.internal:1234]
```

Port matters: an IP-only entry for `127.0.0.1` re-opens `:5432`, `:6379` and
every other locally bound service. Compare on `(resolved IP, effective port)`,
handling implicit `:80`/`:443`.

**Resolve once, check that IP, connect to the pinned IP.** One mechanism, three
properties: the allowlist survives the address changing (`host.containers.internal`
is `192.168.127.254` under gvproxy but `10.0.2.2` under slirp4netns and different
again on native Linux — never hardcode it); DNS rebinding cannot swap the address
between check and connect; and alternate spellings (`127.1`, `2130706433`,
`[::ffff:127.0.0.1]`) cannot fool it, because the string is never inspected.

Two traps make the naive version wrong:

1. **Redirects.** `urlopen` follows up to 10 hops via `HTTPRedirectHandler`, so
   guarding only the input URL is bypassed by an allowed host returning a `302`
   to `169.254.169.254`. That handler's own scheme allowlist is
   `('http', 'https', 'ftp')` and the default opener includes `FTPHandler`, so a
   redirect can also escape the http/https restriction the tool enforces up
   front. Either re-check every hop or stop following redirects and surface the
   `Location` to the model.
2. **Pinning breaks TLS.** Connecting to an IP for an `https` target breaks SNI
   and certificate validation unless the original hostname is carried through as
   both the `Host` header and the TLS server name. It doesn't bite a plain-http
   local server; it does bite any allowlisted https host. This is why the fix is
   a custom opener rather than a check bolted in front of `urlopen`.

Surface rejections as ordinary tool errors. Tests need no network: stub
resolution and the opener.

## Pluggable / keyed web_search backends

**Problem.** The `web_search` tool (`src/agentharness/tools/search_tools.py`) has
a single, keyless DuckDuckGo backend that scrapes DDG's HTML endpoint. That keeps
search usable with zero setup, but it is unofficial and best-effort: it can break
when DDG's markup changes and may be rate-limited or blocked on datacenter IPs.

**Why deferred.** Keyless-first was the deliberate choice (zero setup). A robust
keyed backend (Brave, Tavily) needs an account and key and was out of scope for
the first pass.

**Sketch of the fix.** Add a keyed backend function alongside `_duckduckgo_search`
— e.g. `_brave_search(query, count)` (REST via `urllib`, no new dependency) —
returning the same `SearchResult` list (the shared contract). Add a
`search_backend` config field (default `"duckduckgo"`) to select it, and thread
it into `web_search_tool`. Keep DuckDuckGo as the keyless default so search still
works out of the box.

The config convention this needs now exists: provider aliases introduced
`api_key_env`, which names the environment variable to read rather than holding
the secret. A keyed search backend should reuse that pattern rather than invent
a second one.

## Unverified: does the tightened subagent prompt curb exploration?

**Problem.** A subagent asked to "respond only with your model name" once made
five tool calls — `list_models`, `list_dir`, `read_file`, `list_providers`,
`list_dir('..')` — exhausting a provider's free-tier minute quota mid-run.
`SUBAGENT_SYSTEM` (`repl.py`) was rewritten to make tool use conditional on the
task rather than encouraged, but that is a prompt change and has not been
observed working.

**Why open.** Only a real subagent run against a real provider tells you.
`list_models()` on that particular task is legitimate and should still happen;
what should stop is the unrelated file and provider poking.

**If it doesn't hold.** The structural fix is to stop `effective_system()` from
advertising the workspace on the subagent path. It is evaluated *after*
`states.new()` makes the subagent active, so a subagent inherits the workspace
line written for the main conversation.

## An ignore list for list_dir

**Problem.** `list_dir(recursive=True)` reports everything. On a Python project
that means `__pycache__/`, `*.egg-info/`, `.venv/`, and `.git/` — build output
the model neither asked for nor can use. Measured on a real turn: a recursive
listing of a checkout cost roughly 2,700 input tokens on the *following* call
(tool results are resent as history), a large share of it artifacts. It also
crowds the useful entries out of the model's attention.

**Why deferred.** `list_dir` was written to be honest — it shows what is there.
Hiding entries by default is a real behaviour change, and a naive version
introduces a worse failure than the one it fixes (see below), so it needs the
reporting design, not just a filter.

**Sketch of the fix.** Default patterns in config, matched per path *segment*
with `fnmatch` so `*.egg-info` works:

```yaml
workspace_ignore: ["__pycache__", "*.egg-info", ".venv", ".git", "node_modules"]
```

Prune during the walk rather than filtering the output — `rglob` will happily
descend into a 3,000-file `.venv` and pay the I/O before anything is discarded,
so this means a manual recursive walk in place of `Path.rglob`.

Note this is a *listing* concern only. `read_file` on an explicitly named
ignored path must keep working: the model should not be able to stumble into
build output, but it should still be able to read a file it was told about.
Visibility and access are separate.

### From the tool's point of view

The model must not need to know the ignore list in order to use the tool, so
the argument surface stays one boolean:

```
list_dir(path=".", recursive=true)                      -> filtered (default)
list_dir(path=".", recursive=true, include_ignored=true) -> everything
```

Rejected alternatives: an `ignore: [...]` argument (the model cannot know what
the defaults are, so it would be overriding blind) and an `include: [...]`
un-ignore list (same problem, more surface). A single opt-out is legible from
the model's side and cannot be got subtly wrong.

**The filtering must announce itself.** This is the part that matters, and the
reason a naive filter is worse than none: an agent that is silently shown less
than exists will confidently report the absence. This harness has already
produced "your workspace is essentially empty" from a correct-but-narrow
listing. So the result should end with something like:

```
… 412 entries hidden by ignore rules (__pycache__, *.egg-info);
    pass include_ignored: true to see them
```

That line is cheap, it keeps the tool honest, and it hands the model the exact
escape hatch — so a model that genuinely needs to see build output can, without
the operator editing config.

The tool description should say the listing is filtered by default and name the
flag; a model that reads "lists the contents of a directory" and gets a filtered
result has been misled by the schema.

## Migrate to the mcp 2.0 SDK

**Problem.** `pyproject.toml` pins `mcp>=1.9,<2`. The 2.0 release (2026-07-28)
renamed `McpError` to `MCPError` with no compatibility alias and changed enough
besides that the client here does not work on it: a fresh install against 2.0.0
fails 27 tests and errors 11 more, every stdio and HTTP connect ending in
`MCPError: Connection closed`.

**Why deferred.** The bound restores a working install immediately, which is
what mattered — the dependency was previously unbounded, so any fresh
`pip install -e .` on or after that date silently produced a broken harness.
Doing the migration properly means reading 2.0's changelog for the moved
symbols rather than chasing test failures until they pass.

**Sketch of the fix.** Install 2.0 in a scratch venv, run the suite, and work
the failures back to their imports — start with `mcp.shared.exceptions`,
`mcp.client.stdio`, `mcp.client.streamable_http`, and `mcp.client.auth`, which
is everything `agentharness/mcp/` imports. Then raise the bound in one commit
with the suite green on 2.x. Note this cuts both ways: a *server* launched via
`uvx` resolves its own `mcp` independently, so the harness and a server it
spawns can legitimately be on different major versions.

## MCP resources and prompts

**Problem.** The MCP client implements *tools* only. The protocol has two other
primitives, and neither has a slot in the harness today:

- **Resources** — read-only data addressed by URI (`file:///specs/api.md`,
  `db://tables/users`). The spec calls these application-controlled: an IDE
  surfaces them in an @-mention picker and the *user* chooses what enters the
  context. A CLI has no such picker.
- **Prompts** — named, parameterised instruction templates the server offers,
  meant to be invoked as slash commands by the user.

**Why deferred.** Tools are what an agent loop actually exercises; the other two
need a design decision first, not just code. Resources have no natural home
without a picker, and prompts overlap Agent Skills — which are already a catalog
of named instruction blobs loaded on demand — so shipping both would give the
model two catalogs with no story about which to prefer.

**Sketch of the fix.** Resources: two generic tools, `mcp_list_resources` and
`mcp_read_resource(server, uri)`, so the *model* pulls them (the only workable
inversion of control in a CLI). `ClientSession` already has `list_resources` and
`read_resource`; the manager would gain a pass-through and the rendering already
exists (`render_result` handles `EmbeddedResource`). Prompts: surface as REPL
commands (`/mcp-prompt <server>:<name>`) that expand server-side and seed the
next turn — never as something the model calls — and say plainly in the docs how
they differ from skills.

## stdio MCP servers inside the container

**Problem.** The image is UBI10 + `python3.14` and nothing else. Most published
MCP servers are distributed as npm or PyPI packages run via `npx` or `uvx`,
neither of which exists in the image, so a `type: stdio` entry that works on the
host fails in the container with a bare "No such file or directory".

**Why deferred.** The fix is a policy choice about image size, not a bug. Adding
`nodejs` + `npm` roughly doubles the image; adding `uv` is small but only covers
Python servers. Meanwhile `type: http` works in the container today, which is
the transport a containerised harness should probably prefer anyway — the server
is then a separate, separately-updated process rather than a subprocess of the
agent.

**Sketch of the fix.** Either (a) document the host as the place for stdio
servers and leave the image alone; (b) add `uv` to the Containerfile (~30 MB)
for Python servers only; or (c) publish a second image tag with a Node runtime
for people who need the npm ecosystem. If (b) or (c), the connect error should
name the missing runtime rather than letting `FileNotFoundError` speak for
itself.

## Prompt caching

**Problem.** Every model call re-sends and re-pays for a large fixed prefix. Measured
on a real turn: 2,846 input tokens to answer "what is the capital of Ireland" — of
which roughly 1,260 is tool schemas (10 tools; `web_fetch` alone ~255), ~110 the
skills catalog, ~60 the system prompt, ~25 the workspace line, and the rest
provider-side tool-use scaffolding. That block is byte-identical on every call, and
a turn using tools makes several calls. On short conversations it is the dominant
cost.

`docs/plan.md` lists caching as out of scope for v1, which predates knowing the
size of that prefix.

**Why deferred.** It is not one change. Caching is provider-specific — Anthropic
takes an explicit `cache_control` marker, OpenAI caches automatically with no API,
Gemini has its own mechanism — so it lands in each adapter, not in the loop. And
the neutral `Usage` (`providers/base.py`) has only `input_tokens` / `output_tokens`,
so cache activity is currently invisible: `/usage` and the per-turn line would keep
reporting full-price totals and silently misreport the moment caching is on.

**Sketch of the fix.**

*Anthropic adapter.* Render order is `tools` → `system` → `messages`, and the whole
thing is a prefix match — one byte earlier invalidates everything after. That order
is what makes this cheap here: a single breakpoint on the last system block caches
the tool schemas *and* the system prompt together, which is the entire fixed cost
above. Max 4 breakpoints per request. For multi-turn, a second breakpoint on the
last content block of the most recent turn extends the cached prefix as the
conversation grows.

*Accounting.* Add `cache_creation_input_tokens` and `cache_read_input_tokens` to the
neutral `Usage` and map them in each adapter (`cache_read_input_tokens` on
Anthropic, `prompt_tokens_details.cached_tokens` on OpenAI,
`cached_content_token_count` on Gemini — the read side is genuinely neutral; cache
*writes* are an Anthropic concept and will be zero elsewhere). Then `input_tokens`
means "uncached remainder", not "prompt size" — total prompt is the sum of all
three, and both `/usage` and the per-turn line need updating or they will
understate.

*Economics.* Reads are ~0.1× base input price; writes are 1.25× at the default
5-minute TTL, 2× at `ttl: "1h"`. Break-even is two requests at 5 minutes, three at
an hour. A REPL turn makes several calls in quick succession, so the default TTL is
right and the payback is immediate.

**Four traps, three of which this harness walks straight into:**

1. **Minimum cacheable prefix is model-dependent and not monotonic** — 512 tokens on
   Opus 5, 1024 on Opus 4.8 and Sonnet 5, 2048 on Opus 4.7, but **4096 on Opus 4.6
   and Haiku 4.5**. The ~2,850-token prefix measured above caches on Opus 4.8 and
   silently does not on Opus 4.6 — no error, just `cache_creation_input_tokens: 0`.
2. **The 20-block lookback.** A breakpoint searches back at most 20 content blocks
   for a prior entry. `max_tool_iterations` defaults to 10, and each round-trip adds
   an assistant message plus a tool message with N results — a long tool loop can
   exceed 20 blocks in a single turn and silently miss. Fix: an intermediate
   breakpoint roughly every 15 blocks.
3. **`/reload` and the subagent registry both change the tool list**, which renders
   at position 0 and invalidates everything. `/reload` is expected. The subagent
   registry legitimately differs (no `spawn_subagent`), so subagents keep their own
   cache entry — correct, but it means a delegating turn pays two cold writes.
4. **The workspace line is conditional** on the root existing (`effective_system`).
   Creating or removing the directory mid-session changes the system prefix. Harmless
   but worth knowing when a cache-hit rate moves for no obvious reason.
5. **MCP tools enlarge the prefix and can change mid-session.** Their schemas come
   from remote servers, so a configuration with several servers can dwarf the
   built-in ~1,260 tokens of tool schemas — which makes caching *more* valuable, not
   less. But `/mcp reconnect` rebuilds the registries, so it invalidates the whole
   prefix exactly like `/reload`.

**Verifying.** `cache_read_input_tokens` staying at zero across repeated identical
prefixes means something is invalidating silently — that is the check to write a
test around, not the mere presence of a `cache_control` key.

## Per-endpoint TLS trust (`verify` on providers and MCP servers) — low priority

**Problem.** TLS trust is process-global. An endpoint with a private CA or a
self-signed certificate is configured with `SSL_CERT_FILE`, which applies to
every consumer in the process — the three provider SDKs, the HTTP MCP transport,
and `web_fetch` — whether or not they need it. Two consequences:

- The variable *replaces* the trust store rather than adding to it (see
  `httpx/_config.py`, and stdlib `ssl.create_default_context()` behaves the same
  way), so a bundle that omits the public roots silently breaks every public
  endpoint at once. The README documents the combined-bundle fix.
- There is no way to relax verification for **one** endpoint. A self-signed
  certificate whose SAN does not match the configured `base_url` cannot be
  trusted by any bundle, and the only alternative is disabling verification
  everywhere, which nothing here offers on purpose.

**Why deferred — and why low priority.** The zero-code option works in every case
that matters: concatenate the private CA (or the self-signed leaf) with certifi's
roots and point `SSL_CERT_FILE` at the result. That covers private CAs,
self-signed endpoints, and mixed public/private setups, for providers and MCP
servers alike. What a `verify` key would add over it is scoping and ergonomics,
not capability — the one genuine capability gap is the mismatched-SAN case, whose
real fix is reissuing the certificate.

**Sketch of the fix.** One key, the same vocabulary in both blocks:

| Value | Meaning | Passed to httpx |
|---|---|---|
| absent, or `true` | Default trust; still honours `SSL_CERT_FILE` | `True` |
| `/path/ca.pem` | Trust exactly that PEM — a private CA, or a pinned self-signed leaf | `ssl.create_default_context(cafile=…)` |
| `/path/certs/` | Trust that directory | `ssl.create_default_context(capath=…)` |
| `false` | No verification at all | `False` |

```yaml
providers:
  scratch:
    type: openai
    base_url: https://gpu-box.lan:8000/v1
    verify: false                        # self-signed, wrong SAN

mcp_servers:
  internal:
    type: http
    url: https://mcp.corp.example/mcp
    verify: /etc/pki/corp-ca.pem
```

*Providers.* Resolve the config value to `True | False | ssl.SSLContext` in
`factory.py` and hand that neutral value to the adapter, which maps it to its own
SDK — the same neutral-value / per-adapter-mapping split the rest of `providers/`
uses, and necessary because the SDKs do not agree on the hook. All three adapters
already take a `client=` injection kwarg, so tests need no network.

| Adapter | Hook |
|---|---|
| `openai` | `openai.OpenAI(http_client=httpx.Client(verify=v))` |
| `anthropic`, `vertex` | `Anthropic(http_client=…)` / `AnthropicVertex(http_client=…)` |
| `gemini` | `genai.Client(http_options=HttpOptions(client_args={"verify": v}, async_client_args={"verify": v}))` — set both, since either client may be constructed |

`verify` must join `_HARNESS_KEYS` in `factory.py`, or it becomes a stray kwarg:
provider blocks are splatted into adapter constructors.

*MCP servers.* Simpler — `mcp/runtime.py` already builds the `httpx.AsyncClient`
itself and passes it to the transport, so this is one `verify=` argument. OAuth
comes along for free: `oauth.build_auth` returns an `httpx.Auth` attached to that
same client, so discovery, dynamic registration and token exchange inherit the
setting.

*Both.* Validate the value's structure at config load, but check the file at
connect — the rule `mcp/config.py` already documents for credentials, so a bad
path takes down one endpoint with a message naming it rather than the whole REPL.
`ssl.create_default_context(cafile=<missing>)` raises a bare `FileNotFoundError`
and needs wrapping. If both blocks get the key, the resolver belongs in a shared
`agentharness/tls.py`; with only one, leave it where it is used.

`verify: false` should be visibly unsafe — marked in `/providers` and `/mcp`, not
merely accepted.

## Small corrections

Not deferred by design — simply not done yet:

- `docs/plan-workspace-tools.md` pins stale test counts (`115 -> 165`, "165 tests
  pass") at lines 132 and 153. Either update or reword so no number is pinned.
- The README's "Providers" bullet still describes three backends and doesn't
  mention that `providers:` is now a registry of aliases.
