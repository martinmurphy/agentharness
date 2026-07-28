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

## Small corrections

Not deferred by design — simply not done yet:

- `docs/plan-workspace-tools.md` pins stale test counts (`115 -> 165`, "165 tests
  pass") at lines 132 and 153. Either update or reword so no number is pinned.
- The README's "Providers" bullet still describes three backends and doesn't
  mention that `providers:` is now a registry of aliases.
