# agentharness

A small, provider-neutral AI harness with a REPL, tools, and [Agent
Skills](https://agentskills.io), designed to run inside a rootless Podman
container built on UBI10.

The point is legible seams: adding a provider, a tool, or a skill is a new file,
not a refactor. The agent loop is owned by the harness (not any SDK's built-in
loop), so a second backend is just another adapter over the neutral message
model.

## What it does

- **REPL** — type a prompt to the active conversation; slash commands manage
  everything else.
- **Tools** — model-callable functions in a registry: `greet` (a trivial
  example that proves the loop), `read_skill` / `read_skill_file` (Agent Skills
  progressive disclosure), `list_dir` / `read_file` / `write_file` / `make_dir`
  (the workspace directory), `web_search` / `web_fetch` (find and read web
  pages), `list_providers` / `list_models` (discover configured providers and
  their models), and `spawn_subagent` (delegate to a nested agent). See
  *Extending*.
- **MCP servers** — tools from configuration rather than code: local (stdio) or
  remote (streamable HTTP) [MCP](https://modelcontextprotocol.io) servers,
  several at once, authenticated with env-var tokens or OAuth 2.1. See *MCP
  servers*.
- **Skills** — discovered at runtime from a directory (bind-mounted from the
  host in the container), validated against the spec, and advertised to the
  model as a catalog it loads on demand.
- **Workspace** — a second bind-mounted directory the model can read and write
  through the filesystem tools, so a turn's output survives the process. Every
  path is confined to it.
- **Multiple states** — independent in-memory conversations, each with its own
  history, system prompt, provider, and model.
- **Parallel work** — tool calls the model puts in *one* message run
  concurrently, so four `spawn_subagent` calls in one message cost one round
  trip rather than four; and a prompt ending in ` &` runs in the background,
  leaving the prompt free for another state. See *Background jobs*.
- **Providers** — Anthropic, the same models through Google Vertex AI, Google
  Gemini (AI Studio), and any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM,
  …). `providers:` is a registry rather than a fixed list of names: a *second*
  endpoint speaking a protocol the harness already has is an alias in
  configuration — its own `base_url`, key variable, and the model it serves —
  not code. A genuinely new protocol is another adapter, not a refactor. See
  *Provider aliases*.

## Quick start (container)

```bash
make build

# REPL — mounts ./skills read-only, ./workspace read-write, passes API keys through.
ANTHROPIC_API_KEY=sk-... make run

# Or directly:
podman run --rm -it \
  --userns=keep-id:uid=1001,gid=0 \
  -e ANTHROPIC_API_KEY \
  -v "$PWD/skills:/skills:ro,Z" \
  -v "$PWD/workspace:/workspace:Z" \
  agentharness
```

The image runs as UID 1001, which under rootless podman maps to a subuid that
cannot write a host-owned directory — hence `--userns=keep-id:uid=1001,gid=0`,
which maps *you* onto that UID so files the model writes come back owned by you.
Drop it with `make run USERNS=` if your setup doesn't need it (it is usually
unnecessary on macOS). Read-only mounts like `/skills` never needed it.

`--list-skills` needs no API key and no model call:

```bash
make list-skills
# or: podman run --rm -v "$PWD/skills:/skills:ro,Z" agentharness --list-skills
```

The `:Z` mount flag is a no-op on macOS but relabels for SELinux hosts (RHEL,
Fedora). API keys are read from the environment only — never baked into the
image or a config file.

## Quick start (host, for development)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

.venv/bin/agentharness --list-skills     # no API key needed
ANTHROPIC_API_KEY=sk-... .venv/bin/agentharness

.venv/bin/pytest -q                      # no network, no API key
.venv/bin/ruff check .
```

## REPL commands

```
/help                                    this message
/states                                  list conversation states
/new <name> [--provider P] [--model M]   create and switch to a new state
/switch <name>                           switch to an existing state
/delete <name>                           delete a state
/reset                                   clear the active state's history
/skills                                  list loaded skills (and load failures)
/skill <name>                            print a skill's SKILL.md body
/tools                                   list registered tools
/workspace                               show the workspace directory and its contents
/providers                               list providers and whether their keys are set
/models                                  list models the active provider can reach
/reload                                  re-scan the skills directory
/mcp [tools S | reconnect S | login S]   MCP servers, their tools, status, and OAuth login
/usage [--all]                           token usage for the active state (or every state)
/jobs                                    list background jobs
/job <id>                                replay a background job's output
/quit                                    exit
```

After each turn a dim line reports what that turn cost and the running session
total — `[usage] turn 9,147 in / 61 out = 9,208 (2 calls) · session 9,208`.
The call count is the number of model round-trips, so a turn that used tools
shows more than one. Set `show_usage: false` to silence it; `/usage` still
reports the total. A turn that reported no usage prints nothing, since some
OpenAI-compatible servers omit the field.

Anything not starting with `/` is a prompt to the active state. Because skills
are mounted from the host, editing a `SKILL.md` and running `/reload` picks up
the change without rebuilding the image.

Each state can target a different backend:

```
/new research --provider openai --model gpt-4o
/new gem      --provider gemini --model gemini-3.5-flash
```

### Background jobs

A prompt ending in a standalone ` &` runs in the background and hands the prompt
straight back:

```
[default] › summarise the workspace &
[job 1] started on default
[default] › /jobs
    1  running  default     12s  summarise the workspace
[default] › /new scratch
[scratch] › what is 2+2                 # works while job 1 runs
[scratch] › /job 1                      # the job's output, replayed in full
```

Backgrounding is always explicit — a slow turn never takes the terminal away on
its own. A job's output is buffered rather than printed, because a job writing
while you are typing would scribble over the line readline owns; only a one-line
completion notice reaches the terminal, and it waits for the next prompt.

**One in-flight turn per state.** A second prompt on a busy state is refused
rather than queued: two turns appending to one history interleave into a
transcript neither of them wrote. Concurrency comes from using more states —
`/switch` or `/new`. `/delete` and `/reset` are refused on a busy state for the
same reason. `max_jobs` (default 4) caps how many run at once.

Within a single turn, tool calls the model puts in **one message** run
concurrently, up to `max_concurrency` (default 8); set it to `1` for the old
sequential dispatch. Results are *rendered* as they land, so you can see
progress, but recorded in the model's call order, so the transcript replays the
same way every time.

The "one message" part is the model's choice, not the harness's. A model that
issues four `spawn_subagent` calls across four messages gets four sequential
round-trips whatever `max_concurrency` says, because each message only ever
carried one call — no provider API has a knob that asks for batching. The
`spawn_subagent` description tells the model to put independent tasks in one
message; the `(N calls)` count on the `[usage]` line is how you check it did.
Four subagents batched is 2 calls; spawned one at a time it is 5.

## Configuration

Config is resolved from `$AGENTHARNESS_CONFIG` → `/config/config.yaml` →
`./config.yaml`. Any *scalar* key may be overridden by an `AGENTHARNESS_<KEY>`
environment variable; the `providers:` block is file-only, so per-provider
settings like `base_url` have no env equivalent. See
[`config.example.yaml`](config.example.yaml). API keys are **not** config keys —
they come from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`.

In the container the file comes from the `./config` bind mount (`make run`
mounts it read-only; an empty directory just means the defaults apply).

Gemini uses the **AI Studio (Developer API)** path — get a key at
<https://aistudio.google.com/apikey>, export it as `GEMINI_API_KEY`, and:

```yaml
provider: gemini
model: gemini-3.5-flash        # a current flash model with free-tier quota
```

Model IDs change over time and vary by account. If a model 404s ("no longer
available to new users") or 429s with `limit: 0` (pro tiers need billing), list
what your key can reach:

```bash
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY&pageSize=200" \
  | python3 -c "import sys,json; [print(m['name'].split('/')[-1]) for m in json.load(sys.stdin).get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]"
```

Pass it to the container the same way as the others:

```bash
podman run --rm -it -e GEMINI_API_KEY -v "$PWD/skills:/skills:ro,Z" agentharness
```

### Provider aliases

The three built-in names (`anthropic`, `openai`, `gemini`) are one adapter each,
with one `base_url` apiece. To reach a *second* endpoint of the same protocol,
add an alias — any other key under `providers:` with a `type` naming the adapter:

```yaml
provider: lmstudio
model: qwen3.5-9b-mlx

providers:
  lmstudio:                                        # a local server: no key at all
    type: openai
    base_url: http://host.containers.internal:1234/v1
  together:                                        # a hosted one: key from the env
    type: openai
    base_url: https://api.together.xyz/v1
    api_key_env: TOGETHER_API_KEY
```

Aliases are ordinary provider names everywhere: `/new local --provider lmstudio
--model qwen3.5-9b-mlx`, `/providers`, `/models`, and `spawn_subagent`'s
provider list all pick them up.

An endpoint that serves exactly one model can say so, and then naming the
provider is enough:

```yaml
providers:
  llama-70b:
    type: openai
    base_url: https://llama-70b.corp.example/v1
    api_key_env: LLAMA70B_API_KEY
    model: meta-llama/Llama-3.3-70B-Instruct
```

```
/new work --provider llama-70b          # binds to the model above
```

A model is resolved in three steps: an explicit `--model`, then the provider's
own `model:`, then the top-level `model:`. **The provider's key beats the
top-level one**, which is the fallback for providers that declare nothing rather
than a competing choice — the top-level `model:` has a default value, so
"was it set deliberately?" is not a question the config can answer. `/providers`
shows each provider's declared model so it is clear what a bare `--provider`
would select.

The same resolution applies to delegation: `spawn_subagent` carries the caller's
model over only when it stays on the caller's provider, since a model name means
nothing to a backend that has never heard of it.

**Keys still never live in the config.** A built-in's SDK reads its own env var;
an alias names the variable with `api_key_env` and the harness resolves it at
construction. If that variable is unset you get a clear error naming it, not a
401 from the SDK.

Two remaining details:

- **`base_url` ends at `/v1`** — the SDK appends `/models` and
  `/chat/completions` itself.
- **The hostname differs by where you run.** `host.containers.internal` reaches
  the host from inside a podman container; on the host it doesn't resolve — use
  `localhost` there.

A `base_url` with no key resolvable either way is treated as a keyless local
server and gets a placeholder, because the OpenAI SDK refuses to construct a
client with no key at all — so a local endpoint needs no key line. An env var
that *is* set is never overridden by that placeholder, so pointing a built-in at
a proxy still uses your real key. `/providers` shows keyless endpoints as
`no key needed` rather than `no key`.

### Endpoints with a private or self-signed certificate

A model served from a private endpoint often presents a certificate signed by an
internal CA, or a self-signed one. Nothing in the harness configures TLS trust —
it is set process-wide with `SSL_CERT_FILE`, which every consumer here honours:
the Anthropic, OpenAI and Gemini SDKs (through `httpx`), the HTTP MCP transport,
and `web_fetch` (through stdlib `urllib`).

**The trap is that `SSL_CERT_FILE` *replaces* the trust store rather than adding
to it.** Point it at a file containing only your internal CA and every public
endpoint stops verifying — a remote MCP server, `web_fetch`, and the hosted model
APIs all fail at once, which does not look like a certificate problem when it
happens.

So the bundle has to contain the public roots too. `certifi` ships them (it is
already installed as an `httpx` dependency), so build a combined file:

```bash
.venv/bin/python -c "import certifi; print(certifi.where())"   # where the public roots live

cat corp-ca.pem "$(.venv/bin/python -c 'import certifi; print(certifi.where())')" > bundle.pem
export SSL_CERT_FILE="$PWD/bundle.pem"
```

Use the interpreter the harness runs on — `certifi` arrives as an `httpx`
dependency, so it is in the virtualenv and usually not in the system `python3`.
Concatenated PEM files are a valid bundle, and order does not matter. Rebuild it
after upgrading `certifi`, or the roots in it go stale.

A **self-signed** endpoint — one with no CA behind it — works the same way: add
the server's own certificate to the bundle, since a self-signed certificate is
its own issuer. Verification still checks the hostname, so a certificate whose
subject alternative name does not cover the name in your `base_url` will be
rejected however you configure trust; reissue it with the right SAN. Skipping
verification for one endpoint is not currently possible — see
[`docs/future-work.md`](docs/future-work.md), which sketches per-provider and
per-server `verify` options.

In the container, mount the bundle and pass the variable:

```bash
podman run --rm -it \
  -e ANTHROPIC_API_KEY -e SSL_CERT_FILE=/certs/bundle.pem \
  -v "$PWD/bundle.pem:/certs/bundle.pem:ro,Z" \
  -v "$PWD/skills:/skills:ro,Z" agentharness
```

### Claude models through Google Vertex AI

`type: vertex` reaches Anthropic's models through Vertex AI. It is the same
Messages API — same adapter, so adaptive thinking, effort, tools and signed
thinking replay all behave identically — with a different endpoint and a
different way of authenticating:

```yaml
provider: vertex
model: claude-opus-4-8

providers:
  vertex:
    type: vertex
    project_id: my-gcp-project
    region: global          # or a multi-region ("us", "eu") or a specific region
```

**There is no API key.** Vertex authenticates with Google Application Default
Credentials, so `ANTHROPIC_API_KEY` is neither needed nor sent. On the host:

```bash
gcloud auth application-default login
gcloud services enable aiplatform.googleapis.com --project my-gcp-project
```

`project_id` and `region` may both be omitted, in which case the SDK reads
`ANTHROPIC_VERTEX_PROJECT_ID` and `CLOUD_ML_REGION` from the environment —
useful in the container, where you can pass them through instead of mounting a
config. Note that ADC credentials live in `~/.config/gcloud` on the host, so a
container run needs that directory mounted or a service-account key supplied by
whatever GCP-native means you use; `vertex` is a host-first path here for the
same reason MCP OAuth is.

Two things behave differently from the first-party API:

- **`/models` does not work.** Vertex serves the Messages API but not the Models
  API, so there is nothing to enumerate — `/models` says so rather than
  erroring. Name the model in config. Model IDs take no prefix: current models
  use the bare ID (`claude-opus-4-8`), and dated snapshots use an `@` separator
  (`claude-opus-4-5@20251101`, *not* `-20251101`).
- **`/providers` shows it as `no key needed`,** which is true of API keys but
  says nothing about whether your Google credentials are present or valid — that
  surfaces on the first turn. It is the same heuristic the built-ins use.

### MCP servers

Tools can also come from [MCP](https://modelcontextprotocol.io) servers, which
means from configuration rather than from code. Each entry under `mcp_servers:`
is one server; every tool it offers is registered as `mcp__<server>__<tool>` and
is thereafter an ordinary harness tool — it appears in `/tools`, subagents get
it, and the agent loop dispatches it like any other.

```yaml
mcp_servers:
  fs:                                    # a local subprocess
    type: stdio
    command: uvx
    args: [mcp-server-filesystem, /workspace]
    env_pass: [HOME]                     # forward named variables only

  github:                                # a remote server, static token
    type: http
    url: https://api.githubcopilot.com/mcp/
    token_env: GITHUB_MCP_TOKEN          # -> Authorization: Bearer <value>
    tools: [create_issue, search_code]   # optional allowlist

  internal:                              # a remote server, OAuth 2.1 (host only)
    type: http
    url: https://mcp.corp.example/mcp
    auth: oauth
    headers:
      X-Tenant: ${CORP_TENANT}           # ${...} is an env lookup
```

Common keys: `enabled` (default true — a declared server you can switch off),
`timeout` (seconds, default 15), and `tools` (an allowlist; an explicit empty
list advertises nothing). stdio servers also take `args`, `env_pass` and `cwd`;
HTTP servers take `headers`, `token_env`, `auth`, `callback_port` and
`oauth_timeout`.

**Keys never live in the config**, same rule as providers: `token_env` names the
variable holding a bearer token, `${VAR}` in a header value is an environment
lookup, and `env_pass` names the extra variables a stdio subprocess is given. A
missing variable fails that one server with a message naming it.

A stdio server inherits only a small safe set by default (`PATH`, `HOME`,
`SHELL`, `TERM`, `USER`, `LOGNAME` — a server launched as `uvx …` needs `PATH`
to exist at all); `env_pass` adds to that set, and nothing else reaches the
subprocess. So a server gets the credentials it was granted and no others.

Servers connect concurrently at startup, because the tool list has to be known
before the first turn. One that fails costs you that server and nothing else:
the reason is recorded and shown by `/mcp`, which also lists what each server
contributed and can `reconnect` one that was down.

```
[default] › /mcp
  fs      stdio  connected  4 tool(s)  (env: HOME)
  github  http   connected  12 tool(s)  (bearer: GITHUB_MCP_TOKEN)
  legacy  stdio  failed  see below
  ! legacy: mcp server 'legacy': McpError: Connection closed
      ModuleNotFoundError: No module named 'legacy_server'
```

A failing server's own stderr is captured, not printed — it would otherwise
scribble over the REPL — and the tail is quoted back in the error, which is
usually where a server that would not start says why.

**OAuth 2.1 is host-only.** `auth: oauth` runs the full flow (discovery, dynamic
client registration, PKCE, refresh); the authorisation URL is printed and opened
in a browser, and a one-shot listener on `127.0.0.1` catches the redirect. Tokens
are cached in `$XDG_CONFIG_HOME/agentharness/mcp-tokens.json` (mode `0600`) and
reused on later runs; `/mcp login <server>` forgets the grant and authorises
again. Inside the container there is no browser and no writable config
directory, so `auth: oauth` is refused there with a message pointing at
`token_env` — use a token issued outside the container instead.

Two things worth knowing before pointing this at a server you do not control:

- **A `stdio` entry is arbitrary local command execution** written in a config
  file. That is what the transport is. The container image ships no `npx` or
  `uvx`, so stdio servers there need a runtime added to the image — HTTP is the
  container-native path.
- **Tool names and descriptions are untrusted text** injected into the model's
  context every turn. Namespacing stops a server shadowing `write_file`; the
  per-server `tools:` allowlist limits what a chatty or hostile server can put in
  front of the model at all.

One operational gotcha: MCP servers validate the `Host` header against DNS
rebinding. Reaching one from inside the container by a different name — the
usual `http://host.containers.internal:PORT/mcp` — gives `421 Misdirected
Request` until that name is in the *server's* allowed hosts. That is the server
refusing, not the harness.

## Extending

- **Add a tool** — write a handler and a `Tool` (see `tools/greet.py`, or
  `tools/provider_tools.py` for one that exposes harness internals to the
  model), then register it in `tools/registry.build_default_registry`. A tool
  that needs live state (e.g. `list_models`, which queries the active provider —
  or any named provider, or `all` of them whose keys are set; or
  `spawn_subagent`, which runs a nested agent) is registered by the harness
  instead — see `Harness._build_registry` in `repl.py`.
- **Web access** — two tools: `web_search` (keyless DuckDuckGo search → result
  titles/URLs/snippets) to *discover* pages, and `web_fetch` (GET/POST an
  http/https URL) to *read* one. For `web_fetch` POST, pass a `body` and a
  `content_type` (input type); `accept` sets the desired response type. Both use
  stdlib `urllib`. Caveats: `web_search` is best-effort scraping (may break or be
  rate-limited); `web_fetch` is an SSRF surface (scheme is http/https-only and
  the body is size-capped, but host allowlisting is future work). See
  `docs/future-work.md`.
- **Filesystem** — four tools over the bind-mounted workspace: `list_dir`
  (optionally recursive), `read_file` (UTF-8 text, size-capped), `write_file`
  (`overwrite` / `append` / `create` modes), and `make_dir` (creates parents).
  Every path is resolved and required to stay under the workspace root, which
  rejects `..`, absolute paths, and symlink escapes in one check
  (`workspace.resolve_in`). `write_file` deliberately does *not* create parent
  directories — a missing parent is an error naming `make_dir`, so a mistyped
  path fails loudly. There is no delete or move: a bad call can clobber one file,
  not erase a tree. Set `workspace_writable: false` and the two write tools are
  never registered, so the model isn't offered them at all.
- **Delegation** — the `spawn_subagent` tool lets the model create a fresh
  conversation state (defaulting to its own provider/model, or a different one),
  run a full tool/skill loop on it until it produces an answer, and get that
  answer back as the tool result. Subagents get the base toolset without
  `spawn_subagent`, so delegation is one level deep. Their processing is logged
  to the REPL, and the subagent's state persists (inspect it with `/switch`).
- **Add an MCP server** — no code: add an entry under `mcp_servers:` (see *MCP
  servers*). Its tools are discovered at startup and registered alongside the
  built-in ones. `agentharness/mcp/` holds the client: `config.py` validates the
  block, `runtime.py` is the sync/async bridge (one loop thread, one supervisor
  task per server), `manager.py` owns the connections and decides tool identity,
  and `oauth.py` supplies token storage and the browser leg.
- **Add a skill** — create `skills/<name>/SKILL.md` with `name` (matching the
  directory) and `description` frontmatter. Optional `references/`, `assets/`,
  `scripts/` files are read as text via `read_skill_file`. Run `/reload`.
- **Add a provider** — for another endpoint speaking a protocol the harness
  already has, no code: add an alias under `providers:` with a `type` (see
  *Provider aliases*). For a genuinely new protocol, implement the `Provider`
  protocol in `providers/base.py` (map to and from the neutral `Message` model),
  then add it to `_ADAPTERS` and the dispatch in `providers/factory.py`. Nothing
  in the agent loop changes.

## Layout

```
src/agentharness/
  __main__.py          CLI: --version, --list-skills, else REPL
  config.py            Config dataclass; YAML + env resolution
  repl.py              REPL loop and slash commands
  agent.py             provider-neutral agent loop (yields events)
  context.py           the running turn: its state and its output sink
  jobs.py              background turns: Job + JobRunner
  state.py             ConversationState + StateManager (in-memory)
  workspace.py         workspace root + confined filesystem operations
  skills/              Skill model, discovery, validation, catalog
  mcp/                 MCP client: config, runtime bridge, manager, oauth
  tools/               registry + built-in tools (greet, skills, files, web,
                       providers, models, subagent, mcp)
  providers/           neutral model + Anthropic / Gemini / OpenAI adapters
skills/                example skills (bind-mounted to /skills at runtime)
workspace/             the model's read-write area (bind-mounted to /workspace)
Containerfile          UBI10 + python3.14
```

## Scope (v1)

Conversation state is in-memory only (histories do not survive a restart —
files the model wrote to the workspace do); skill `scripts/` are readable as
text but never executed, and nothing here executes anything either — `write_file`
writes bytes; no streaming yet. Each is a clean addition against the existing
seams. Design notes live in [`docs/plan.md`](docs/plan.md)
(the build plan) and [`docs/future-work.md`](docs/future-work.md) (deferred
items, incl. `web_fetch` SSRF allowlisting and keyed `web_search` backends).
