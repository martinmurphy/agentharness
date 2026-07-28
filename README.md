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
- **Skills** — discovered at runtime from a directory (bind-mounted from the
  host in the container), validated against the spec, and advertised to the
  model as a catalog it loads on demand.
- **Workspace** — a second bind-mounted directory the model can read and write
  through the filesystem tools, so a turn's output survives the process. Every
  path is confined to it.
- **Multiple states** — independent in-memory conversations, each with its own
  history, system prompt, provider, and model.
- **Providers** — Anthropic, Google Gemini (AI Studio), and any
  OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, …); the abstraction is
  built so the next backend is another adapter, not a refactor.

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

.venv/bin/pytest -q                      # 184 tests, no network
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
/usage [--all]                           token usage for the active state (or every state)
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
  state.py             ConversationState + StateManager (in-memory)
  workspace.py         workspace root + confined filesystem operations
  skills/              Skill model, discovery, validation, catalog
  tools/               registry + built-in tools (greet, skills, files, web,
                       providers, models, subagent)
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
