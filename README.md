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
- **Tools** — a trivial `greet` tool proves the loop end to end, plus
  `read_skill` / `read_skill_file` that implement Agent Skills' progressive
  disclosure.
- **Skills** — discovered at runtime from a directory (bind-mounted from the
  host in the container), validated against the spec, and advertised to the
  model as a catalog it loads on demand.
- **Multiple states** — independent in-memory conversations, each with its own
  history, system prompt, provider, and model.
- **Providers** — Anthropic, Google Gemini (AI Studio), and any
  OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, …); the abstraction is
  built so the next backend is another adapter, not a refactor.

## Quick start (container)

```bash
make build

# REPL — mounts ./skills read-only and passes your API keys through.
ANTHROPIC_API_KEY=sk-... make run

# Or directly:
podman run --rm -it \
  -e ANTHROPIC_API_KEY \
  -v "$PWD/skills:/skills:ro,Z" \
  agentharness
```

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

.venv/bin/pytest -q                      # 60 tests, no network
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
/providers                               list providers and whether their keys are set
/models                                  list models the active provider can reach
/reload                                  re-scan the skills directory
/usage                                   token usage for the active state
/quit                                    exit
```

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
`./config.yaml`. Any key may be overridden by an `AGENTHARNESS_<KEY>`
environment variable. See [`config.example.yaml`](config.example.yaml). API keys
are **not** config keys — they come from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
or `GEMINI_API_KEY`.

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

To use a local OpenAI-compatible server:

```yaml
provider: openai
model: llama3.1
providers:
  openai:
    base_url: http://localhost:11434/v1   # Ollama, for example
```

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
- **Delegation** — the `spawn_subagent` tool lets the model create a fresh
  conversation state (defaulting to its own provider/model, or a different one),
  run a full tool/skill loop on it until it produces an answer, and get that
  answer back as the tool result. Subagents get the base toolset without
  `spawn_subagent`, so delegation is one level deep. Their processing is logged
  to the REPL, and the subagent's state persists (inspect it with `/switch`).
- **Add a skill** — create `skills/<name>/SKILL.md` with `name` (matching the
  directory) and `description` frontmatter. Optional `references/`, `assets/`,
  `scripts/` files are read as text via `read_skill_file`. Run `/reload`.
- **Add a provider** — implement the `Provider` protocol in
  `providers/base.py` (map to and from the neutral `Message` model), then wire
  it into `providers/factory.build_provider`. Nothing in the agent loop
  changes.

## Layout

```
src/agentharness/
  __main__.py          CLI: --version, --list-skills, else REPL
  config.py            Config dataclass; YAML + env resolution
  repl.py              REPL loop and slash commands
  agent.py             provider-neutral agent loop (yields events)
  state.py             ConversationState + StateManager (in-memory)
  skills/              Skill model, discovery, validation, catalog
  tools/               registry, greet, read_skill / read_skill_file
  providers/           neutral model + Anthropic / Gemini / OpenAI adapters
skills/                example skills (bind-mounted to /skills at runtime)
Containerfile          UBI10 + python3.14
```

## Scope (v1)

In-memory state only (nothing persists across restarts); skill `scripts/` are
readable as text but never executed; no streaming yet. Each is a clean addition
against the existing seams — see the design doc in the plan.
