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
- **Providers** — Anthropic and any OpenAI-compatible endpoint (OpenAI, Ollama,
  vLLM, …) today; the abstraction is built for Gemini and others next.

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
```

## Configuration

Config is resolved from `$AGENTHARNESS_CONFIG` → `/config/config.yaml` →
`./config.yaml`. Any key may be overridden by an `AGENTHARNESS_<KEY>`
environment variable. See [`config.example.yaml`](config.example.yaml). API keys
are **not** config keys — they come from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`.

To use a local OpenAI-compatible server:

```yaml
provider: openai
model: llama3.1
providers:
  openai:
    base_url: http://localhost:11434/v1   # Ollama, for example
```

## Extending

- **Add a tool** — write a handler and a `Tool` (see `tools/greet.py`), then
  register it in `tools/registry.build_default_registry`.
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
  providers/           neutral model, Anthropic + OpenAI-compatible adapters
skills/                example skills (bind-mounted to /skills at runtime)
Containerfile          UBI10 + python3.14
```

## Scope (v1)

In-memory state only (nothing persists across restarts); skill `scripts/` are
readable as text but never executed; no streaming, no Gemini yet. Each of those
is a clean addition against the existing seams — see the design doc in the plan.
