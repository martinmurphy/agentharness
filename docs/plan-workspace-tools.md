# Add filesystem tools over a bind-mounted workspace

## Context

The harness could read nothing on disk but skill bundles. `read_skill_file` reaches into
`/skills`, which is deliberately mounted read-only and scoped to a single skill's directory —
there was no place for the model to keep or consume ordinary files. Every artifact a turn
produced (a fetched page, a subagent's report, a draft) died with the process, because state
is in-memory only.

This adds a second bind-mounted directory, `/workspace`, and four tools over it. The mount has
the same shape as `/skills` — host-owned, edited freely outside the container, no image
rebuild — but read-write, so the model can produce durable output.

Decisions taken up front:

| Decision | Choice |
|---|---|
| Tool set | `list_dir`, `read_file`, `write_file`, `make_dir` — **no delete, no move** |
| Write gating | `workspace_writable` config key, default `true`; write tools registered only when true |
| Mount | `/workspace`, config `workspace_dir`, env `AGENTHARNESS_WORKSPACE_DIR` |

Omitting delete and move is the blast-radius choice: a mistaken or prompt-injected call can
clobber one file, not erase a tree. Adding them later is a new entry in `make_fs_tools`.

## Design

### 1. The core — `workspace.py` (new)

Mirrors the existing seam: `skills/loader.py` owns `read_skill_file` (the safety-critical IO),
`tools/skill_tools.py` owns the `Tool` wrappers. So `workspace.py` imports nothing from
`tools/` and is unit-testable on its own.

```python
MAX_READ_BYTES  = 256 * 1024      # matches skills.loader.MAX_FILE_BYTES
MAX_WRITE_BYTES = 1024 * 1024
MAX_ENTRIES     = 1000

@dataclass(frozen=True)
class Workspace:
    root: Path
    writable: bool = True
```

`Workspace` is the analogue of `SkillSet`: a config-derived value the harness holds and hands
to the tool factory. Confinement is a single choke point, the same one-check pattern as
`read_skill_file`:

```python
def resolve_in(ws: Workspace, rel_path: str) -> Path:
    base = ws.root.resolve()
    target = (ws.root / rel_path).resolve()   # non-strict: works for paths yet to exist
    if not target.is_relative_to(base):
        raise ValueError(f"path {rel_path!r} escapes the workspace")
    return target
```

`..`, absolute paths (pathlib's `/` discards the base for those — `is_relative_to` is what
catches it), and symlink escapes all fail here. It runs before any file is opened, so a
symlink inside the workspace pointing out cannot be written *through*, only named.

Everything else is an ordinary function over that: `read_file`, `list_dir`, `write_file`,
`make_dir`, each raising `ValueError` with a short lowercase message, per the house style.

### 2. The tools — `tools/fs_tools.py` (new)

`make_fs_tools(ws) -> list[Tool]`, built to the shape of `make_skill_tools`: closures over the
live `Workspace`, JSON Schema with `additionalProperties: False`, handlers that raise rather
than return error strings so `ToolRegistry.dispatch` produces the uniform
`ToolResult(is_error=True)`.

The write gate lives here and nowhere else: a non-writable workspace returns two tools instead
of four, so registration code stays unconditional and the model is never offered a tool that
would only refuse.

| Tool | Args | Notes |
|---|---|---|
| `list_dir` | `path="."`, `recursive=false` | `dir`/`file` lines with sizes; capped at `MAX_ENTRIES` with an explicit truncation line. `rglob` doesn't follow directory symlinks — left that way |
| `read_file` | `path` | UTF-8, size-capped, `UnicodeDecodeError` → a clear error rather than binary in the context |
| `write_file` | `path`, `content`, `mode` | `overwrite` (default) / `append` / `create` |
| `make_dir` | `path` | `parents=True`, idempotent |

Two behaviours worth stating because they are visible in the API:

- **`write_file` does not create parent directories.** A missing parent is an error naming
  `make_dir`. Each tool does one thing and a mistyped path fails loudly instead of silently
  growing a tree; the cost is one round trip on the first write into a new directory.
- **The root is never auto-created.** Read tools on a missing root say so; `make_dir` will
  create it, which is the one sanctioned way it comes into existence. This mirrors
  `load_skills`, where a missing directory is non-fatal.

### 3. Wiring

`build_default_registry(skillset)` becomes `build_default_registry(skillset, workspace)`. A
`Workspace` is the same category of thing as a `SkillSet` — config-derived, live, rebuilt on
`/reload` — so it belongs as a peer parameter rather than as a harness-only extra. Subagents
inherit the tools (they share the base registry), consistent with `web_fetch`.

`Harness.__init__` builds the `Workspace` from config; `effective_system()` appends one line
naming the root and the available verbs when the root exists, without which the model has the
tools but no reason to reach for them. `/workspace` prints the root, its mode, and a top-level
listing.

### 4. Container

`/workspace` joins `/skills` and `/config` in the Containerfile's mkdir/chgrp/chmod line, and
`AGENTHARNESS_WORKSPACE_DIR=/workspace` joins the `ENV` block. The Makefile's `run` target
mounts it `Z` without `ro`.

**The one genuinely new problem is UID mapping.** `/skills` is read-only, so ownership never
mattered. A writable mount does: the image runs `USER 1001`, which under rootless podman maps
to a subuid that cannot write a host-owned directory. `--userns=keep-id:uid=1001,gid=0` maps
the invoking user onto that UID and fixes it; it is overridable via `make run USERNS=` because
it is usually unnecessary on macOS, where the podman VM's virtiofs handles ownership.

## Files

- `src/agentharness/workspace.py` — **new**, the core.
- `src/agentharness/tools/fs_tools.py` — **new**, `make_fs_tools`.
- `src/agentharness/config.py` — `workspace_dir`, `workspace_writable`; both in
  `_SCALAR_FIELDS`, so YAML and env overrides work through the existing generic loop. The
  truthy-string lambda `show_thinking` used became a named `_as_bool` shared by both.
- `src/agentharness/tools/registry.py` — the `workspace` parameter and the `make_fs_tools` loop.
- `src/agentharness/repl.py` — build the `Workspace`, thread it into the registries, the
  system-prompt line, `/workspace`, and its `HELP` entry.
- `Containerfile`, `Makefile`, `workspace/.gitkeep`, `.gitignore`, `.containerignore`,
  `config.example.yaml`, `README.md`, `docs/plan.md`.

## Verification

```bash
.venv/bin/pytest -q          # no network, no API key
.venv/bin/ruff check .

printf '/workspace\n/tools\n/quit\n' | .venv/bin/agentharness
printf '/tools\n/quit\n' | AGENTHARNESS_WORKSPACE_WRITABLE=false .venv/bin/agentharness
#   expect: list_dir + read_file only; write_file and make_dir absent

podman build -t agentharness .
podman run --rm --userns=keep-id:uid=1001,gid=0 -v "$PWD/workspace:/workspace:Z" \
  --entrypoint python3.14 agentharness -c "..."   # write inside, read on the host
```

New tests: `tests/test_workspace.py` (confinement, caps, every mode and error path),
`tests/test_tools.py` (dispatch-level, plus the read-only gate), `tests/test_config.py` (the
two keys from YAML and env), `tests/test_agent.py` (`/workspace`, `/help`, registration, and
the system-prompt line).

---

## Status: implemented

The suite passes, ruff clean, the image builds, and a file written inside the container is
present and host-owned afterwards.
