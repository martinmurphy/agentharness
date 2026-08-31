# Run a skill's Python scripts

## Context

The Agent Skills spec has always had three bundle directories — `references/`,
`assets/`, `scripts/` — and this harness has treated all three the same way: as text
to hand the model. `read_skill_file` will happily return a script's source, and
that is where it ends. The README says so as a scope decision: *"skill `scripts/` are
readable as text but never executed, and nothing here executes anything either."*

That makes a whole class of skill impossible. A skill that knows how to fill a PDF
form, parse a proprietary log, or check a file's structure cannot *do* it; it can only
describe the doing, and the model then reimplements the script badly through
`write_file`. The script sitting next to the instructions is the reliable version, and
it is inert.

This adds one tool that runs it.

Decisions taken up front:

| Decision | Choice |
|---|---|
| Trust boundary | Files under a loaded skill's `scripts/` directory. **Never model-authored code** |
| Mechanism | Subprocess on `sys.executable`. Never `shell=True` |
| Dependencies | Whatever the harness's own interpreter has. Per-skill environments deferred |
| Environment | Built from empty. Secrets only where the skill asks *and* the operator permits |
| Tool surface | One generic `run_skill_script`; the `SKILL.md` body documents each script |
| Gate | `skill_scripts.enabled`, default `false`; the tool is registered only when true |

The trust boundary is the decision the rest follows from. `/skills` is operator-curated
and mounted read-only, so a vetted script is trusted exactly the way vetted `SKILL.md`
prose is already trusted — the model chooses *which* to run and with what arguments, and
cannot author what runs. Extending this to the workspace would turn a prompt injection
into arbitrary code execution, which is a different feature with a different threat
model, and is not this one.

## Design

### 1. The seam — `skills/runner.py` and `tools/script_tools.py`

The repo already draws this line twice: `workspace.py` owns confined filesystem IO and
imports nothing from `tools/`; `skills/loader.py` owns `read_skill_file` while
`tools/skill_tools.py` owns the `Tool` wrappers. Execution is more safety-critical than
either, so it splits the same way.

`skills/runner.py` holds path resolution, environment construction, the subprocess call,
and the policy object. It imports `Skill` and `Workspace` and nothing else from the
harness, so every rule in it is unit-testable without a registry, a provider, or a REPL.
`tools/script_tools.py` is the thin wrapper: schema, argument coercion, rendering.

### 2. What may run — resolution

`read_skill_file` resolves against the skill directory. This resolves against the skill's
`scripts/` directory instead, which is tighter — but tighter takes two checks, not one,
because `.resolve()` follows symlinks on every path it touches:

```python
def resolve_script(skill: Skill, rel_path: str) -> Path:
    skill_root = skill.path.resolve()
    base = (skill.path / "scripts").resolve()
    # base must itself be inside skill_root before it is trusted as a boundary:
    # if scripts/ is itself a symlink, base silently becomes wherever that
    # symlink points, and everything under it would pass the check below.
    if not base.is_relative_to(skill_root):
        raise ValueError(f"skill {skill.name!r}: its scripts/ directory escapes the skill directory")
    target = (skill.path / rel_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"{rel_path!r} is not under the skill's scripts/ directory")
    if target.suffix != ".py":
        raise ValueError(f"only .py files can be run: {rel_path}")
    if not target.is_file():
        raise ValueError(f"no such script: {rel_path}")
    return target
```

The first `is_relative_to` check confirms `scripts/` itself hasn't been symlinked out of
the skill directory; only once that holds does the second reject `..`, absolute paths,
and symlink escapes within it — the same single-choke-point pattern as
`workspace.resolve_in`, applied to the base as well as the target. The rule the pair
leaves is one sentence a skill author can hold in their head: **`references/` is read,
`scripts/` is run.** A skill cannot be talked into executing its own documentation, and
`.py` is explicit rather than inferred so adding another interpreter later is a
deliberate act.

One asymmetry falls out of comparing resolved paths rather than reasoning about intent,
and predates this fix: symlinking the whole `scripts/` directory to a real directory kept
elsewhere in the bundle works, because `base` moves with it and the target still resolves
underneath; symlinking a single file back out to elsewhere in the bundle
(`scripts/tool.py -> ../lib/tool.py`) does not, because `base` stays put and the target
resolves outside it. Both are "a symlink that stays inside the skill," and only one is
honoured. Relocating the whole directory is the supported way to keep a script's real
home elsewhere; relocating one file is not, and the fix above doesn't change either.

### 3. Invocation

```python
proc = subprocess.Popen(
    [sys.executable, str(target), *args],
    cwd=ws.root,
    env=child_env,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", errors="replace",
    start_new_session=True,
)
try:
    out, err = proc.communicate(stdin_text, timeout=timeout)
except subprocess.TimeoutExpired:
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    out, err = proc.communicate()
    raise ScriptTimeout(timeout, out, err)
```

Four things here are load-bearing:

- **A list, never `shell=True`.** `args` must be a list of strings, validated before the
  call. There is no shell, so there is nothing to inject into — a model that puts
  `; rm -rf /` in an argument has passed a script one odd string.
- **`sys.executable`.** In the image that is `python3.14`; on the host it is the `.venv`
  interpreter. Both give the same story: the stdlib plus whatever agentharness itself
  depends on. Nothing is resolved from `PATH`, so a `python` earlier on it cannot
  substitute itself.
- **`start_new_session=True`.** `subprocess`'s timeout kills the direct child only. A
  script that forked would leave its children running and holding pipes open forever;
  the new session gives a process group to `killpg`.
- **`errors="replace"`.** A script that writes non-UTF-8 to stdout is a script with a
  bug, not a reason for the harness to raise.

`cwd` is the workspace root, so the natural thing a script does with a relative path is
the right thing. Note the honest limit: a subprocess is **not** confined by
`workspace.resolve_in`. It can reach any path the container user can. The container is
the confinement here, which is why the trust boundary is where it is.

### 4. The environment

Built from empty rather than inherited, so nothing arrives by accident:

```python
env = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "HOME": os.environ.get("HOME", str(ws.root)),
    "LANG": "C.UTF-8",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "AGENTHARNESS_WORKSPACE_DIR": str(ws.root),
    "AGENTHARNESS_SKILL_DIR": str(skill.path),
}
for name in sorted(skill.script_env & policy.env_allowlist):
    value = os.environ.get(name)
    if value is not None:
        env[name] = value
```

`PYTHONDONTWRITEBYTECODE` earns its place: `/skills` is mounted read-only, and a
`__pycache__` write that silently fails on every run is noise nobody will ever chase.
The two `AGENTHARNESS_*` variables mean a script can find both directories without the
model having to pass paths it might get wrong.

No provider key is in that dictionary. A script's stdout goes straight back into the
model's context, so an inherited `ANTHROPIC_API_KEY` is one `print(os.environ)` away
from the transcript.

**The allowlist is an intersection.** A skill declares what it needs, in `metadata` —
the extension point the Agent Skills spec actually sanctions — space-separated in the
style of `allowed-tools`:

```yaml
metadata:
  env: "GITHUB_TOKEN JIRA_TOKEN"
```

The loader validates each name against `^[A-Z_][A-Z0-9_]*$` and stores it on `Skill` as
`script_env: frozenset[str]`, leaving `metadata` itself untouched. A bad name fails
*that skill* with a `SkillLoadError`, which is how the loader already handles everything
else — one bad skill never hides the good ones.

But a declaration is a **request, not a grant**. Skills are bind-mounted and may come
from anywhere, including a registry; a downloaded skill that declares
`env: "ANTHROPIC_API_KEY"` must not get it merely by asking. What reaches the child is
the intersection with the operator's `skill_scripts.env_allowlist`, which is empty by
default. A variable that is declared, permitted, and simply unset is not forwarded and
is not an error — the script's own `os.environ` check reports it better than the harness
could.

### 5. The call, and what comes back

```json
{
  "skill":   {"type": "string"},
  "path":    {"type": "string"},
  "args":    {"type": "array", "items": {"type": "string"}, "default": []},
  "stdin":   {"type": "string"},
  "timeout": {"type": "integer"}
}
```

`skill` and `path` are required; the shape deliberately echoes `read_skill_file`, so a
model that can read a bundled file can run one. `args` must be strings — numbers are
rejected with a message saying so rather than silently stringified, because a model that
passed `3` where a script wanted `"3"` should learn that now and not through a
`TypeError` two layers down. Omitted `stdin` means the child's stdin is closed
immediately, not left open: a script that reads it gets EOF rather than a hang. An
omitted `timeout` uses `default_timeout`; a supplied one is clamped to `max_timeout`
without erroring, since a model guessing 600 wants a long run, not a failed call.

The result is a rendered block, not a JSON blob — the consumer is a language model:

```
exit status: 0
--- stdout ---
wrote b.pdf (12 fields)
--- stderr ---
(empty)
```

Each stream is capped independently at `max_output_bytes`, measured on the UTF-8
encoding and cut on a character boundary, then marked with
`... truncated: N more bytes` — echoing `list_dir`'s existing phrasing. An empty stream
renders as `(empty)` rather than nothing, so the model can tell a silent script from a
truncated frame.

**A non-zero exit is not a harness error.** It returns normally with the status visible,
because a script that exits 2 and explains why on stderr has told the model something it
can act on; `is_error=True` would encourage the model to treat the whole tool as broken.
`is_error` is reserved for the harness failing to run the thing at all: unknown skill,
path rejected, no such script, scripts disabled, or a timeout — and the timeout error
carries whatever partial output was captured before the kill, since that is usually
where the hang is explained.

### 6. The gate — `skill_scripts` config

```yaml
skill_scripts:
  enabled: false          # default
  default_timeout: 30     # seconds, when the model names none
  max_timeout: 120        # a model-supplied timeout is clamped to this
  max_output_bytes: 65536
  env_allowlist: []       # the operator half of the intersection above
```

`skill_scripts` joins `providers` and `mcp_servers` in the block loop in
`load_config`, held raw on `Config` for the same reason those are: `config.py` knows
nothing about what the block means. `skills/runner.py` parses and validates it into a
frozen `ScriptPolicy`, exactly as `mcp/config.py` does for `mcp_servers`.
`AGENTHARNESS_SKILL_SCRIPTS_ENABLED` overrides `enabled` through the same `_as_bool`
coercion, which is what makes `-e` and a test fixture pleasant.

Disabled means the tool is never **registered** — the model is not offered a capability
it cannot have, the same way `workspace_writable: false` never registers `write_file`
and `make_dir`. That also makes the wiring trivial: a `make_script_tools(...)` that
returns `[]` when disabled.

Default `false` is deliberate. The README currently promises that this program executes
nothing; turning that off should be an act an operator performs, not a property they
discover. It is one line.

### 7. Concurrency

Nothing to build. Registry handlers are plain synchronous callables that `_dispatch_all`
already runs in a per-turn thread pool (`agent.py`), and `spawn_subagent` already blocks
a worker for the length of a whole nested turn. A blocking `communicate(timeout=…)` is
the same shape, and `max_concurrency` already bounds how many run at once.

### 8. Container

Nothing to add to the `Containerfile`. `sys.executable` is the image's `python3.14`; the
read-only `/skills` mount is a feature, since a script cannot be rewritten between the
model reading it and the harness running it.

A skill needing a third-party package is an operator concern today: add it to the
`Containerfile` (and to the host `.venv` for host runs). That is the documented answer
until per-skill environments exist — and the tool signature does not change if they do,
because only the choice of interpreter would.

## Files

- `src/agentharness/skills/runner.py` — **new**. `ScriptPolicy`, `resolve_script`,
  the environment builder, `run_script`, and its error types.
- `src/agentharness/tools/script_tools.py` — **new**. `make_script_tools(skillset,
  workspace, policy)`; returns `[]` when disabled.
- `src/agentharness/skills/model.py` — `script_env: frozenset[str]` on `Skill`.
- `src/agentharness/skills/loader.py` — parse and validate `metadata.env`.
- `src/agentharness/config.py` — `skill_scripts` raw block; add it to the block loop.
- `src/agentharness/tools/registry.py` — thread the policy through
  `build_default_registry` and register the tool.
- `src/agentharness/repl.py` — build the `ScriptPolicy`, thread it into both registries
  (subagents get the tool too — it is not `spawn_subagent`), and a system-prompt line
  only when enabled.
- `config.example.yaml`, `README.md` (the *Scope (v1)* paragraph and *Extending* both
  currently state the opposite), `docs/plan.md`.
- `skills/` — an example skill with a runnable script, so the path is exercised by hand
  as well as by tests.

## Verification

```bash
.venv/bin/pytest -q          # no network, no API key
.venv/bin/ruff check .

printf '/tools\n/quit\n' | .venv/bin/agentharness
#   expect: no run_skill_script
printf '/tools\n/quit\n' | AGENTHARNESS_SKILL_SCRIPTS_ENABLED=true .venv/bin/agentharness
#   expect: run_skill_script present

podman build -t agentharness . && make run
```

New `tests/test_script_runner.py`, over a temporary skills tree:

- a script that prints, one that exits non-zero, one that reads stdin
- `../` , an absolute path, a symlink out of `scripts/`, a skill whose `scripts/`
  directory is itself a symlink out of the skill, a file in `references/`, a
  non-`.py` file — each rejected, each with its own assertion
- **the timeout kills the group**: a script that forks a long-lived child, then assert
  by PID that the grandchild is gone, not merely that we stopped waiting
- environment scrubbing asserted positively — run a script that dumps `os.environ` with
  a provider key set in the parent, and assert the key is absent
- the intersection: declared-and-permitted forwards, declared-only does not,
  permitted-only does not, declared-permitted-but-unset is silently absent
- an invalid `metadata.env` name yields a `SkillLoadError` and leaves other skills loaded
- output truncation at the cap, and `cwd` being the workspace root
- disabled policy registers no tool

---

## Status: implemented

The suite passes, ruff is clean, and the bundled `word-frequency` skill runs
end to end in the container with `skill_scripts.enabled: true`.

**Correction.** The final whole-branch review found that section 2's
`resolve_script` as originally specified above was exploitable: resolving
both the `scripts/` base and the target through symlinks means a skill whose
own `scripts/` directory is a symlink out of the skill (e.g. into the
workspace) made the target check pass for a file the model wrote itself.
Section 2 above has been corrected to the shipped fix — a containment check
on `base` itself, before it is trusted as a boundary — and the test list
below now names that case explicitly.
