"""Tests for the skill script policy, resolution, environment, and subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time

import pytest

from agentharness.config import Config
from agentharness.skills.loader import load_skills
from agentharness.skills.runner import (
    DEFAULT_MAX_OUTPUT_BYTES,
    ScriptConfigError,
    ScriptPolicy,
    ScriptResult,
    ScriptTimeout,
    child_env,
    parse_policy,
    resolve_script,
    run_script,
)
from agentharness.workspace import Workspace


def _skill(tmp_path, *, name="demo", env="", scripts=None):
    """Build a skill directory and load it through the real loader.

    Going through load_skills rather than constructing a Skill directly is
    deliberate: script_env then comes from the same parsing path production
    uses, so a test cannot pass against a shape the loader never produces.
    """
    root = tmp_path / "skills"
    directory = root / name
    (directory / "scripts").mkdir(parents=True)
    meta = f'metadata:\n  env: "{env}"\n' if env else ""
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A demo skill used by tests.\n{meta}---\n\nBody.\n",
        encoding="utf-8",
    )
    for rel, source in (scripts or {}).items():
        path = directory / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    skill = load_skills(root).by_name(name)
    assert skill is not None
    return skill


def _ws(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return Workspace(root=root, writable=True)


def _enabled(**kwargs):
    return ScriptPolicy(enabled=True, **kwargs)


def test_policy_defaults_to_disabled(monkeypatch):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    policy = parse_policy(Config())
    assert policy.enabled is False
    assert policy.env_allowlist == frozenset()
    assert policy.max_output_bytes == DEFAULT_MAX_OUTPUT_BYTES


def test_policy_from_block(monkeypatch):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    policy = parse_policy(Config(skill_scripts={
        "enabled": True,
        "default_timeout": 10,
        "max_timeout": 60,
        "max_output_bytes": 2048,
        "env_allowlist": ["GITHUB_TOKEN"],
    }))
    assert policy.enabled is True
    assert policy.env_allowlist == frozenset({"GITHUB_TOKEN"})
    assert policy.default_timeout == 10


def test_env_override_wins_over_file(monkeypatch):
    monkeypatch.setenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", "true")
    assert parse_policy(Config(skill_scripts={"enabled": False})).enabled is True
    monkeypatch.setenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", "no")
    assert parse_policy(Config(skill_scripts={"enabled": True})).enabled is False


@pytest.mark.parametrize("block, fragment", [
    ({"enabbled": True}, "enabbled"),
    ({"max_timeout": 0}, "max_timeout"),
    ({"max_timeout": "30"}, "max_timeout"),
    ({"default_timeout": 90, "max_timeout": 60}, "default_timeout"),
    ({"env_allowlist": "GITHUB_TOKEN"}, "env_allowlist"),
    ({"env_allowlist": ["lower_case"]}, "lower_case"),
])
def test_bad_block_is_rejected_at_load(monkeypatch, block, fragment):
    monkeypatch.delenv("AGENTHARNESS_SKILL_SCRIPTS_ENABLED", raising=False)
    with pytest.raises(ScriptConfigError) as exc:
        parse_policy(Config(skill_scripts=block))
    assert fragment in str(exc.value)


def test_clamp_timeout():
    policy = ScriptPolicy(enabled=True, default_timeout=30, max_timeout=120)
    assert policy.clamp_timeout(None) == 30
    assert policy.clamp_timeout(5) == 5
    assert policy.clamp_timeout(600) == 120       # clamped, not rejected
    with pytest.raises(ValueError):
        policy.clamp_timeout(0)
    with pytest.raises(ValueError):
        policy.clamp_timeout("30")
    with pytest.raises(ValueError):
        policy.clamp_timeout(True)                # bool is an int; not a timeout


def test_resolves_a_script_under_scripts(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/ok.py": "print('hi')\n"})
    assert resolve_script(skill, "scripts/ok.py").name == "ok.py"


@pytest.mark.parametrize("rel_path, fragment", [
    ("scripts/../../escape.py", "scripts/"),
    ("/etc/passwd", "scripts/"),
    ("references/REGIONAL.md", "scripts/"),
    ("notes.py", "scripts/"),
    ("scripts/notes.txt", "only .py"),
    ("scripts/missing.py", "no such script"),
    (123, "non-empty string"),
    (None, "non-empty string"),
    ("", "non-empty string"),
])
def test_rejects(tmp_path, rel_path, fragment):
    skill = _skill(tmp_path, scripts={
        "scripts/ok.py": "print('hi')\n",
        "scripts/notes.txt": "not a script\n",
        "references/REGIONAL.md": "docs\n",
        "notes.py": "print('top level')\n",
    })
    (tmp_path / "skills" / "escape.py").write_text("print('nope')\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        resolve_script(skill, rel_path)
    assert fragment in str(exc.value)


def test_symlink_out_of_scripts_is_rejected(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/ok.py": "print('hi')\n"})
    outside = tmp_path / "outside.py"
    outside.write_text("print('nope')\n", encoding="utf-8")
    link = skill.path / "scripts" / "link.py"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_script(skill, "scripts/link.py")


def test_scripts_directory_itself_a_symlink_out_is_rejected(tmp_path):
    """`scripts` itself as a symlink out of the skill directory must be refused.

    ``.resolve()`` follows symlinks on both the base and the target, so if
    ``scripts`` is a symlink pointing elsewhere, the base resolves to that
    elsewhere and every file under it passes the plain ``is_relative_to``
    check — even though none of it is inside the skill directory. This is
    the asymmetry: a symlink AT ``scripts/link.py`` is refused (the test
    above), but a symlink AT ``scripts`` itself was, before the fix, honoured.
    """
    skill = _skill(tmp_path, name="linked")
    outside = tmp_path / "outside-scripts"
    outside.mkdir()
    (outside / "evil.py").write_text("print('nope')\n", encoding="utf-8")
    scripts_dir = skill.path / "scripts"
    scripts_dir.rmdir()
    scripts_dir.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes the skill"):
        resolve_script(skill, "scripts/evil.py")


def test_scripts_symlink_at_the_workspace_is_rejected(tmp_path):
    """The headline case from the review: a skill whose scripts/ is a symlink
    into the workspace directory must not let a model-authored file run.

    Reproduced directly (not through run_script) so this pins resolve_script's
    contract; the end-to-end route through the real ToolRegistry is covered
    separately in tests/test_tools.py.
    """
    skill = _skill(tmp_path, name="linked")
    ws = _ws(tmp_path)
    (ws.root / "evil.py").write_text(
        "print('MODEL AUTHORED CODE RAN')\n", encoding="utf-8"
    )
    scripts_dir = skill.path / "scripts"
    scripts_dir.rmdir()
    scripts_dir.symlink_to(ws.root)
    with pytest.raises(ValueError, match="escapes the skill"):
        resolve_script(skill, "scripts/evil.py")


def test_scripts_symlink_staying_inside_the_skill_still_works(tmp_path):
    """The fix must not be over-broad: a `scripts` symlink is fine as long as
    it stays inside the skill's own directory — e.g. a skill author who keeps
    the real directory elsewhere in the bundle and links it in.
    """
    skill = _skill(tmp_path, name="relocated")
    real_scripts = skill.path / "actual-scripts"
    real_scripts.mkdir()
    (real_scripts / "ok.py").write_text("print('hi')\n", encoding="utf-8")
    scripts_dir = skill.path / "scripts"
    scripts_dir.rmdir()
    scripts_dir.symlink_to(real_scripts)
    assert resolve_script(skill, "scripts/ok.py").name == "ok.py"


def test_ordinary_real_scripts_directory_still_works(tmp_path):
    """Regression guard: the common case, a real (non-symlink) scripts/
    directory, must be unaffected by the new containment check.
    """
    skill = _skill(tmp_path, scripts={"scripts/ok.py": "print('hi')\n"})
    assert not (skill.path / "scripts").is_symlink()
    assert resolve_script(skill, "scripts/ok.py").name == "ok.py"


def test_child_env_is_built_not_inherited(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_OTHER_VAR", "leak")
    skill = _skill(tmp_path)
    ws = _ws(tmp_path)
    env = child_env(skill, ws, _enabled())
    assert "ANTHROPIC_API_KEY" not in env
    assert "SOME_OTHER_VAR" not in env
    assert env["AGENTHARNESS_WORKSPACE_DIR"] == str(ws.root)
    assert env["AGENTHARNESS_SKILL_DIR"] == str(skill.path)
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_env_forwarded_only_when_declared_and_permitted(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "gh-1")
    monkeypatch.setenv("JIRA_TOKEN", "jira-1")
    monkeypatch.setenv("SLACK_TOKEN", "slack-1")
    skill = _skill(tmp_path, env="GITHUB_TOKEN JIRA_TOKEN")
    policy = _enabled(env_allowlist=frozenset({"GITHUB_TOKEN", "SLACK_TOKEN"}))
    env = child_env(skill, _ws(tmp_path), policy)
    assert env["GITHUB_TOKEN"] == "gh-1"   # declared and permitted
    assert "JIRA_TOKEN" not in env         # declared, not permitted
    assert "SLACK_TOKEN" not in env        # permitted, not declared


def test_declared_and_permitted_but_unset_is_silently_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    skill = _skill(tmp_path, env="GITHUB_TOKEN")
    env = child_env(skill, _ws(tmp_path), _enabled(env_allowlist=frozenset({"GITHUB_TOKEN"})))
    assert "GITHUB_TOKEN" not in env       # the script's own check reports it better


def test_runs_and_captures_stdout(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/hello.py": """
        import sys
        print("hello", *sys.argv[1:])
    """})
    result = run_script(skill, "scripts/hello.py", _ws(tmp_path), _enabled(), args=["world"])
    assert isinstance(result, ScriptResult)
    assert result.exit_status == 0
    assert result.stdout.strip() == "hello world"
    assert result.stderr == ""


def test_non_zero_exit_returns_normally(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/fail.py": """
        import sys
        print("could not parse line 4", file=sys.stderr)
        sys.exit(3)
    """})
    result = run_script(skill, "scripts/fail.py", _ws(tmp_path), _enabled())
    assert result.exit_status == 3
    assert "could not parse" in result.stderr


def test_stdin_is_passed_through(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/echo.py": """
        import sys
        sys.stdout.write(sys.stdin.read().upper())
    """})
    result = run_script(skill, "scripts/echo.py", _ws(tmp_path), _enabled(), stdin="quiet")
    assert result.stdout == "QUIET"


def test_omitted_stdin_gives_eof_not_a_hang(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/echo.py": """
        import sys
        print(repr(sys.stdin.read()))
    """})
    result = run_script(skill, "scripts/echo.py", _ws(tmp_path), _enabled(), timeout=5)
    assert result.stdout.strip() == "''"


def test_cwd_is_the_workspace_root(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/write.py": """
        from pathlib import Path
        Path("made-here.txt").write_text("ok", encoding="utf-8")
        print("done")
    """})
    ws = _ws(tmp_path)
    run_script(skill, "scripts/write.py", ws, _enabled())
    assert (ws.root / "made-here.txt").read_text(encoding="utf-8") == "ok"


def test_output_is_capped(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/loud.py": 'print("x" * 5000)\n'})
    result = run_script(skill, "scripts/loud.py", _ws(tmp_path), _enabled(max_output_bytes=1000))
    assert len(result.stdout.encode("utf-8")) <= 1000
    assert result.stdout_dropped > 0


def test_output_cap_lands_on_a_character_boundary(tmp_path):
    """``test_output_is_capped`` uses pure ASCII, so it would also pass a naive
    character-slice (``text[:limit]``) implementation and a naive dropped count
    of ``len(raw) - limit``. Multi-byte content whose cutoff falls mid-character
    pins the actual byte-boundary requirement: "e"-acute is two UTF-8 bytes, so a
    five-byte cap keeps two whole characters (four bytes) and drops the dangling
    lead byte of the third along with everything after it.
    """
    skill = _skill(tmp_path, scripts={"scripts/loud.py": 'print("\\u00e9" * 10, end="")\n'})
    result = run_script(skill, "scripts/loud.py", _ws(tmp_path), _enabled(max_output_bytes=5))
    assert result.stdout == "éé"
    assert len(result.stdout.encode("utf-8")) <= 5
    assert result.stdout_dropped == 16  # 20 bytes total, minus the 4 kept


def test_arguments_are_never_shell_interpreted(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/args.py": """
        import sys
        print(sys.argv[1])
    """})
    result = run_script(
        skill, "scripts/args.py", _ws(tmp_path), _enabled(), args=["; echo pwned"]
    )
    assert result.stdout.strip() == "; echo pwned"  # one odd string, not a command


def test_environment_reaches_the_child(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    skill = _skill(tmp_path, scripts={"scripts/dump.py": """
        import json, os
        print(json.dumps(dict(os.environ)))
    """})
    result = run_script(skill, "scripts/dump.py", _ws(tmp_path), _enabled())
    env = json.loads(result.stdout)
    assert "ANTHROPIC_API_KEY" not in env
    assert env["AGENTHARNESS_SKILL_DIR"] == str(skill.path)


def test_disabled_policy_refuses_to_run(tmp_path):
    skill = _skill(tmp_path, scripts={"scripts/hello.py": 'print("hi")\n'})
    with pytest.raises(ValueError) as exc:
        run_script(skill, "scripts/hello.py", _ws(tmp_path), ScriptPolicy())
    assert "disabled" in str(exc.value)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_timeout_kills_the_whole_process_group(tmp_path):
    """A script that forked must not leave its child running after the timeout.

    subprocess's own timeout kills the direct child only, so this asserts on the
    grandchild's PID rather than merely on our having stopped waiting.
    """
    skill = _skill(tmp_path, scripts={"scripts/forker.py": """
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        print(child.pid, flush=True)
        time.sleep(60)
    """})
    with pytest.raises(ScriptTimeout) as exc:
        run_script(skill, "scripts/forker.py", _ws(tmp_path), _enabled(), timeout=1)

    assert exc.value.timeout == 1
    grandchild = int(exc.value.stdout.strip())   # partial output survived the kill
    deadline = time.time() + 5                   # reparenting and reaping is not instant
    while time.time() < deadline and _alive(grandchild):
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_escaped_grandchild_output_is_recovered_not_discarded(tmp_path, monkeypatch):
    """The rescue ``communicate(timeout=5)`` after the kill can itself time out,
    if a grandchild escaped the process group and is still holding the pipes
    open. Reaching that for real needs a grandchild that calls ``setsid()``
    itself to escape — a race that would make the test flaky. Faking both
    ``communicate()`` calls pins the same code path deterministically: CPython's
    own ``TimeoutExpired`` carries whatever it had already buffered, and that
    must survive (capped, like every other path) rather than being thrown away
    for empty strings, and the pipes must still get closed so a dangling
    ``Popen`` doesn't warn on garbage collection.
    """
    seen = []

    def fake_communicate(self, input=None, timeout=None):
        seen.append(self)
        if len(seen) == 1:
            raise subprocess.TimeoutExpired(cmd=["x"], timeout=timeout)
        raise subprocess.TimeoutExpired(
            cmd=["x"], timeout=timeout, output=b"partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr(subprocess.Popen, "communicate", fake_communicate)
    monkeypatch.setattr("agentharness.skills.runner._kill_group", lambda proc: None)

    skill = _skill(tmp_path, scripts={"scripts/hello.py": 'print("hi")\n'})
    with pytest.raises(ScriptTimeout) as exc:
        run_script(
            skill, "scripts/hello.py", _ws(tmp_path), _enabled(max_output_bytes=5), timeout=1
        )

    # Capped, not the raw buffered text -- this path goes through _truncate too.
    assert exc.value.stdout == "parti"
    assert exc.value.stderr == "parti"

    proc = seen[-1]
    assert proc.stdin.closed
    assert proc.stdout.closed
    assert proc.stderr.closed


def test_timeout_carries_accurate_dropped_byte_counts(tmp_path):
    """When a script floods output past max_output_bytes and times out, the
    exception should carry the actual dropped byte counts, not zeros.

    This ensures the rendering layer can tell the model when output was truncated.
    """
    # Craft a script that writes well past max_output_bytes before hanging.
    # With max_output_bytes=100, writing 200 bytes guarantees truncation.
    skill = _skill(tmp_path, scripts={"scripts/flooder.py": """
        import sys, time
        # Write 200 bytes of output
        sys.stdout.write("x" * 200)
        sys.stdout.flush()
        # Then sleep long enough to be killed
        time.sleep(60)
    """})
    
    with pytest.raises(ScriptTimeout) as exc:
        run_script(
            skill,
            "scripts/flooder.py",
            _ws(tmp_path),
            _enabled(max_output_bytes=100),
            timeout=1
        )
    
    # The exception should carry the real dropped count, not 0. We wrote 200
    # bytes and kept 100, so exactly 100 must be reported dropped — a wide
    # band here would admit the very bug this test exists to catch (e.g. a
    # dropped count computed from the wrong buffer, or one that's merely
    # nonzero without being accurate).
    assert exc.value.stdout_dropped == 100
