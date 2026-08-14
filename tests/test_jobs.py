"""Background jobs: the runner, and the REPL surface over it.

Nothing here asserts on elapsed time. Where a test needs a job to be *running*
it holds it open on an event and releases it explicitly; where it needs one to
be finished it waits for that condition with a timeout.
"""

from __future__ import annotations

import threading
import time

import pytest

from agentharness.config import Config
from agentharness.jobs import JobRunner
from agentharness.providers.base import Message, ProviderResponse, TextBlock, Usage

TIMEOUT = 5.0


@pytest.fixture
def runner():
    r = JobRunner(max_jobs=4)
    yield r
    r.shutdown(TIMEOUT)


def _await_finish(job, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.005)
    assert job.status != "running", f"job {job.id} never finished"


# ---- the runner -------------------------------------------------------------


def test_submit_records_and_runs_the_job(runner):
    job = runner.submit("default", "summarise the repo", lambda write: write("hello"))
    _await_finish(job)
    assert job.status == "done"
    assert job.output() == ["hello"]
    assert job.state_name == "default"
    assert runner.get(job.id) is job
    assert [j.id for j in runner.jobs()] == [job.id]


def test_get_returns_none_for_an_unknown_id(runner):
    assert runner.get(99) is None


def test_one_in_flight_turn_per_state(runner):
    release = threading.Event()
    first = runner.submit("default", "slow", lambda write: release.wait(TIMEOUT))
    try:
        assert runner.busy("default") is first
        with pytest.raises(ValueError, match="already running job"):
            runner.submit("default", "second", lambda write: None)
    finally:
        release.set()
    _await_finish(first)
    # …and once it is done the state is free again.
    assert runner.busy("default") is None
    runner.submit("default", "third", lambda write: None)


def test_another_state_is_not_blocked(runner):
    release = threading.Event()
    slow = runner.submit("default", "slow", lambda write: release.wait(TIMEOUT))
    try:
        other = runner.submit("scratch", "quick", lambda write: write("done"))
        _await_finish(other)
        assert other.output() == ["done"]
        assert slow.status == "running"
    finally:
        release.set()
    _await_finish(slow)


def test_notices_are_drained_exactly_once(runner):
    job = runner.submit("default", "p", lambda write: None)
    _await_finish(job)
    notices = runner.drain_notices()
    assert len(notices) == 1
    assert f"/job {job.id}" in notices[0]
    assert runner.drain_notices() == []  # taken, not copied


def test_a_body_that_raises_fails_the_job_rather_than_the_worker(runner):
    def boom(write):
        write("got this far")
        raise RuntimeError("kaboom")

    job = runner.submit("default", "p", boom)
    _await_finish(job)
    assert job.status == "failed"
    assert "kaboom" in job.error
    assert job.output() == ["got this far"]  # what it managed is still kept
    assert "kaboom" in runner.drain_notices()[0]


def test_shutdown_reports_what_is_still_running():
    runner = JobRunner(max_jobs=2)
    release = threading.Event()
    job = runner.submit("default", "wedged", lambda write: release.wait(TIMEOUT))
    try:
        stragglers = runner.shutdown(timeout=0.05)
        assert [j.id for j in stragglers] == [job.id]
    finally:
        release.set()  # let the worker go, so it does not outlive the test
    _await_finish(job)


def test_shutdown_of_a_quiet_runner_reports_nothing(runner):
    job = runner.submit("default", "p", lambda write: None)
    _await_finish(job)
    assert runner.shutdown(TIMEOUT) == []


# ---- the REPL surface -------------------------------------------------------


class _Fake:
    """A provider that answers once, blocking first if given an event."""

    name = "fake"

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate = gate

    def chat(self, *, system, messages, tools, max_tokens):
        if self.gate is not None:
            self.gate.wait(TIMEOUT)
        return ProviderResponse(
            message=Message(role="assistant", blocks=[TextBlock("the answer")]),
            stop_reason="end_turn",
            usage=Usage(input_tokens=3, output_tokens=2),
        )


def _harness(tmp_path, monkeypatch, gate=None):
    from agentharness import repl

    monkeypatch.setattr(repl, "build_provider", lambda name, model, config: _Fake(gate))
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    h = repl.Harness(Config(skills_dir=str(tmp_path), workspace_dir=str(workspace)))
    return h, repl


def test_split_background():
    from agentharness.repl import split_background

    assert split_background("summarise the repo &") == ("summarise the repo", True)
    assert split_background("summarise the repo") == ("summarise the repo", False)
    # Only a standalone trailing token — an ampersand inside a prompt is prose.
    assert split_background("tell me about Ada & Grace") == ("tell me about Ada & Grace", False)
    assert split_background("what is R&D") == ("what is R&D", False)


def test_a_background_turn_is_captured_not_printed(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._run_or_background(h, "what is 2+2 &")
    job = h.jobs.get(1)
    _await_finish(job)

    out = capsys.readouterr().out
    assert "[job 1] started on default" in out
    assert "the answer" not in out  # the turn's output never reached the terminal
    assert "the answer" in "\n".join(job.output())
    assert "[usage]" in "\n".join(job.output())  # including its usage line
    assert h.states.active.messages[-1].text() == "the answer"  # ran on that state


def test_a_foreground_prompt_still_prints(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._run_or_background(h, "what is 2+2")
    assert "the answer" in capsys.readouterr().out
    assert h.jobs.jobs() == []


def test_a_second_prompt_on_a_busy_state_is_refused(tmp_path, monkeypatch, capsys):
    gate = threading.Event()
    h, repl = _harness(tmp_path, monkeypatch, gate=gate)
    try:
        repl._run_or_background(h, "slow one &")
        capsys.readouterr()
        repl._run_or_background(h, "another")  # foreground, same state
        out = capsys.readouterr().out
        assert "[busy] default is running job 1" in out
        assert "/switch" in out
        assert "the answer" not in out  # it really did not run
    finally:
        gate.set()
    _await_finish(h.jobs.get(1))


def test_delete_and_reset_refuse_a_busy_state(tmp_path, monkeypatch, capsys):
    gate = threading.Event()
    h, repl = _harness(tmp_path, monkeypatch, gate=gate)
    try:
        repl._run_or_background(h, "slow one &")
        h.states.new("other")  # so `default` is not the last remaining state
        capsys.readouterr()
        repl._handle_command(h, "/delete default")
        assert "[busy]" in capsys.readouterr().out
        assert "default" in h.states.names()

        h.states.switch("default")
        h.states.active.add(Message(role="user", blocks=[TextBlock("keep me")]))
        repl._handle_command(h, "/reset")
        assert "[busy]" in capsys.readouterr().out
        assert h.states.active.messages  # history survived the refusal
    finally:
        gate.set()
    _await_finish(h.jobs.get(1))


def test_jobs_command_lists_and_job_command_replays(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._run_or_background(h, "summarise the repo &")
    _await_finish(h.jobs.get(1))
    capsys.readouterr()

    repl._handle_command(h, "/jobs")
    listing = capsys.readouterr().out
    assert "summarise the repo" in listing and "done" in listing and "default" in listing

    repl._handle_command(h, "/job 1")
    assert "the answer" in capsys.readouterr().out


def test_jobs_command_when_there_are_none(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/jobs")
    assert "no background jobs" in capsys.readouterr().out


def test_job_command_rejects_a_bad_id(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/job 7")
    assert "no such job: 7" in capsys.readouterr().out
    repl._handle_command(h, "/job banana")
    assert "no such job: banana" in capsys.readouterr().out
    repl._handle_command(h, "/job")
    assert "usage: /job <id>" in capsys.readouterr().out


def test_help_mentions_the_job_commands(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/help")
    out = capsys.readouterr().out
    assert "/jobs" in out and "/job <id>" in out and "&" in out


def test_reload_warns_that_a_running_job_keeps_the_old_tools(tmp_path, monkeypatch, capsys):
    gate = threading.Event()
    h, repl = _harness(tmp_path, monkeypatch, gate=gate)
    try:
        repl._run_or_background(h, "slow one &")
        capsys.readouterr()
        repl._handle_command(h, "/reload")
        assert "will finish on the previous tool set" in capsys.readouterr().out
    finally:
        gate.set()
    _await_finish(h.jobs.get(1))


def test_reload_says_nothing_extra_when_no_job_is_running(tmp_path, monkeypatch, capsys):
    h, repl = _harness(tmp_path, monkeypatch)
    repl._handle_command(h, "/reload")
    assert "previous tool set" not in capsys.readouterr().out
