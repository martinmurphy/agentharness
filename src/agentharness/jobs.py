"""Background turns: run a prompt off the prompt, and keep its output.

A turn that takes a minute owns the terminal for a minute. A job is the way out:
``summarise the repo &`` hands the turn to a worker thread and gives the prompt
straight back, so the next thing can start on another state.

Two rules shape everything here.

*One in-flight turn per state.* Two turns appending to one ``state.messages``
interleave into a transcript neither of them wrote — an assistant message
answering the wrong user message. Concurrency comes from using several states,
which the harness already has, so a second prompt on a busy state is refused
rather than queued.

*Output is captured, never printed.* A job that printed as it went would scribble
over whatever is being typed at the prompt, because readline owns that line and
knows nothing about the writer. So a job's rendered lines go into its own buffer,
replayed on demand by ``/job <id>``; only a one-line completion notice reaches
the terminal, and even that waits for the top of the REPL loop.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal

JobStatus = Literal["running", "done", "failed"]

# What a job actually does: run a turn, rendering into the sink it is handed.
JobBody = Callable[[Callable[[str], None]], None]


@dataclass
class Job:
    id: int
    state_name: str
    prompt: str
    started_at: float
    status: JobStatus = "running"
    finished_at: float | None = None
    # Rendered output, captured rather than printed. Appended by the worker,
    # read by the REPL thread; both operations are atomic on a list, so no lock
    # is needed for a consistent (if possibly mid-turn) view.
    lines: list[str] = field(default_factory=list)
    error: str | None = None

    def output(self) -> list[str]:
        return list(self.lines)

    def elapsed(self, now: float | None = None) -> float:
        end = self.finished_at if self.finished_at is not None else (now or time.monotonic())
        return end - self.started_at


class JobRunner:
    """Owns the worker pool, the job table, and the completion notices."""

    def __init__(self, max_jobs: int = 4) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_jobs), thread_name_prefix="job"
        )
        self._lock = threading.Lock()
        self._jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._notices: list[str] = []
        self._futures: list[Future[None]] = []

    # ---- submission ---------------------------------------------------------

    def submit(self, state_name: str, prompt: str, body: JobBody) -> Job:
        """Start ``body`` on a worker, or raise if that state is already busy.

        The busy check is inside the lock rather than left to the caller's
        ``busy()``: the point of the rule is that it cannot be raced, and a
        caller that checked first would only *usually* be right.
        """
        with self._lock:
            running = self._busy_locked(state_name)
            if running is not None:
                raise ValueError(f"{state_name} is already running job {running.id}")
            job = Job(
                id=next(self._ids),
                state_name=state_name,
                prompt=prompt,
                started_at=time.monotonic(),
            )
            self._jobs[job.id] = job
            future = self._pool.submit(self._run, job, body)
            self._futures.append(future)
        return job

    def _run(self, job: Job, body: JobBody) -> None:
        error: str | None = None
        try:
            body(job.lines.append)
        except BaseException as exc:  # noqa: BLE001 - a job must not die silently
            # run_turn_into already renders provider failures into the buffer,
            # so reaching here means a bug in the harness rather than a bad
            # turn. Record it against the job instead of losing it on a worker.
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            job.status = "failed" if error else "done"
            job.error = error
            job.finished_at = time.monotonic()
            note = f"[job {job.id}] {job.status} — /job {job.id} for output"
            if error:
                note += f" ({error})"
            self._notices.append(note)

    # ---- queries ------------------------------------------------------------

    def busy(self, state_name: str) -> Job | None:
        """The job currently running on this state, if any."""
        with self._lock:
            return self._busy_locked(state_name)

    def _busy_locked(self, state_name: str) -> Job | None:
        return next(
            (
                j
                for j in self._jobs.values()
                if j.state_name == state_name and j.status == "running"
            ),
            None,
        )

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def any_running(self) -> bool:
        with self._lock:
            return any(j.status == "running" for j in self._jobs.values())

    def drain_notices(self) -> list[str]:
        """Completion notices since the last call, taken not copied."""
        with self._lock:
            notices, self._notices = self._notices, []
        return notices

    # ---- teardown -----------------------------------------------------------

    def shutdown(self, timeout: float = 2.0) -> list[Job]:
        """Stop accepting work, wait briefly, and report what is still running.

        Deliberately bounded. A job blocked in a tool call cannot be
        interrupted, and the alternative — an unbounded wait — is a REPL whose
        ``/quit`` never returns. The caller names the stragglers and leaves.
        """
        self._pool.shutdown(wait=False, cancel_futures=True)
        deadline = time.monotonic() + timeout
        for future in list(self._futures):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                future.result(timeout=remaining)
            except (TimeoutError, CancelledError):
                # A straggler, or one cancelled before it started. Both show up
                # in the returned list; a job that *failed* recorded itself.
                pass
        with self._lock:
            return [j for j in self._jobs.values() if j.status == "running"]
