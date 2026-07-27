"""The workspace: a bind-mounted directory the filesystem tools are confined to.

This is the safety-critical core, kept free of any Tool imports so it can be
tested on its own; ``tools/fs_tools.py`` wraps it. The split mirrors
``skills/loader.py`` (which owns ``read_skill_file``) vs ``tools/skill_tools.py``.

Every path argument here comes from the model and is untrusted. ``resolve_in``
is the single choke point: resolve the joined path and require it to stay under
the resolved root, which rejects ``..`` traversal, absolute paths, and symlink
escapes together. Nothing below touches the filesystem before that check passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MAX_READ_BYTES = 256 * 1024  # matches skills.loader.MAX_FILE_BYTES
MAX_WRITE_BYTES = 1024 * 1024  # cap on content accepted in one write_file call
MAX_ENTRIES = 1000  # cap on the number of entries list_dir reports

WRITE_MODES = ("overwrite", "append", "create")


@dataclass(frozen=True)
class Workspace:
    """The root the filesystem tools operate in, plus whether writes are allowed.

    ``writable`` gates *tool registration* (see ``make_fs_tools``), so a
    read-only harness never advertises the write tools at all.
    """

    root: Path
    writable: bool = True

    def describe(self) -> str:
        mode = "read-write" if self.writable else "read-only"
        exists = "" if self.root.is_dir() else " (does not exist)"
        return f"{self.root}{exists} [{mode}]"


def resolve_in(ws: Workspace, rel_path: str) -> Path:
    """Resolve ``rel_path`` against the workspace root, refusing to escape it.

    ``resolve()`` is non-strict, so this works for paths that don't exist yet
    (a file about to be written). An absolute ``rel_path`` makes ``/`` discard
    the root entirely — the ``is_relative_to`` check is what catches that.
    """
    if not isinstance(rel_path, str) or not rel_path:
        raise ValueError("'path' is required and must be a non-empty string")
    base = ws.root.resolve()
    target = (ws.root / rel_path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"path {rel_path!r} escapes the workspace")
    return target


def _require_root(ws: Workspace) -> None:
    if not ws.root.is_dir():
        raise ValueError(f"workspace directory does not exist: {ws.root}")


def _rel(ws: Workspace, target: Path) -> str:
    """Display form of ``target``: relative to the root, '.' for the root itself."""
    rel = target.resolve().relative_to(ws.root.resolve())
    return str(rel) if str(rel) != "." else "."


def read_file(ws: Workspace, rel_path: str) -> str:
    """Read a UTF-8 text file inside the workspace."""
    _require_root(ws)
    target = resolve_in(ws, rel_path)
    if target.is_dir():
        raise ValueError(f"path is a directory, not a file: {rel_path}")
    if not target.is_file():
        raise ValueError(f"no such file: {rel_path}")
    if target.stat().st_size > MAX_READ_BYTES:
        raise ValueError(f"file too large (> {MAX_READ_BYTES} bytes): {rel_path}")
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"file is not valid UTF-8 text: {rel_path}") from exc


def list_dir(ws: Workspace, rel_path: str = ".", *, recursive: bool = False) -> str:
    """List a directory inside the workspace, one entry per line.

    ``rglob`` does not follow directory symlinks, so a link pointing outside
    cannot be walked into. Links are still *named*, which is harmless — reading
    through one is rejected by ``resolve_in``.
    """
    _require_root(ws)
    target = resolve_in(ws, rel_path)
    if not target.exists():
        raise ValueError(f"no such directory: {rel_path}")
    if not target.is_dir():
        raise ValueError(f"path is a file, not a directory: {rel_path}")

    entries = sorted(target.rglob("*") if recursive else target.iterdir())
    lines: list[str] = []
    for entry in entries[:MAX_ENTRIES]:
        name = str(entry.relative_to(target))
        if entry.is_dir():
            lines.append(f"dir   {name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:  # broken symlink
                lines.append(f"file  {name}  (unreadable)")
                continue
            lines.append(f"file  {name}  {size}")
    if len(entries) > MAX_ENTRIES:
        lines.append(f"... truncated: {len(entries) - MAX_ENTRIES} more entries")
    if not lines:
        return f"{_rel(ws, target)} is empty"
    return "\n".join(lines)


def write_file(ws: Workspace, rel_path: str, content: str, mode: str = "overwrite") -> str:
    """Write text to a file inside the workspace.

    Parent directories are deliberately not created — a missing parent is an
    error naming ``make_dir``, so a mistyped path fails loudly instead of
    silently growing a tree.
    """
    _require_root(ws)
    if mode not in WRITE_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose one of {list(WRITE_MODES)}")
    if not isinstance(content, str):
        raise ValueError("'content' is required and must be a string")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        raise ValueError(
            f"content too large ({len(encoded)} bytes > {MAX_WRITE_BYTES}): {rel_path}"
        )

    target = resolve_in(ws, rel_path)
    if target.is_dir():
        raise ValueError(f"path is a directory, not a file: {rel_path}")
    if mode == "create" and target.exists():
        raise ValueError(f"file already exists: {rel_path} (use mode 'overwrite' to replace it)")
    if not target.parent.is_dir():
        raise ValueError(
            f"parent directory does not exist: {rel_path} (create it with make_dir first)"
        )

    with target.open("a" if mode == "append" else "w", encoding="utf-8") as fh:
        fh.write(content)
    verb = "appended" if mode == "append" else "wrote"
    return f"{verb} {len(encoded)} bytes to {_rel(ws, target)}"


def make_dir(ws: Workspace, rel_path: str) -> str:
    """Create a directory inside the workspace, parents included. Idempotent.

    This is the one sanctioned way the workspace root itself comes into
    existence, so it does not call ``_require_root``.
    """
    target = resolve_in(ws, rel_path)
    if target.is_file():
        raise ValueError(f"path is a file, not a directory: {rel_path}")
    existed = target.is_dir()
    target.mkdir(parents=True, exist_ok=True)
    return f"directory {'already exists' if existed else 'created'}: {_rel(ws, target)}"
