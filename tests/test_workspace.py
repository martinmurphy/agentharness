"""Tests for the workspace core: confinement, caps, and the read/write helpers."""

from __future__ import annotations

import os

import pytest

from agentharness.workspace import (
    MAX_ENTRIES,
    MAX_READ_BYTES,
    MAX_WRITE_BYTES,
    Workspace,
    list_dir,
    make_dir,
    read_file,
    resolve_in,
    write_file,
)


def _ws(tmp_path, *, writable=True, create=True):
    root = tmp_path / "workspace"
    if create:
        root.mkdir()
    return Workspace(root=root, writable=writable)


# ---- confinement ---------------------------------------------------------


def test_traversal_rejected(tmp_path):
    ws = _ws(tmp_path)
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        read_file(ws, "../secret.txt")


def test_absolute_path_rejected(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        read_file(ws, "/etc/hostname")


def test_symlink_escape_rejected(tmp_path):
    ws = _ws(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    try:
        os.symlink(outside, ws.root / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ValueError, match="escapes"):
        read_file(ws, "link.txt")


def test_write_through_escaping_symlink_rejected(tmp_path):
    """The confinement check runs before any file is opened, writes included."""
    ws = _ws(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("original", encoding="utf-8")
    try:
        os.symlink(outside, ws.root / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(ValueError, match="escapes"):
        write_file(ws, "link.txt", "clobbered")
    assert outside.read_text(encoding="utf-8") == "original"


def test_nested_path_inside_workspace_allowed(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "a" / "b").mkdir(parents=True)
    (ws.root / "a" / "b" / "c.txt").write_text("deep", encoding="utf-8")
    assert read_file(ws, "a/b/c.txt") == "deep"
    assert resolve_in(ws, "a/../a/b/c.txt") == ws.root / "a" / "b" / "c.txt"


def test_empty_path_rejected(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        resolve_in(ws, "")


def test_missing_root_reported_clearly(tmp_path):
    ws = _ws(tmp_path, create=False)
    with pytest.raises(ValueError, match="workspace directory does not exist"):
        read_file(ws, "notes.md")
    with pytest.raises(ValueError, match="workspace directory does not exist"):
        list_dir(ws)


# ---- read_file -----------------------------------------------------------


def test_read_file(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "notes.md").write_text("hello", encoding="utf-8")
    assert read_file(ws, "notes.md") == "hello"


def test_read_missing_file(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="no such file"):
        read_file(ws, "nope.md")


def test_read_directory_rejected(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "sub").mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        read_file(ws, "sub")


def test_read_oversize_file_rejected(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "big.txt").write_text("x" * (MAX_READ_BYTES + 1), encoding="utf-8")
    with pytest.raises(ValueError, match="file too large"):
        read_file(ws, "big.txt")


def test_read_non_utf8_rejected(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        read_file(ws, "blob.bin")


# ---- list_dir ------------------------------------------------------------


def test_list_dir_marks_types_and_sizes(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "reports").mkdir()
    (ws.root / "notes.md").write_text("hello", encoding="utf-8")
    out = list_dir(ws)
    assert "dir   reports/" in out
    assert "file  notes.md  5" in out


def test_list_dir_empty(tmp_path):
    assert "empty" in list_dir(_ws(tmp_path))


def test_list_dir_recursive(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "a").mkdir()
    (ws.root / "a" / "b.txt").write_text("x", encoding="utf-8")
    assert "a/b.txt" not in list_dir(ws)
    assert "a/b.txt" in list_dir(ws, ".", recursive=True)


def test_list_dir_subdirectory(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "a").mkdir()
    (ws.root / "a" / "b.txt").write_text("x", encoding="utf-8")
    assert "file  b.txt  1" in list_dir(ws, "a")


def test_list_dir_on_file_rejected(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "notes.md").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="is a file"):
        list_dir(ws, "notes.md")


def test_list_dir_missing_rejected(tmp_path):
    with pytest.raises(ValueError, match="no such directory"):
        list_dir(_ws(tmp_path), "nope")


def test_list_dir_truncates(tmp_path):
    ws = _ws(tmp_path)
    for i in range(MAX_ENTRIES + 5):
        (ws.root / f"f{i:05d}.txt").write_text("x", encoding="utf-8")
    out = list_dir(ws)
    assert out.splitlines()[-1] == "... truncated: 5 more entries"
    assert len(out.splitlines()) == MAX_ENTRIES + 1


# ---- write_file ----------------------------------------------------------


def test_write_file_overwrite_is_default(tmp_path):
    ws = _ws(tmp_path)
    write_file(ws, "notes.md", "first")
    result = write_file(ws, "notes.md", "second")
    assert (ws.root / "notes.md").read_text(encoding="utf-8") == "second"
    assert "wrote 6 bytes to notes.md" == result


def test_write_file_append(tmp_path):
    ws = _ws(tmp_path)
    write_file(ws, "log.txt", "a")
    result = write_file(ws, "log.txt", "b", "append")
    assert (ws.root / "log.txt").read_text(encoding="utf-8") == "ab"
    assert result.startswith("appended")


def test_write_file_append_creates_missing_file(tmp_path):
    ws = _ws(tmp_path)
    write_file(ws, "log.txt", "a", "append")
    assert (ws.root / "log.txt").read_text(encoding="utf-8") == "a"


def test_write_file_create_refuses_existing(tmp_path):
    ws = _ws(tmp_path)
    write_file(ws, "notes.md", "first")
    with pytest.raises(ValueError, match="already exists"):
        write_file(ws, "notes.md", "second", "create")
    assert (ws.root / "notes.md").read_text(encoding="utf-8") == "first"


def test_write_file_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="unknown mode"):
        write_file(_ws(tmp_path), "notes.md", "x", "clobber")


def test_write_file_missing_parent_names_make_dir(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="make_dir"):
        write_file(ws, "reports/2026/q3.md", "x")
    assert not (ws.root / "reports").exists()


def test_write_file_onto_directory_rejected(tmp_path):
    ws = _ws(tmp_path)
    (ws.root / "sub").mkdir()
    with pytest.raises(ValueError, match="is a directory"):
        write_file(ws, "sub", "x")


def test_write_file_oversize_rejected(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="content too large"):
        write_file(ws, "big.txt", "x" * (MAX_WRITE_BYTES + 1))
    assert not (ws.root / "big.txt").exists()


def test_write_file_unicode_round_trip(tmp_path):
    ws = _ws(tmp_path)
    result = write_file(ws, "u.md", "héllo")  # 6 bytes, 5 characters
    assert "wrote 6 bytes" in result
    assert read_file(ws, "u.md") == "héllo"


# ---- make_dir ------------------------------------------------------------


def test_make_dir_creates_parents(tmp_path):
    ws = _ws(tmp_path)
    result = make_dir(ws, "reports/2026")
    assert (ws.root / "reports" / "2026").is_dir()
    assert result == "directory created: reports/2026"
    write_file(ws, "reports/2026/q3.md", "ok")  # the parent now exists


def test_make_dir_is_idempotent(tmp_path):
    ws = _ws(tmp_path)
    make_dir(ws, "reports")
    assert "already exists" in make_dir(ws, "reports")


def test_make_dir_onto_file_rejected(tmp_path):
    ws = _ws(tmp_path)
    write_file(ws, "notes.md", "x")
    with pytest.raises(ValueError, match="is a file"):
        make_dir(ws, "notes.md")


def test_make_dir_creates_the_root_itself(tmp_path):
    """The one sanctioned way a missing workspace root comes into existence."""
    ws = _ws(tmp_path, create=False)
    assert not ws.root.exists()
    make_dir(ws, "reports")
    assert (ws.root / "reports").is_dir()


def test_make_dir_cannot_escape(tmp_path):
    ws = _ws(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        make_dir(ws, "../evil")
    assert not (tmp_path / "evil").exists()
