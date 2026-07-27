"""Filesystem tools over the bind-mounted workspace directory.

``list_dir`` / ``read_file``   -> always available
``write_file`` / ``make_dir``  -> only when the workspace is writable

The write gate lives here, in one place: a read-only workspace simply doesn't
return the write tools, so the model is never offered a tool that would refuse.
All paths are relative to the workspace root; confinement is enforced by
``agentharness.workspace.resolve_in``.

Handlers raise ValueError on bad input; ToolRegistry.dispatch turns that into a
ToolResult(is_error=True) the model can recover from.
"""

from __future__ import annotations

from typing import Any

from agentharness.tools.registry import Tool
from agentharness.workspace import WRITE_MODES, Workspace, list_dir, make_dir, read_file, write_file


def _require_str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key!r} is required and must be a non-empty string")
    return value


def make_fs_tools(ws: Workspace) -> list[Tool]:
    def _list_dir(args: dict[str, Any]) -> str:
        path = args.get("path", ".")
        if not isinstance(path, str) or not path:
            path = "."
        return list_dir(ws, path, recursive=bool(args.get("recursive", False)))

    def _read_file(args: dict[str, Any]) -> str:
        return read_file(ws, _require_str(args, "path"))

    def _write_file(args: dict[str, Any]) -> str:
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("'content' is required and must be a string")
        return write_file(
            ws, _require_str(args, "path"), content, str(args.get("mode", "overwrite"))
        )

    def _make_dir(args: dict[str, Any]) -> str:
        return make_dir(ws, _require_str(args, "path"))

    shared = "you can read and write" if ws.writable else "you can read"
    tools = [
        Tool(
            name="list_dir",
            description=(
                f"List the contents of a directory in the workspace, the shared folder "
                f"{shared}. Paths are relative to the workspace root; "
                f"use '.' for the root itself."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list, relative to the workspace root.",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "List all nested entries rather than one level.",
                        "default": False,
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_list_dir,
        ),
        Tool(
            name="read_file",
            description=(
                "Read a UTF-8 text file from the workspace, returned as text. "
                "Paths are relative to the workspace root."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to read, relative to the workspace root.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_read_file,
        ),
    ]

    if not ws.writable:
        return tools

    tools.append(
        Tool(
            name="write_file",
            description=(
                "Write text to a file in the workspace. Parent directories are not "
                "created automatically — call make_dir first if the directory is new."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File to write, relative to the workspace root.",
                    },
                    "content": {"type": "string", "description": "Text to write."},
                    "mode": {
                        "type": "string",
                        "enum": list(WRITE_MODES),
                        "description": (
                            "overwrite: replace any existing file. append: add to the end. "
                            "create: fail if the file already exists."
                        ),
                        "default": "overwrite",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_write_file,
        )
    )
    tools.append(
        Tool(
            name="make_dir",
            description=(
                "Create a directory in the workspace, including any missing parent "
                "directories. Succeeds if it already exists."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to create, relative to the workspace root.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_make_dir,
        )
    )
    return tools
