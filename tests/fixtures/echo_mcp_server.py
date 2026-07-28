"""A tiny MCP server over stdio, used as a real peer in the runtime tests.

Kept deliberately small and dependency-free beyond the SDK itself. Run as a
subprocess by tests; not part of the package.
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

server = FastMCP("echo")


@server.tool()
def echo(text: str) -> str:
    """Return the text it was given."""
    return f"echo: {text}"


@server.tool()
def boom() -> str:
    """Always fail, so error mapping can be exercised."""
    raise ValueError("this tool always fails")


@server.tool()
def read_env(name: str) -> str:
    """Report one environment variable, to prove env_pass is an allowlist."""
    return os.environ.get(name, "<unset>")


@server.tool()
def slow(seconds: float) -> str:
    """Sleep, so call timeouts can be exercised."""
    import time

    time.sleep(seconds)
    return "done"


if __name__ == "__main__":
    # Announce our PID so a test can check the process really was reaped, not
    # merely forgotten about.
    pid_file = os.environ.get("ECHO_PID_FILE")
    if pid_file:
        with open(pid_file, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))

    # A server that never handshakes, for connect-timeout tests.
    if "--hang" in sys.argv:
        import time

        while True:
            time.sleep(3600)

    # A slow starter, so concurrent connects can be told from serial ones.
    if "--delay" in sys.argv:
        import time

        time.sleep(float(sys.argv[sys.argv.index("--delay") + 1]))

    server.run()
