#!/usr/bin/env python3
"""PostToolUse hook: after Edit/Write/MultiEdit touches a *.rs file, run `cargo fmt --all`.

Reads the hook JSON payload from stdin. Never blocks the turn -- formatting
failures (e.g. no Cargo.toml yet, syntax error mid-edit) are swallowed.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def touched_path(payload: dict) -> str:
    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    # MultiEdit puts file_path in tool_input too; single-edit fallback to tool_response.
    return tool_input.get("file_path") or tool_response.get("filePath") or ""


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0

    path = touched_path(payload)
    if not path.endswith(".rs"):
        return 0

    try:
        subprocess.run(
            ["cargo", "fmt", "--all"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            timeout=30,
        )
    except Exception:
        pass  # never block on formatting failures
    return 0


if __name__ == "__main__":
    sys.exit(main())
