#!/usr/bin/env python3
"""Stop hook: fail-fast Rust quality gate, run only when Rust files changed.

Chain: cargo fmt --check -> cargo clippy --all-targets -- -D warnings -> cargo test
Runs only if `git status --porcelain` shows a modified/staged/untracked *.rs file
(so it's silent on turns that never touched Rust code). On failure, blocks the
stop and surfaces the failing command's output back to the user/model instead
of swallowing it.
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GATE = [
    ["cargo", "fmt", "--all", "--check"],
    ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
    ["cargo", "test"],
]


def rust_files_changed() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return False
    return any(
        line[3:].strip().endswith(".rs")
        for line in result.stdout.splitlines()
        if line.strip()
    )


def has_cargo_workspace() -> bool:
    return (PROJECT_ROOT / "Cargo.toml").exists()


def main() -> int:
    # Drain stdin JSON (unused, but hooks receive it on stdin).
    try:
        json.load(sys.stdin)
    except Exception:
        pass

    if not has_cargo_workspace() or not rust_files_changed():
        return 0

    for cmd in GATE:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as exc:
            print(json.dumps({
                "decision": "block",
                "reason": f"Could not run `{' '.join(cmd)}`: {exc}",
            }))
            return 0

        if result.returncode != 0:
            tail = (result.stdout + result.stderr)[-4000:]
            print(json.dumps({
                "decision": "block",
                "reason": f"`{' '.join(cmd)}` failed:\n{tail}",
            }))
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
