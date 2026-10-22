"""Local process helper for the sync fixture and CLI exit-code check."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_init_fixture(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def git_empty_commit(path: Path) -> None:
    subprocess.run([
        "git", "-C", str(path), "-c", "user.name=Test",
        "-c", "user.email=test@example.invalid", "commit", "-q",
        "--allow-empty", "-m", "fixture",
    ], check=True)


def run_sync_check(script: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(script), "las-voces", "--check"],
        capture_output=True, text=True,
    )
