"""Dependency-free helpers for reproducible experiment manifests."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _git(repo_root: str, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    # Preserve porcelain-status leading columns; only remove line endings.
    return completed.stdout.rstrip()


def collect_source_revision(repo_root: str | None = None) -> dict[str, Any]:
    """Return the tracked source state used to launch an experiment."""

    root = os.path.realpath(repo_root or os.path.dirname(__file__))
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    tracked_changes = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "tracked_worktree_dirty": bool(tracked_changes),
        "tracked_changes": tracked_changes.splitlines() if tracked_changes else [],
    }
