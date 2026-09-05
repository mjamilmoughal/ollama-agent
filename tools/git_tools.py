"""Read-only git history tools -- log, diff, and blame -- scoped to the project's repo.

Every function here shells out to `git` with a fixed, hardcoded subcommand and
argument list (never a raw string from the model), run with cwd=PROJECT_ROOT and
no shell interpretation, so there's no command-injection surface and no path to a
mutating git operation (checkout, reset, add, commit, push, ...).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from ._project import PROJECT_ROOT

MAX_OUTPUT_CHARS = 8_000
MAX_LOG_ENTRIES = 50
MAX_BLAME_LINES = 400
GIT_TIMEOUT_SECONDS = 15


def _run_git(args: list[str]) -> tuple[bool, str]:
    """Run a read-only git subcommand scoped to PROJECT_ROOT. Returns (ok, output)."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return False, "git is not installed or not on PATH."
    except subprocess.TimeoutExpired:
        return False, "git command timed out."
    if result.returncode != 0:
        return False, (result.stderr or result.stdout or "git command failed").strip()
    return True, result.stdout


def _resolve_tracked_path(path: str) -> str | None:
    """Resolve a path for use as a git pathspec: must exist and be inside PROJECT_ROOT."""
    path = (path or "").strip().strip('"')
    if not path:
        return None
    candidate = Path(path) if os.path.isabs(path) else (PROJECT_ROOT / path)
    resolved = candidate.resolve()
    if not resolved.exists():
        return None
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return None
    return str(resolved)


@tool
def git_log(path: str = "", limit: int = 10) -> str:
    """Show recent commit history for the project, or for one file.

    Leave `path` empty for the whole repo's history, or pass a file path
    (absolute, or relative to the project) to see only commits that touched
    that file. Each line is hash | date | author | subject.
    """
    limit = max(1, min(limit, MAX_LOG_ENTRIES))
    args = ["log", f"-{limit}", "--date=short", "--pretty=format:%h | %ad | %an | %s"]
    if path:
        resolved = _resolve_tracked_path(path)
        if resolved is None:
            return f"'{path}' does not exist in this project."
        args += ["--", resolved]

    ok, output = _run_git(args)
    if not ok:
        return f"Could not get git log: {output}"
    if not output.strip():
        return f"No commit history found{f' for {path}' if path else ''}."
    header = f"Recent commits{f' for {path}' if path else ''}:"
    return f"{header}\n{output}"[:MAX_OUTPUT_CHARS]


@tool
def git_diff(path: str = "", staged: bool = False) -> str:
    """Show uncommitted changes in the project's working tree.

    Leave `path` empty to see every changed file, or pass one file path
    (absolute, or relative to the project) to see just its diff. Set
    staged=True to see changes already staged for commit instead of
    unstaged working-tree changes.
    """
    args = ["diff", "--cached"] if staged else ["diff"]
    if path:
        resolved = _resolve_tracked_path(path)
        if resolved is None:
            return f"'{path}' does not exist in this project."
        args += ["--", resolved]

    ok, output = _run_git(args)
    if not ok:
        return f"Could not get git diff: {output}"
    if not output.strip():
        kind = "staged" if staged else "unstaged"
        return f"No {kind} changes{f' in {path}' if path else ''}."
    truncated = output[:MAX_OUTPUT_CHARS]
    if len(output) > MAX_OUTPUT_CHARS:
        truncated += "\n... [diff truncated]"
    return truncated


@tool
def git_blame(path: str, start_line: int = 0, end_line: int = 0) -> str:
    """Show per-line authorship (who last changed each line, and when) for a file.

    `path` is required (absolute, or relative to the project). Optionally pass
    start_line/end_line (1-based, inclusive) to limit blame to a range; leave
    both at 0 for the whole file, capped at the first 400 lines.
    """
    resolved = _resolve_tracked_path(path)
    if resolved is None:
        return f"'{path}' does not exist in this project."

    args = ["blame", "--date=short"]
    if start_line and end_line:
        args += ["-L", f"{start_line},{end_line}"]
    args += ["--", resolved]

    ok, output = _run_git(args)
    if not ok:
        return f"Could not get git blame for '{path}': {output}"

    lines = output.splitlines()
    truncated_note = ""
    if len(lines) > MAX_BLAME_LINES:
        lines = lines[:MAX_BLAME_LINES]
        truncated_note = f"\n... [truncated to {MAX_BLAME_LINES} lines; narrow with start_line/end_line]"
    return f"Blame for '{path}':\n" + "\n".join(lines) + truncated_note
