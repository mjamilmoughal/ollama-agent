"""A directory-tree tool for getting oriented in a project or folder."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from ._project import PROJECT_ROOT

MAX_ENTRIES = 300
MAX_DEPTH = 5

EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


def _resolve(path: str) -> Path:
    path = (path or "").strip().strip('"')
    if not path:
        return PROJECT_ROOT
    candidate = Path(path) if os.path.isabs(path) else (PROJECT_ROOT / path)
    return candidate.resolve()


def _build_tree(root: Path, max_depth: int, show_hidden: bool, counter: list[int]) -> list[str]:
    lines: list[str] = []

    def walk(dir_path: Path, depth: int, prefix: str) -> None:
        if depth > max_depth or counter[0] >= MAX_ENTRIES:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except OSError:
            return
        entries = [
            e for e in entries
            if (show_hidden or not e.name.startswith("."))
            and not (e.is_dir() and e.name.lower() in EXCLUDED_DIR_NAMES)
        ]
        for i, entry in enumerate(entries):
            if counter[0] >= MAX_ENTRIES:
                lines.append(f"{prefix}... [truncated at {MAX_ENTRIES} entries]")
                return
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            name = entry.name + "/" if entry.is_dir() else entry.name
            lines.append(f"{prefix}{connector}{name}")
            counter[0] += 1
            if entry.is_dir():
                walk(entry, depth + 1, prefix + ("    " if is_last else "│   "))

    walk(root, 1, "")
    return lines


@tool
def list_directory(path: str = "", max_depth: int = 2, show_hidden: bool = False) -> str:
    """Show the folder/file structure of a directory as a tree.

    Pass a path (absolute, or relative to the current project); leave empty
    to list the project root. Use this to get oriented in an unfamiliar
    project before searching for a specific file by name with
    find_project_file. Depth and result count are bounded, and common noisy
    directories (.git, node_modules, __pycache__, venv, build artifacts) are
    skipped automatically -- set show_hidden=True to include dotfiles too.
    """
    root = _resolve(path)
    if not root.exists():
        return f"'{path or '.'}' does not exist."
    if not root.is_dir():
        return f"'{path}' is a file, not a directory. Use read_file to read it."

    max_depth = max(1, min(max_depth, MAX_DEPTH))
    counter = [0]
    lines = _build_tree(root, max_depth, show_hidden, counter)

    try:
        label = str(root.relative_to(PROJECT_ROOT)) or "."
    except ValueError:
        label = str(root)
    header = f"{label}/ (depth={max_depth}{', showing hidden' if show_hidden else ''}):"
    if not lines:
        return f"{header}\n(empty)"
    return "\n".join([header, *lines])
