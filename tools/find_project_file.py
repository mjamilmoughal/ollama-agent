"""A project-scoped file/folder search tool -- unlike find_file, needs no location confirmation."""

from __future__ import annotations

import os

from langchain_core.tools import tool

from ._file_search import format_results, search_tree
from ._project import PROJECT_ROOT

EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", "venv", ".venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".ruff_cache",
}


@tool
def find_project_file(name: str, kind: str = "any") -> str:
    """Search the current project directory for a file or folder by name.

    Use this whenever the user refers to "this project", "this repo", "the
    current folder", or otherwise expects a file to already be part of what
    you're working in. It only looks inside the project directory, so unlike
    find_file it never needs the user to confirm a location first. For
    anything outside the project, use find_file instead.

    Args:
        name: the file or folder name to look for. Matching is fuzzy and
              case-insensitive, so a partial name or a close guess is fine.
              Include the extension if the user gave one (e.g. "app.py").
        kind: "file" to only match files, "folder" to only match folders, or
              "any" (default) to match either.
    """
    name = (name or "").strip()
    if not name:
        return "Missing a file/folder name to search for."

    kind = (kind or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    matches, truncated = search_tree(str(PROJECT_ROOT), name, kind, EXCLUDED_DIR_NAMES)
    relative_matches = [(tier, os.path.relpath(path, PROJECT_ROOT), k) for tier, path, k in matches]
    return format_results(relative_matches, name, kind, f"the project ({PROJECT_ROOT.name})", truncated)
