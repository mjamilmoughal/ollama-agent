"""A local file/folder search tool for locating something outside the current project."""

from __future__ import annotations

import os
import platform
import string
from pathlib import Path

from langchain_core.tools import tool

from ._file_search import format_results, search_tree

IS_WINDOWS = platform.system() == "Windows"

# Directories that are either huge, irrelevant, or unsafe to walk into.
EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", "venv", ".venv", "env",
    "$recycle.bin", "system volume information", "windows", "programdata",
    "$windows.~bt", "$windows.~ws", "appdata", "proc", "sys", "dev",
}


def _available_locations() -> dict[str, str]:
    """Named roots the user can pick without typing a raw path: drive letters on
    Windows, the home directory everywhere else."""
    if IS_WINDOWS:
        return {d: f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")}
    return {"home": str(Path.home())}


def _resolve_location(location: str) -> tuple[str, str] | None:
    """Resolve a user-confirmed location to (root_path, label). Accepts a named
    location (e.g. "C" or "home") or an existing absolute folder path. None if invalid."""
    location = (location or "").strip()
    if not location:
        return None

    named = _available_locations()
    key = location.rstrip(":\\/ ").upper() if IS_WINDOWS else location.lower()
    if key in named:
        label = f"{key}:" if IS_WINDOWS else key
        return named[key], label

    candidate = os.path.expanduser(location)
    if os.path.isabs(candidate) and os.path.isdir(candidate):
        return candidate, candidate

    return None


@tool
def find_file(name: str, location: str, kind: str = "any") -> str:
    """Search a broader area of the user's computer (outside the current project) for a file or folder by name.

    You MUST ask the user which location to search before calling this,
    unless they already said -- never guess or default to a huge root like a
    whole drive or filesystem. Good choices to offer: "home" (their home
    directory), a drive letter like "C" or "D" if they're on Windows, or a
    specific folder path they name. For files that are part of the current
    project, use find_project_file instead -- it needs no confirmation. If
    you call this with an invalid or missing location, it fails and tells you
    the valid named locations so you can ask the user.

    Args:
        name: the file or folder name to look for. Matching is fuzzy and
              case-insensitive, so a partial name or a close guess is fine.
              Include the extension if the user gave one (e.g. "report.pdf").
        location: a location the user confirmed: "home", a drive letter such
              as "C" (Windows only), or an absolute folder path.
        kind: "file" to only match files, "folder" to only match folders, or
              "any" (default) to match either.
    """
    name = (name or "").strip()
    if not name:
        return "Missing a file/folder name to search for."

    resolved = _resolve_location(location)
    if resolved is None:
        options = ", ".join(_available_locations().keys())
        return (
            f"Invalid or missing location '{location}'. Ask the user which location to "
            f"search. Available named locations: {options} (or give a specific folder path)."
        )
    root, label = resolved

    kind = (kind or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    matches, truncated = search_tree(root, name, kind, EXCLUDED_DIR_NAMES)
    return format_results(matches, name, kind, label, truncated)
