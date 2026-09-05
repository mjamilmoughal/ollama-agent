"""Shared project-root resolution for the project-scoped tools."""

from __future__ import annotations

from pathlib import Path

# The directory the CLI was launched from -- the "current project" that
# find_project_file searches and read_file resolves relative paths against.
PROJECT_ROOT = Path.cwd().resolve()
