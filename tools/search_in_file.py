"""Bounded text search within one local file."""

from __future__ import annotations

import os
import re

from langchain_core.tools import tool

from ._project import PROJECT_ROOT

MAX_FILE_BYTES = 20_000_000
MAX_RESULTS = 50
MAX_OUTPUT_CHARS = 8_000


@tool
def search_in_file(path: str, query: str, use_regex: bool = False, max_results: int = 20) -> str:
    """Search inside a text file and return matching line numbers and excerpts.

    Use this to locate a key, phrase, symbol, or section in a large file without
    reading every page. `path` may be absolute or relative to the current
    project. Matching is case-insensitive. Set `use_regex` only when a regular
    expression is needed. After finding a relevant line, use read_file with a
    nearby offset for surrounding content, or analyze_json for JSON queries.
    """
    path = (path or "").strip().strip('"')
    query = (query or "").strip()
    if not path or not query:
        return "Missing a file path or search query."
    if not os.path.isabs(path):
        path = str((PROJECT_ROOT / path).resolve())
    if not os.path.isfile(path):
        return f"'{path}' is not a file. Use find_project_file or find_file first."
    if os.path.getsize(path) > MAX_FILE_BYTES:
        return f"'{path}' is too large to search safely (limit: {MAX_FILE_BYTES} bytes)."

    try:
        pattern = re.compile(query if use_regex else re.escape(query), re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regular expression '{query}': {exc}"

    max_results = max(1, min(max_results, MAX_RESULTS))
    matches: list[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as file:
            for line_number, line in enumerate(file, start=1):
                if not pattern.search(line):
                    continue
                excerpt = line.strip()
                if len(excerpt) > 300:
                    excerpt = excerpt[:297] + "..."
                matches.append(f"{line_number}: {excerpt}")
                if len(matches) >= max_results:
                    break
    except OSError as exc:
        return f"Could not search '{path}': {exc}"

    if not matches:
        return f"No matches for '{query}' in '{path}'."
    result = f"Matches for '{query}' in '{path}':\n" + "\n".join(matches)
    return result[:MAX_OUTPUT_CHARS]