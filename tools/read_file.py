"""A local file-reading tool the agent can call to inspect a file's full content."""

from __future__ import annotations

import os

from langchain_core.tools import tool

MAX_LINES_PER_CALL = 300
MAX_CHARS_PER_CALL = 12_000
MAX_FULL_SCAN_BYTES = 20_000_000  # above this, skip counting exact total lines
BINARY_SNIFF_BYTES = 8192
_TEXT_BYTES = bytes({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7F})


def _looks_binary(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    nontext = sum(b not in _TEXT_BYTES for b in chunk)
    return nontext / len(chunk) > 0.30


@tool
def read_file(path: str, offset: int = 1, limit: int = 300) -> str:
    """Read a local text/code file so you can understand its full content.

    Pass the exact file path (e.g. one returned by find_file, or one the user
    gave you directly). Returns the file's lines, numbered, starting at
    `offset` (1-based) for up to `limit` lines. If the file is longer than
    what's returned, a note at the end tells you the next offset to use --
    call this again with that offset to keep reading until you've seen the
    whole file before answering questions about its content. Only works on
    text/code files (source code, config, markdown, logs, csv, json, etc.);
    binary files (images, executables, archives, PDFs) are refused.
    """
    path = (path or "").strip().strip('"')
    if not path:
        return "Missing a file path to read."
    if not os.path.isabs(path):
        return f"'{path}' is not an absolute path. Ask the user for the full path, or use find_file first."
    if not os.path.exists(path):
        return f"'{path}' does not exist. Use find_file if you're not sure of the exact path."
    if os.path.isdir(path):
        return f"'{path}' is a directory, not a file."
    if _looks_binary(path):
        return f"'{path}' looks like a binary file, not text -- cannot read its content as text."

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"Could not stat '{path}': {exc}"

    offset = max(1, offset)
    limit = max(1, min(limit, MAX_LINES_PER_CALL))
    count_exact_total = size <= MAX_FULL_SCAN_BYTES

    lines: list[str] = []
    total_lines = 0
    chars_used = 0
    truncated_by_chars = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, raw_line in enumerate(f, start=1):
                if i < offset:
                    if count_exact_total:
                        total_lines = i
                    continue
                if len(lines) >= limit:
                    if not count_exact_total:
                        break
                    total_lines = i
                    continue
                if chars_used >= MAX_CHARS_PER_CALL:
                    truncated_by_chars = True
                    if not count_exact_total:
                        break
                    total_lines = i
                    continue
                line = raw_line.rstrip("\n")
                if chars_used + len(line) > MAX_CHARS_PER_CALL:
                    line = line[: MAX_CHARS_PER_CALL - chars_used] + "…[line truncated]"
                    truncated_by_chars = True
                chars_used += len(line)
                lines.append(f"{i:>6}\t{line}")
                total_lines = i
    except OSError as exc:
        return f"Could not read '{path}': {exc}"

    if not lines:
        if count_exact_total and offset > total_lines:
            return f"'{path}' has {total_lines} line(s) total -- offset {offset} is past the end of the file."
        return f"No content read from '{path}' at offset {offset} (may be past the end of the file)."

    header = f"{path} ({size} bytes" + (f", {total_lines} lines total" if count_exact_total else "") + "):"
    body = "\n".join(lines)
    next_offset = offset + len(lines)

    footer_parts = []
    if count_exact_total:
        if next_offset <= total_lines:
            footer_parts.append(f"Showing lines {offset}-{next_offset - 1} of {total_lines}. Call again with offset={next_offset} to keep reading.")
        else:
            footer_parts.append(f"Showing lines {offset}-{next_offset - 1} of {total_lines} (end of file).")
    else:
        footer_parts.append(
            f"Large file -- showing lines {offset}-{next_offset - 1} (total not counted). "
            f"Call again with offset={next_offset}; an empty/short result means you've reached the end."
        )
    if truncated_by_chars:
        footer_parts.append("Note: cut short by per-call size limit; re-call with a higher offset or smaller limit if needed.")

    return "\n".join([header, body, "", " ".join(footer_parts)])
