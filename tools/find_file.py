"""A local file/folder search tool the agent can call to locate something on disk."""

from __future__ import annotations

import os
import string
import time
from difflib import SequenceMatcher

from langchain_core.tools import tool

# Directories that are either huge, irrelevant, or unsafe to walk into.
EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", "node_modules", "__pycache__", "venv", ".venv", "env",
    "$recycle.bin", "system volume information", "windows", "programdata",
    "$windows.~bt", "$windows.~ws", "appdata",
}
MAX_RESULTS = 20
MAX_SCAN_SECONDS = 60
MAX_ENTRIES_SCANNED = 400_000
FUZZY_THRESHOLD = 0.72


def _available_drives() -> list[str]:
    return [d for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]


def _match_tier(query: str, candidate: str) -> int | None:
    """Score how well candidate matches query. Lower tier = better match. None = no match."""
    q = query.lower().strip()
    c = candidate.lower()
    stem, sep, ext = c.rpartition(".")
    c_stem = stem if sep and ext else c

    if q == c:
        return 0
    if q == c_stem:
        return 1
    if q in c:
        return 2
    if SequenceMatcher(None, q, c_stem or c).ratio() >= FUZZY_THRESHOLD:
        return 3
    return None


@tool
def find_file(name: str, drive: str, kind: str = "any") -> str:
    """Search an entire drive for a file or folder by name.

    You MUST ask the user which drive to search (e.g. "C" or "D") before
    calling this tool, unless they already told you in their message. Never
    guess or default to a drive on your own -- the C: drive especially is
    huge and mostly irrelevant OS files, so only search it if the user
    explicitly says so. If you call this without a valid drive, it fails and
    tells you which drives exist so you can ask the user.

    Args:
        name: the file or folder name to look for. Matching is fuzzy and
              case-insensitive, so a partial name or a close guess is fine.
              Include the extension if the user gave one (e.g. "report.pdf").
        drive: a single drive letter the user confirmed, e.g. "C" or "D".
        kind: "file" to only match files, "folder" to only match folders, or
              "any" (default) to match either.
    """
    name = (name or "").strip()
    if not name:
        return "Missing a file/folder name to search for."

    drives = _available_drives()
    drive = (drive or "").strip().rstrip(":\\/").upper()
    if len(drive) != 1 or drive not in drives:
        return (
            f"Invalid or missing drive '{drive}'. Ask the user which drive to search. "
            f"Available drives: {', '.join(d + ':' for d in drives)}."
        )

    kind = (kind or "any").strip().lower()
    if kind not in ("file", "folder", "any"):
        kind = "any"

    root = f"{drive}:\\"
    matches: list[tuple[int, str, str]] = []
    scanned = 0
    start = time.monotonic()
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDED_DIR_NAMES]

        if time.monotonic() - start > MAX_SCAN_SECONDS or scanned > MAX_ENTRIES_SCANNED:
            truncated = True
            break

        if kind in ("folder", "any"):
            for d in dirnames:
                scanned += 1
                tier = _match_tier(name, d)
                if tier is not None:
                    matches.append((tier, os.path.join(dirpath, d), "folder"))

        if kind in ("file", "any"):
            for f in filenames:
                scanned += 1
                tier = _match_tier(name, f)
                if tier is not None:
                    matches.append((tier, os.path.join(dirpath, f), "file"))

    if not matches:
        note = " (search was stopped early due to size/time limits)" if truncated else ""
        target = kind if kind != "any" else "file or folder"
        return f"No {target} matching '{name}' found on {drive}:{note}."

    matches.sort(key=lambda m: (m[0], len(m[1]), m[1].lower()))
    top = matches[:MAX_RESULTS]

    lines = [f"Found {len(matches)} match(es) for '{name}' on {drive}: (showing top {len(top)}):"]
    for _, path, kind_label in top:
        lines.append(f"- [{kind_label}] {path}")
    if truncated:
        lines.append("Note: the search was stopped early (time/size limit) -- results may be incomplete.")
    return "\n".join(lines)
