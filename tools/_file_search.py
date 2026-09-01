"""Shared fuzzy filename-matching engine used by find_file and find_project_file."""

from __future__ import annotations

import os
import time
from difflib import SequenceMatcher

MAX_RESULTS = 20
MAX_SCAN_SECONDS = 60
MAX_ENTRIES_SCANNED = 400_000
FUZZY_THRESHOLD = 0.72


def match_tier(query: str, candidate: str) -> int | None:
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


def search_tree(root: str, name: str, kind: str, excluded_dir_names: set[str]) -> tuple[list[tuple[int, str, str]], bool]:
    """Walk `root` for files/folders matching `name`. Returns (sorted matches, truncated)."""
    matches: list[tuple[int, str, str]] = []
    scanned = 0
    start = time.monotonic()
    truncated = False

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d.lower() not in excluded_dir_names]

        if time.monotonic() - start > MAX_SCAN_SECONDS or scanned > MAX_ENTRIES_SCANNED:
            truncated = True
            break

        if kind in ("folder", "any"):
            for d in dirnames:
                scanned += 1
                tier = match_tier(name, d)
                if tier is not None:
                    matches.append((tier, os.path.join(dirpath, d), "folder"))

        if kind in ("file", "any"):
            for f in filenames:
                scanned += 1
                tier = match_tier(name, f)
                if tier is not None:
                    matches.append((tier, os.path.join(dirpath, f), "file"))

    matches.sort(key=lambda m: (m[0], len(m[1]), m[1].lower()))
    return matches, truncated


def format_results(matches: list[tuple[int, str, str]], name: str, kind: str, location_label: str, truncated: bool) -> str:
    if not matches:
        note = " (search was stopped early due to size/time limits)" if truncated else ""
        target = kind if kind != "any" else "file or folder"
        return f"No {target} matching '{name}' found in {location_label}:{note}."

    top = matches[:MAX_RESULTS]
    lines = [f"Found {len(matches)} match(es) for '{name}' in {location_label} (showing top {len(top)}):"]
    for _, path, kind_label in top:
        lines.append(f"- [{kind_label}] {path}")
    if truncated:
        lines.append("Note: the search was stopped early (time/size limit) -- results may be incomplete.")
    return "\n".join(lines)
