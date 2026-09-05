"""Read-only structured queries for JSON files without loading them into model context."""

from __future__ import annotations

import json
import os
from collections import Counter, deque
from typing import Any

from langchain_core.tools import tool

from ._project import PROJECT_ROOT

MAX_FILE_BYTES = 50_000_000
MAX_OUTPUT_CHARS = 8_000
MAX_SCHEMA_DEPTH = 5
MAX_SCHEMA_PATHS = 300


def _resolve_file(path: str) -> str | None:
    path = (path or "").strip().strip('"')
    if not path:
        return None
    if not os.path.isabs(path):
        path = str((PROJECT_ROOT / path).resolve())
    return path


def _select(data: Any, path: str) -> Any:
    value = data
    for part in (segment for segment in path.split(".") if segment):
        if isinstance(value, dict):
            value = value[part]
        elif isinstance(value, list) and part.isdigit():
            value = value[int(part)]
        else:
            raise KeyError(part)
    return value


def _compact(value: Any) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False)
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + "\n... [output truncated]"
    return text


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _describe_schema(value: Any) -> str:
    """Describe representative JSON paths and types without returning the dataset."""
    lines: list[str] = []
    pending = deque([(value, "", 0)])
    while pending and len(lines) < MAX_SCHEMA_PATHS:
        current, path, depth = pending.popleft()
        description = _type_name(current)
        if isinstance(current, list):
            item_types = ", ".join(sorted({_type_name(item) for item in current[:100]})) or "empty"
            description += f"[{len(current)}], items: {item_types}"
        lines.append(f"{path or '<root>'}: {description}")
        if depth >= MAX_SCHEMA_DEPTH:
            continue
        if isinstance(current, dict):
            pending.extend(
                (child, f"{path}.{key}" if path else key, depth + 1)
                for key, child in current.items()
            )
        elif isinstance(current, list) and current:
            pending.append((current[0], f"{path}[]" if path else "[]", depth + 1))
    if pending:
        lines.append(f"... [{len(pending)} queued paths omitted; inspect a specific path for more detail]")
    return "\n".join(lines)


def _invoke_analysis(**arguments: Any) -> str:
    return str(analyze_json.invoke(arguments))


@tool
def analyze_json(
    path: str,
    operation: str,
    collection: str = "",
    field: str = "",
    group_by: str = "",
    nested_collection: str = "",
    limit: int = 10,
) -> str:
    """Query or aggregate a JSON file deterministically.

    Use this instead of reading a large JSON file page by page when the user
    asks for a specific value, ranking, count, maximum, or aggregation.

    Operations:
      - inspect: describe the value at `collection` (empty means the root).
    - get: return the value at `collection`, such as "metadata.version".
      - top: rank object records in `collection` by numeric `field`, returning
        all records tied for the maximum plus up to `limit` ranked records.
      - group_count: count records in `collection` by `group_by`.
      - nested_group_count: within every parent record in `collection`, count
        nested records from `nested_collection` by `group_by`, then return all
        ties for the highest count with their parent record included. Example:
        collection="orders", nested_collection="items", group_by="sku".

    Dot-separated paths select nested keys; numeric segments index arrays.
    This tool only reads JSON and never executes code or modifies the file.
    """
    resolved = _resolve_file(path)
    if not resolved:
        return "Missing a JSON file path to analyze."
    if not os.path.isfile(resolved):
        return f"'{path}' is not a file. Use find_project_file or find_file first."
    if os.path.getsize(resolved) > MAX_FILE_BYTES:
        return f"'{path}' is too large to analyze safely (limit: {MAX_FILE_BYTES} bytes)."

    try:
        with open(resolved, "r", encoding="utf-8") as file:
            data = json.load(file)
        selected = _select(data, collection) if collection else data
    except (OSError, json.JSONDecodeError) as exc:
        return f"Could not parse '{path}' as JSON: {exc}"
    except (KeyError, IndexError) as exc:
        return f"JSON path '{collection}' was not found in '{path}': {exc}"

    operation = (operation or "").strip().lower()
    limit = max(1, min(limit, 100))

    if operation == "get":
        return f"JSON value at '{collection or '<root>'}':\n{_compact(selected)}"

    if operation == "inspect":
        summary = _describe_schema(selected)
        return f"JSON structure at '{collection or '<root>'}':\n{summary[:MAX_OUTPUT_CHARS]}"

    if not isinstance(selected, list):
        return f"Operation '{operation}' requires '{collection or '<root>'}' to be an array."

    if operation == "top":
        if not field:
            return "The top operation requires a numeric field."
        records = [record for record in selected if isinstance(record, dict) and isinstance(record.get(field), (int, float))]
        if not records:
            return f"No records with numeric field '{field}' found at '{collection}'."
        ranked = sorted(records, key=lambda record: record[field], reverse=True)
        maximum = ranked[0][field]
        result = {
            "maximum": maximum,
            "ties": [record for record in ranked if record[field] == maximum],
            "ranked_records": ranked[:limit],
        }
        return f"Top records from '{collection}' by '{field}':\n{_compact(result)}"

    if operation == "group_count":
        if not group_by:
            return "The group_count operation requires group_by."
        counts = Counter(record.get(group_by) for record in selected if isinstance(record, dict) and group_by in record)
        result = [{group_by: key, "count": count} for key, count in counts.most_common(limit)]
        return f"Counts from '{collection}' grouped by '{group_by}':\n{_compact(result)}"

    if operation == "nested_group_count":
        if not nested_collection or not group_by:
            return "The nested_group_count operation requires nested_collection and group_by."
        maximum = 0
        ties: list[dict[str, Any]] = []
        for parent in selected:
            if not isinstance(parent, dict) or not isinstance(parent.get(nested_collection), list):
                continue
            counts = Counter(
                record.get(group_by)
                for record in parent[nested_collection]
                if isinstance(record, dict) and group_by in record
            )
            for key, count in counts.items():
                parent_summary = {name: value for name, value in parent.items() if name != nested_collection}
                result = {group_by: key, "count": count, "parent": parent_summary}
                if count > maximum:
                    maximum, ties = count, [result]
                elif count == maximum:
                    ties.append(result)
        result = {
            "maximum": maximum,
            "ties": ties,
        }
        return f"Highest nested count in '{collection}.{nested_collection}' by '{group_by}':\n{_compact(result)}"

    return "Unknown operation. Use inspect, get, top, group_count, or nested_group_count."


@tool
def inspect_json(file_path: str, json_path: str = "") -> str:
    """Inspect a JSON file's structure before querying an unfamiliar dataset.

    `file_path` is the local .json filename. `json_path` is an optional
    dot-separated location inside it. Recursively returns field names, value
    types, array lengths, item structures, and small scalar examples without
    returning the full dataset. Use discovered paths and fields exactly in
    subsequent calls. Works with any valid JSON object, array, or scalar.
    """
    return _invoke_analysis(path=file_path, operation="inspect", collection=json_path)


@tool
def get_json_value(file_path: str, json_path: str) -> str:
    """Get a specific value or section from a JSON file.

    `file_path` is the local .json filename. `json_path` is a dot-separated
    key path such as "metadata.version". Use inspect_json first if the exact
    keys are unknown.
    """
    return _invoke_analysis(path=file_path, operation="get", collection=json_path)


@tool
def rank_json_records(file_path: str, array_path: str, numeric_field: str, limit: int = 10) -> str:
    """Rank records in a JSON array by a numeric field and return all top ties.

    `numeric_field` must be a scalar number in each record. Use this for any
    maximum or ranking over existing numeric values. Do not use it to count
    items in nested arrays; use highest_nested_json_count for that. Discover
    the exact array path and numeric field with inspect_json first.
    """
    return _invoke_analysis(
        path=file_path,
        operation="top",
        collection=array_path,
        field=numeric_field,
        limit=limit,
    )


@tool
def highest_nested_json_count(
    file_path: str,
    parent_array_path: str,
    nested_array_field: str,
    group_by_field: str,
) -> str:
    """Find who or what occurs most often inside any one parent JSON record.

    For every object in `parent_array_path`, count repeated values of
    `group_by_field` inside its `nested_array_field`. Returns the largest count
    and every tied value with its parent object. This is generic across event
    logs, orders and line items, sessions and actions, or similar nested data.
    Discover all three exact field names with inspect_json first.
    """
    return _invoke_analysis(
        path=file_path,
        operation="nested_group_count",
        collection=parent_array_path,
        nested_collection=nested_array_field,
        group_by=group_by_field,
    )