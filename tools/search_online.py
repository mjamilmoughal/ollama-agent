"""A web search tool the agent can call when it needs current, real-world information."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def search_online(query: str) -> str:
    """Search the public web for current, real-time, or up-to-date information.

    Use this whenever the user asks you to search online, look something up,
    or check the web, or whenever answering well requires information that
    could have changed since your training data (news, prices, releases,
    weather, current events, live facts). Returns the top results as short
    title/snippet/URL entries.
    """
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        results = list(DDGS().text(query, max_results=5))
    except DDGSException as exc:
        return f"Search failed for '{query}': {exc}"
    except Exception as exc:
        return f"Search failed for '{query}': {exc}"

    if not results:
        return f"No results found for '{query}'."

    lines = [f"Search results for '{query}':"]
    for i, result in enumerate(results, start=1):
        title = result.get("title", "").strip()
        body = result.get("body", "").strip()
        href = result.get("href", "").strip()
        lines.append(f"{i}. {title}\n   {body}\n   {href}")
    lines.append("Tip: call fetch_url on one of these URLs if you need more detail or want to verify a claim.")
    return "\n".join(lines)
