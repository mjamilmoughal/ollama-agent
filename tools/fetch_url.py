"""A page-reading tool the agent can call to go deeper than a search snippet."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from langchain_core.tools import tool

MAX_CHARS = 3000


def _rejection_reason(url: str) -> str | None:
    if "://" not in url:
        return (
            "that is not a real URL. Pass the exact http(s) URL string from a previous "
            "search_online result (the line starting with 'https://'), not a placeholder or description"
        )
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"unsupported URL scheme '{parsed.scheme}' (only http/https allowed)"
    hostname = parsed.hostname or ""
    if not hostname:
        return "no hostname in URL"
    if hostname.lower() == "localhost":
        return "refusing to fetch localhost"
    try:
        resolved_ips = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except OSError as exc:
        return f"could not resolve host: {exc}"
    for ip in resolved_ips:
        addr = ipaddress.ip_address(ip)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            return f"refusing to fetch internal/private address ({ip})"
    return None


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return its main readable text content.

    Use this after search_online when the search snippets aren't detailed
    or credible enough to answer confidently, when you need to verify a
    claim against the original source, or when the user gives you a
    specific URL to read. Pass a real URL, ideally one returned by
    search_online. Returns the page's main article text (boilerplate like
    navigation/ads stripped), trimmed to a reasonable length.
    """
    reason = _rejection_reason(url)
    if reason:
        return f"Could not fetch '{url}': {reason}."

    import trafilatura

    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        return f"Could not fetch '{url}': {exc}"

    if not downloaded:
        return f"Could not fetch '{url}': no content returned (dead link, blocked, or timed out)."

    text = trafilatura.extract(downloaded, include_comments=False)
    if not text or not text.strip():
        return f"Fetched '{url}' but could not extract readable article text from it."

    text = text.strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "... [truncated]"
    return f"Content from {url}:\n{text}"
