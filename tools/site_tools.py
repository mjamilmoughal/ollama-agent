"""Bounded website discovery and multi-page reading tools."""

from __future__ import annotations

import re
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

from langchain_core.tools import tool

from .fetch_url import _rejection_reason

USER_AGENT = "OllamaLocalAssistant/1.0"
REQUEST_TIMEOUT_SECONDS = 12
MAX_DOWNLOAD_BYTES = 2_000_000
MAX_DISCOVERED_PAGES = 100
MAX_VISITED_PAGES = 500
MAX_CRAWL_DEPTH = 3
MAX_SITEMAPS = 10
MAX_BATCH_PAGES = 20
MAX_CHARS_PER_PAGE = 4_000
MAX_TOTAL_OUTPUT_CHARS = 20_000


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data.strip())

    @property
    def title(self) -> str:
        return " ".join(part for part in self.title_parts if part).strip()


def _normalize_url(url: str) -> str:
    clean, _fragment = urldefrag(url.strip())
    parsed = urlparse(clean)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", parsed.query, ""))


def _same_site(url: str, hostname: str) -> bool:
    return (urlparse(url).hostname or "").lower() == hostname.lower()


def _matches_filters(url: str, include_pattern: str, exclude_patterns: list[str]) -> bool:
    lowered = url.lower()
    if include_pattern and include_pattern.lower() not in lowered:
        return False
    return not any(pattern.lower() in lowered for pattern in exclude_patterns if pattern)


def _download(url: str, hostname: str) -> tuple[str, str, str] | tuple[None, None, str]:
    reason = _rejection_reason(url)
    if reason:
        return None, None, reason
    try:
        request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html, application/xml, text/xml"})
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            final_url = _normalize_url(response.geturl())
            if not _same_site(final_url, hostname):
                return None, None, "redirected outside the requested site"
            content_type = response.headers.get_content_type()
            payload = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(payload) > MAX_DOWNLOAD_BYTES:
                return None, None, f"response exceeds {MAX_DOWNLOAD_BYTES} bytes"
            charset = response.headers.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace"), content_type, final_url
    except Exception as exc:
        return None, None, str(exc)


def _robots(base_url: str, hostname: str) -> tuple[RobotFileParser, list[str]]:
    robots_url = urljoin(base_url, "/robots.txt")
    text, _content_type, _final_url = _download(robots_url, hostname)
    parser = RobotFileParser()
    parser.set_url(robots_url)
    sitemap_urls: list[str] = []
    if text:
        lines = text.splitlines()
        parser.parse(lines)
        sitemap_urls = [line.split(":", 1)[1].strip() for line in lines if line.lower().startswith("sitemap:")]
    else:
        parser.parse([])
    return parser, sitemap_urls


def _sitemap_pages(sitemap_urls: list[str], hostname: str, robots: RobotFileParser) -> list[str]:
    pages: list[str] = []
    pending = deque(sitemap_urls)
    seen: set[str] = set()
    while pending and len(seen) < MAX_SITEMAPS and len(pages) < MAX_DISCOVERED_PAGES:
        sitemap_url = _normalize_url(pending.popleft())
        if sitemap_url in seen or not _same_site(sitemap_url, hostname):
            continue
        seen.add(sitemap_url)
        text, _content_type, _final_url = _download(sitemap_url, hostname)
        if not text:
            continue
        try:
            root = ElementTree.fromstring(text)
        except ElementTree.ParseError:
            continue
        locations = [element.text.strip() for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "loc" and element.text]
        if root.tag.rsplit("}", 1)[-1] == "sitemapindex":
            pending.extend(locations)
        else:
            pages.extend(url for url in locations if _same_site(url, hostname) and robots.can_fetch(USER_AGENT, url))
    return pages


def _discover(
    base_url: str,
    include_pattern: str,
    exclude_patterns: list[str],
    max_pages: int,
    max_depth: int,
) -> tuple[list[dict[str, object]], list[str]]:
    base_url = _normalize_url(base_url)
    parsed = urlparse(base_url)
    hostname = parsed.hostname or ""
    robots, declared_sitemaps = _robots(base_url, hostname)
    sitemap_candidates = declared_sitemaps or [urljoin(base_url, "/sitemap.xml")]
    sitemap_pages = _sitemap_pages(sitemap_candidates, hostname, robots)

    queue = deque([(base_url, 0, "entry"), *((_normalize_url(url), 0, "sitemap") for url in sitemap_pages)])
    seen: set[str] = set()
    pages: list[dict[str, object]] = []
    errors: list[str] = []
    while queue and len(pages) < max_pages and len(seen) < MAX_VISITED_PAGES:
        url, depth, source = queue.popleft()
        url = _normalize_url(url)
        if url in seen or not _same_site(url, hostname):
            continue
        seen.add(url)
        if any(pattern.lower() in url.lower() for pattern in exclude_patterns if pattern):
            continue
        if not robots.can_fetch(USER_AGENT, url):
            continue
        html, content_type, final_url = _download(url, hostname)
        if not html:
            errors.append(f"{url}: {final_url}")
            continue
        if "html" not in (content_type or ""):
            continue
        link_parser = _LinkParser()
        try:
            link_parser.feed(html)
        except Exception:
            pass
        if _matches_filters(final_url, include_pattern, []):
            pages.append({"url": final_url, "title": link_parser.title, "depth": depth, "source": source})
        if depth >= max_depth:
            continue
        for href in link_parser.links:
            absolute = _normalize_url(urljoin(final_url, href))
            if _same_site(absolute, hostname):
                queue.append((absolute, depth + 1, "link"))
    return pages, errors


@tool
def discover_site_pages(
    base_url: str,
    exclude_patterns: list[str] | None = None,
    include_pattern: str = "",
    max_pages: int = 30,
    max_depth: int = 2,
) -> str:
    """Discover readable pages on any public website without assuming its structure.

    Uses robots.txt, declared or conventional sitemaps, and same-site HTML links.
    Use for requests involving an entire site, all pages, navigation, or finding
    which page may contain information. `include_pattern` optionally requires a
    URL substring; `exclude_patterns` skips URL substrings such as `/posts`.
    Results are bounded and report limits, so do not claim completeness when the
    returned inventory says it was capped or pages failed.
    """
    reason = _rejection_reason(base_url)
    if reason:
        return f"Could not discover '{base_url}': {reason}."
    max_pages = max(1, min(max_pages, MAX_DISCOVERED_PAGES))
    max_depth = max(0, min(max_depth, MAX_CRAWL_DEPTH))
    pages, errors = _discover(base_url, include_pattern, exclude_patterns or [], max_pages, max_depth)
    lines = [f"Discovered {len(pages)} page(s) from {base_url} (limit={max_pages}, depth={max_depth}):"]
    lines.extend(f"- {page['title'] or '(untitled)'} | {page['url']} | source={page['source']} depth={page['depth']}" for page in pages)
    if len(pages) >= max_pages:
        lines.append("Inventory reached max_pages and may be incomplete.")
    if errors:
        lines.append(f"Failed pages: {len(errors)} (first 5):")
        lines.extend(f"- {error}" for error in errors[:5])
    return "\n".join(lines)[:MAX_TOTAL_OUTPUT_CHARS]


def _focused_excerpt(text: str, query: str, limit: int) -> str:
    if not query:
        return text[:limit]
    terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_]{3,}", query)}
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    ranked = sorted(paragraphs, key=lambda part: sum(part.lower().count(term) for term in terms), reverse=True)
    selected: list[str] = []
    used = 0
    for paragraph in ranked:
        if used + len(paragraph) > limit:
            paragraph = paragraph[: max(0, limit - used)]
        if paragraph:
            selected.append(paragraph)
            used += len(paragraph)
        if used >= limit:
            break
    return "\n\n".join(selected)


@tool
def fetch_urls(urls: list[str], query: str = "", max_chars_per_page: int = 3000) -> str:
    """Read a bounded batch of public page URLs and return their main text.

    Pass exact URLs obtained from search_online or discover_site_pages. Use
    `query` to prioritize passages relevant to the desired information and keep
    context small; leave it empty when broad page understanding is needed.
    Each page remains separately attributed. The tool accepts at most 20 URLs,
    does not discover additional pages, and reports failures instead of guessing.
    """
    import trafilatura

    urls = urls[:MAX_BATCH_PAGES]
    max_chars_per_page = max(500, min(max_chars_per_page, MAX_CHARS_PER_PAGE))
    sections: list[str] = []
    for url in urls:
        reason = _rejection_reason(url)
        if reason:
            sections.append(f"URL: {url}\nError: {reason}")
            continue
        hostname = urlparse(url).hostname or ""
        html, content_type, detail = _download(_normalize_url(url), hostname)
        if not html:
            sections.append(f"URL: {url}\nError: {detail}")
            continue
        if "html" not in (content_type or ""):
            sections.append(f"URL: {url}\nError: unsupported content type {content_type}")
            continue
        text = trafilatura.extract(html, include_comments=False)
        if not text or not text.strip():
            sections.append(f"URL: {detail}\nError: no readable main text extracted")
            continue
        excerpt = _focused_excerpt(text.strip(), query, max_chars_per_page)
        sections.append(f"URL: {detail}\nContent:\n{excerpt}")
    result = f"Fetched {len(urls)} URL(s); query={query or '(none)'}:\n\n" + "\n\n---\n\n".join(sections)
    return result[:MAX_TOTAL_OUTPUT_CHARS]