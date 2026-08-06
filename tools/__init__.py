"""Tools available to the agent."""

from .fetch_url import fetch_url
from .search_online import search_online

ALL_TOOLS = [search_online, fetch_url]
