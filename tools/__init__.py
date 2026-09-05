"""Tools available to the agent."""

from .analyze_json import get_json_value, highest_nested_json_count, inspect_json, rank_json_records
from .fetch_url import fetch_url
from .find_file import find_file
from .find_project_file import find_project_file
from .git_tools import git_blame, git_diff, git_log
from .list_directory import list_directory
from .read_file import read_file
from .search_in_file import search_in_file
from .search_online import search_online
from .site_tools import discover_site_pages, fetch_urls
from .view_image import view_image

ALL_TOOLS = [
    search_online,
    fetch_url,
    find_project_file,
    find_file,
    list_directory,
    read_file,
    search_in_file,
    inspect_json,
    get_json_value,
    rank_json_records,
    highest_nested_json_count,
    discover_site_pages,
    fetch_urls,
    git_log,
    git_diff,
    git_blame,
]

# Kept out of ALL_TOOLS: only bound to models whose Ollama capabilities include
# "vision" (see model_supports_vision in app.py). Binding an image-returning tool
# to a model that can't process images risks Ollama rejecting or mishandling
# image data it has no way to use.
VISION_TOOLS = [view_image]
