"""A beautiful terminal chat agent for local Ollama models, built with LangGraph."""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator

import ollama as ollama_client
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.segment import Segment, Segments
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from tools import ALL_TOOLS

console = Console()
THREAD_ID = "session"
HISTORY_FILE = Path.home() / ".ollama_chat_history"
SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

# ANSI color names, shared between Rich markup and prompt_toolkit HTML tags,
# so the input line and the panels always agree on a palette.
PT_USER, PT_ACCENT, PT_MUTED = "ansigreen", "ansicyan", "ansibrightblack"

# One shared visual language for the whole session: a single accent color and a small
# set of role glyphs, reused by every print_* helper below instead of ad-hoc styling.
ACCENT = "cyan"
GLYPH_BULLET = "⏺"   # assistant replies and tool calls
GLYPH_RESULT = "⎿"   # a tool call's result, indented under its bullet
GLYPH_OK = "✓"
GLYPH_WARN = "⚠"
GLYPH_ERR = "✗"
GLYPH_THINK = "✻"   # live reasoning trace, and its collapsed "Thought for Xs" summary

# How many of the most recent wrapped reasoning lines stay on screen while a
# thinking-capable model streams its chain of thought -- older lines scroll off
# the top instead of flooding the terminal with the full trace.
THINKING_VISIBLE_LINES = 4


def print_ok(message: str) -> None:
    console.print(f"[green]{GLYPH_OK}[/green] {message}")


def print_warn(message: str) -> None:
    console.print(f"[yellow]{GLYPH_WARN}[/yellow] {message}")


def print_err(message: str) -> None:
    console.print(f"[red]{GLYPH_ERR}[/red] {message}")

SYSTEM_PROMPT = (
    "You have five tools: search_online (quick web search, short snippets), "
    "fetch_url (reads one full page in detail), find_project_file (searches "
    "only the current project directory for a file or folder by name), "
    "find_file (searches a broader location on the user's computer for a "
    "file or folder by name), and read_file (reads a local file's content by "
    "path, paginated by line). Search first for current or factual "
    "questions; if the snippets aren't detailed or credible enough to answer "
    "confidently, fetch one of the returned URLs before answering. When "
    "calling fetch_url, copy the exact 'https://...' URL string from a "
    "search_online result verbatim — never a placeholder or description. Say "
    "which source you used. For a file described as part of 'this project', "
    "'this repo', or 'the current folder', call find_project_file directly — "
    "it never needs confirmation. For anything else, before calling find_file "
    "you must ask the user which location to search (e.g. their home "
    "directory, or a specific folder) unless they already said — never guess "
    "or default to a huge root like a whole drive or filesystem. When asked "
    "about a local file's content, use find_project_file or find_file first "
    "if you don't have its exact path, then read_file (it accepts either an "
    "absolute path or one relative to the current project) — if read_file's "
    "result says there's more to read, keep calling it with the next offset "
    "until you've seen the whole file before answering, so your "
    "understanding is based on its complete content, not just the first chunk."
)

COMMANDS = {
    "/model": "Switch the active Ollama model (name, number, or fuzzy match)",
    "/clear": "Clear conversation history",
    "/task": "Mark the start of a new logical task (groups turns for analysis)",
    "/rate": "Tag the last reply's outcome: /rate correct|incorrect|partial",
    "/failure": "Tag the last reply's failure mode, e.g. /failure wrong_tool",
    "/necessary": "Mark the last reply's tool call(s) as necessary",
    "/unnecessary": "Mark the last reply's tool call(s) as unnecessary",
    "/note": "Attach a freeform note to the last reply",
    "/help": "Show this help message",
    "/exit": "Quit the chat (also /quit)",
}

def _format_call_args(args: dict) -> str:
    """Render tool call args as a compact function-call signature, e.g. query="paris weather"."""
    parts = []
    for key, value in args.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f'{key}="{text}"' if isinstance(value, str) else f"{key}={text}")
    return ", ".join(parts)

# Vocabulary suggested for /failure; freeform text is also accepted.
FAILURE_MODES = [
    "malformed_tool_call",
    "wrong_tool",
    "wrong_arguments",
    "ignored_tool_result",
    "hallucinated",
    "infinite_loop",
    "other",
]

# A turn with this many tool calls (or a run of identical repeated calls) is flagged
# as a possible infinite loop for manual review.
TOOL_CALL_LOOP_THRESHOLD = 8
IDENTICAL_CALL_REPEAT_THRESHOLD = 3

# Heuristic only: every tool in tools/ puts its failure wording at the very start of
# the returned string, so a marker found in the first 200 chars suggests the call
# failed. This flags candidates for manual review -- it is not a correctness signal.
_TOOL_ERROR_MARKERS = (
    "could not", "cannot ", "error", "failed", "no results found", "no content read",
    "missing a", "invalid ", "not found", "is not a", "is not an", "does not exist",
    "unsupported url scheme", "refusing to fetch",
)


def _looks_like_tool_error(text: str | None) -> bool:
    lowered = (text or "").strip().lower()[:200]
    return any(marker in lowered for marker in _TOOL_ERROR_MARKERS)


class SessionState:
    """Tracks the bits of live session info shown in the input toolbar."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.turn_count = 0
        self.task_id = 1
        self.start_time = time.monotonic()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class SessionLogger:
    """Persists every turn of a chat session to sessions/<id>.json for later analysis."""

    def __init__(self, initial_model: str):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        started = datetime.now()
        self.session_id = started.strftime("%Y%m%d_%H%M%S")
        self.path = SESSIONS_DIR / f"{self.session_id}_{sanitize_filename(initial_model)}.json"
        self.data = {
            "session_id": self.session_id,
            "started_at": started.isoformat(timespec="seconds"),
            "ended_at": None,
            "initial_model": initial_model,
            "events": [],
            "turns": [],
        }
        self._write()

    def log_event(self, event: str, **fields) -> None:
        self.data["events"].append({"timestamp": datetime.now().isoformat(timespec="seconds"), "event": event, **fields})
        self._write()

    def log_turn(self, **fields) -> None:
        self.data["turns"].append({"timestamp": datetime.now().isoformat(timespec="seconds"), **fields})
        self._write()

    def annotate_last_turn(self, **fields) -> bool:
        """Apply manual analysis tags (outcome, failure_mode, tool_call_necessary, note) to the most recent turn."""
        if not self.data["turns"]:
            return False
        turn = self.data["turns"][-1]
        turn.setdefault("annotation", {"outcome": None, "failure_mode": None, "tool_call_necessary": None, "note": None})
        turn["annotation"].update(fields)
        self._write()
        return True

    def close(self) -> None:
        self.data["ended_at"] = datetime.now().isoformat(timespec="seconds")
        self._write()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")


class SlashCompleter(Completer):
    """Tab-completes /commands, and model names after '/model '."""

    def __init__(self, models: list[str]):
        self._models = models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if text.startswith("/model "):
            partial = text[len("/model ") :]
            for name in self._models:
                if partial.lower() in name.lower():
                    yield Completion(name, start_position=-len(partial))
            return
        if text.startswith("/rate "):
            partial = text[len("/rate ") :]
            for opt in ("correct", "incorrect", "partial"):
                if opt.startswith(partial.lower()):
                    yield Completion(opt, start_position=-len(partial))
            return
        if text.startswith("/failure "):
            partial = text[len("/failure ") :]
            for opt in FAILURE_MODES:
                if opt.startswith(partial.lower()):
                    yield Completion(opt, start_position=-len(partial))
            return
        for cmd in COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))


def get_installed_models() -> list[str]:
    """Return the names of all locally installed Ollama models."""
    try:
        response = ollama_client.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Ollama. Is the Ollama service running?"
        ) from exc

    raw_models = response.get("models") if isinstance(response, dict) else getattr(response, "models", [])
    names = []
    for m in raw_models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def model_supports_thinking(model_name: str) -> bool:
    """Whether Ollama reports a 'thinking' capability for this model (via /api/show).

    Passing reasoning=True to ChatOllama for a model that lacks this capability makes
    Ollama respond with an HTTP 400, so this must be checked before opting in.
    """
    try:
        info = ollama_client.show(model_name)
    except Exception:
        return False
    caps = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
    return bool(caps) and "thinking" in caps


def unload_model(model_name: str) -> None:
    """Ask Ollama to free this model's memory immediately instead of waiting out its
    keep-alive window.

    Without this, switching models (or exiting) leaves the previous model resident for
    several more minutes. On a memory-constrained machine, two or three large models
    staying loaded at once causes severe swap thrashing -- generation that normally
    takes a few seconds can silently stretch to a minute or more. Best-effort: if
    Ollama is unreachable or the model is already gone, there's nothing useful to do.
    """
    try:
        ollama_client.generate(model=model_name, keep_alive=0)
    except Exception:
        pass


def build_app(model_name: str, checkpointer: MemorySaver):
    """Compile a LangGraph chat graph backed by the given Ollama model, with tool-calling enabled."""
    reasoning = True if model_supports_thinking(model_name) else None
    llm = ChatOllama(model=model_name, reasoning=reasoning).bind_tools(ALL_TOOLS)

    def chat_node(state: MessagesState):
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
        if not response.tool_calls:
            fake_call = parse_fake_tool_call(response.content)
            if fake_call:
                response = response.model_copy(update={
                    "content": "",
                    "tool_calls": [{**fake_call, "id": str(uuid.uuid4()), "type": "tool_call"}],
                })
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_edge(START, "chat")
    graph.add_conditional_edges("chat", tools_condition)
    graph.add_edge("tools", "chat")
    return graph.compile(checkpointer=checkpointer)


def stream_reply(graph_app, user_text: str) -> Iterator[AIMessageChunk]:
    """Yield each raw message chunk of the assistant's reply (content + usage metadata)."""
    config = {"configurable": {"thread_id": THREAD_ID}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    for message_chunk, _metadata in graph_app.stream(inputs, config, stream_mode="messages"):
        yield message_chunk


def print_banner(model_count: int) -> None:
    console.print()
    plural = "s" if model_count != 1 else ""
    console.print(
        Panel(
            f"[bold {ACCENT}]Ollama Agent[/bold {ACCENT}]\n"
            f"[dim]{model_count} local model{plural} · LangGraph tool-calling[/dim]",
            border_style="grey50",
            box=box.ROUNDED,
            padding=(0, 2),
            expand=False,
        )
    )


TOOL_NAMES = {t.name for t in ALL_TOOLS}
FAKE_CALL_PREFIX_RE = re.compile(r'^[\s`]*(?:json)?[\s`]*\{\s*"name"\s*:\s*"([^"]+)"', re.IGNORECASE)


def parse_fake_tool_call(content: str) -> dict | None:
    """Detect a tool call a model printed as plain text instead of a real tool_call, and normalize it."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("name") not in TOOL_NAMES:
        return None
    args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
    if not isinstance(args, dict):
        return None
    args = {k: (v["value"] if isinstance(v, dict) and "value" in v else v) for k, v in args.items()}
    return {"name": data["name"], "args": args}


def match_model(query: str, models: list[str]) -> str | None:
    """Resolve a user-typed query to an installed model: index, exact, or unique prefix."""
    query = query.strip()
    if not query:
        return None
    if query.isdigit() and 1 <= int(query) <= len(models):
        return models[int(query) - 1]
    lower_map = {m.lower(): m for m in models}
    if query.lower() in lower_map:
        return lower_map[query.lower()]
    prefix_matches = [m for m in models if m.lower().startswith(query.lower())]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


def suggest_model(query: str, models: list[str]) -> str | None:
    close = difflib.get_close_matches(query, models, n=1, cutoff=0.4)
    return close[0] if close else None


def choose_model(models: list[str], current: str | None = None) -> str:
    table = Table(box=box.SIMPLE, show_lines=False, header_style=f"bold {ACCENT}", show_edge=False, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Model")
    for i, name in enumerate(models, start=1):
        marker = f" [green]{GLYPH_OK} current[/green]" if name == current else ""
        table.add_row(str(i), f"{name}{marker}")
    console.print(table)

    choice = Prompt.ask(f"[bold {ACCENT}]❯[/bold {ACCENT}] Select a model (number or name)").strip()
    matched = match_model(choice, models)
    if matched:
        return matched

    suggestion = suggest_model(choice, models)
    if suggestion:
        print_warn(f"Unknown model '{choice}'. Did you mean [bold]{suggestion}[/bold]?")
    else:
        print_err(f"Unknown model '{choice}'.")
    return current or models[0]


ANALYSIS_COMMANDS = {"/task", "/rate", "/failure", "/necessary", "/unnecessary", "/note"}


def _commands_table(title: str, commands: dict[str, str]) -> Table:
    table = Table(title=title, title_style=f"bold {ACCENT}", title_justify="left", box=box.SIMPLE, header_style="dim", show_edge=False, pad_edge=False)
    table.add_column("Command", style=f"bold {ACCENT}")
    table.add_column("Description", style="dim")
    for cmd, desc in commands.items():
        table.add_row(cmd, desc)
    return table


def print_help() -> None:
    chat_cmds = {k: v for k, v in COMMANDS.items() if k not in ANALYSIS_COMMANDS}
    analysis_cmds = {k: v for k, v in COMMANDS.items() if k in ANALYSIS_COMMANDS}
    console.print(_commands_table("Chat", chat_cmds))
    console.print()
    console.print(_commands_table("Analysis", analysis_cmds))
    console.print()
    console.print("[dim]↑/↓ recalls history · Tab completes commands and model names · Ctrl+C cancels a reply · Ctrl+D exits.[/dim]")
    console.print(
        "[dim]Just ask the agent to search online — it can search, then read a full page for more "
        "detail on its own, even on models without native tool-calling.[/dim]"
    )


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_prompt_message() -> HTML:
    now = datetime.now().strftime("%H:%M:%S")
    return HTML(f"<{PT_MUTED}>{now}</{PT_MUTED}> <{PT_USER}><b>❯</b></{PT_USER}> ")


def render_toolbar(state: SessionState) -> HTML:
    clock = format_duration(time.monotonic() - state.start_time)
    return HTML(
        f" <{PT_ACCENT}><b>{state.model_name}</b></{PT_ACCENT}>  ·  turn {state.turn_count}  ·  {clock}"
        f"  ·  <{PT_MUTED}>/help · Ctrl+C cancel · Ctrl+D exit</{PT_MUTED}>"
    )


def extract_usage(meta: dict | None) -> dict:
    """Pull input/output token counts and true generation speed out of an Ollama response_metadata dict."""
    meta = meta or {}
    input_tokens = meta.get("prompt_eval_count")
    output_tokens = meta.get("eval_count")
    eval_duration_ns = meta.get("eval_duration")
    tokens_per_second = output_tokens / (eval_duration_ns / 1e9) if output_tokens and eval_duration_ns else None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokens_per_second": tokens_per_second,
    }


def format_status(text: str, elapsed: float, usage: dict | None = None) -> str:
    """`usage` is a dict with input_tokens/output_tokens/tokens_per_second, e.g. from extract_usage() or a turn's totals."""
    usage = usage or {}
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if input_tokens is not None and output_tokens is not None:
        token_part = f"↑{input_tokens} ↓{output_tokens}"
        if usage.get("tokens_per_second") is not None:
            token_part += f" · {usage['tokens_per_second']:.0f} tok/s"
    else:
        token_part = f"~{len(text.split())} tok"
    return f"{elapsed:.1f}s · {token_part}"


def render_answer(model_name: str, text: str) -> Group:
    """Header plus the finished reply rendered as Markdown, printed once via plain
    console.print (not Live) -- so its height can be arbitrarily long with no cursor
    math involved at all; the terminal just scrolls it like any other long output.
    """
    header = Text.from_markup(f"[bold {ACCENT}]{GLYPH_BULLET} Agent ({model_name})[/bold {ACCENT}]")
    if not text.strip():
        return Group(header)
    return Group(header, Markdown(text))


# How many of the most recent wrapped lines of the reply stay visible in the live
# preview while it streams -- capped for the same reason as THINKING_VISIBLE_LINES: a
# Live region whose height can grow past the space physically left below its start row
# forces the terminal to scroll mid-redraw in a way Rich's relative cursor-up math
# doesn't account for, corrupting unrelated content printed earlier in the turn. Once
# the reply is done, the full thing is printed once (unbounded) via render_answer.
ANSWER_PREVIEW_LINES = 6


def render_answer_preview(model_name: str, text: str) -> Group:
    """Bounded live preview of the reply while it streams: header plus only the last
    few wrapped lines, mirroring render_thinking's scrolling-window approach.
    """
    header = Text.from_markup(f"[bold {ACCENT}]{GLYPH_BULLET} Agent ({model_name})[/bold {ACCENT}]")
    width = max(20, console.width)
    wrapped = Text(text).wrap(console, width)
    visible = wrapped[-ANSWER_PREVIEW_LINES:]
    return Group(header, *visible)


def render_thinking(text: str, elapsed: float) -> Group:
    """A live preview of streaming reasoning tokens: an animated header plus only the
    last few wrapped lines of the chain-of-thought so far, so a long thinking trace
    scrolls in place instead of flooding the screen -- like looking through a small
    window onto the model's most recent thought.
    """
    header = Spinner("dots", text=Text(f" Thinking… {elapsed:.0f}s", style="dim italic"))
    width = max(20, console.width - 2)
    wrapped = Text(text, style="dim italic").wrap(console, width)
    visible = wrapped[-THINKING_VISIBLE_LINES:]
    body = [Padding(line, (0, 0, 0, 2)) for line in visible]
    return Group(header, *body)


def print_reply(model_name: str, text: str, elapsed: float, usage: dict | None = None) -> None:
    header = Text.from_markup(f"[bold {ACCENT}]{GLYPH_BULLET} Agent ({model_name}):[/bold {ACCENT}] ")
    if not text.strip():
        console.print(header + Text("(empty response)", style="dim"))
    else:
        # Markdown always wraps at the console's full width, so printing it right after the
        # header (no newline) would silently overflow that width by the header's length on
        # line 1. Render it at a narrower width instead and stitch the header onto the first
        # rendered line at the segment level, so the reply starts on the header's own line.
        body_width = max(20, console.width - header.cell_len)
        lines = console.render_lines(Markdown(text), console.options.update(width=body_width), pad=False)
        if lines:
            console.print(Segments(list(header.render(console)) + lines[0] + [Segment.line()]), end="")
            for line in lines[1:]:
                console.print(Segments(line + [Segment.line()]), end="")
        else:
            console.print(header)
    console.print(f"[dim]{format_status(text, elapsed, usage)}[/dim]")


def chat_loop(models: list[str]) -> None:
    model_name = choose_model(models)
    checkpointer = MemorySaver()
    graph_app = build_app(model_name, checkpointer)
    state = SessionState(model_name)
    logger = SessionLogger(model_name)

    session: PromptSession = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        completer=SlashCompleter(models),
        complete_while_typing=True,
    )

    console.print(Rule(style="grey50"))
    console.print(
        f"[dim]model[/dim]  [bold {ACCENT}]{model_name}[/bold {ACCENT}]   "
        f"[dim]log[/dim]  {logger.path.relative_to(SESSIONS_DIR.parent)}   "
        f"[dim]· /help for commands[/dim]"
    )
    console.print()

    try:
        while True:
            try:
                user_text = session.prompt(
                    render_prompt_message,
                    bottom_toolbar=lambda: render_toolbar(state),
                    refresh_interval=1.0,
                ).strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_text:
                continue

            if user_text in ("/exit", "/quit"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif user_text == "/help":
                print_help()
            elif user_text == "/clear":
                checkpointer = MemorySaver()
                graph_app = build_app(model_name, checkpointer)
                state.turn_count = 0
                logger.log_event("clear", model=model_name)
                print_ok("Conversation history cleared.")
            elif user_text == "/model" or user_text.startswith("/model "):
                arg = user_text[len("/model") :].strip()
                matched = match_model(arg, models) if arg else choose_model(models, current=model_name)
                if arg and not matched:
                    suggestion = suggest_model(arg, models)
                    hint = f" Did you mean [bold]{suggestion}[/bold]?" if suggestion else ""
                    print_warn(f"Unknown model '{arg}'.{hint} Keeping [bold {ACCENT}]{model_name}[/bold {ACCENT}].")
                else:
                    logger.log_event("model_switch", from_model=model_name, to_model=matched)
                    if matched != model_name:
                        unload_model(model_name)
                    model_name = matched
                    graph_app = build_app(model_name, checkpointer)
                    state.model_name = model_name
                    print_ok(f"Switched to [bold {ACCENT}]{model_name}[/bold {ACCENT}]")
            elif user_text == "/task":
                state.task_id += 1
                logger.log_event("task_boundary", task_id=state.task_id)
                print_ok(f"Started task #{state.task_id}.")
            elif user_text.startswith("/rate"):
                arg = user_text[len("/rate") :].strip().lower()
                if arg not in ("correct", "incorrect", "partial"):
                    print_warn("Usage: /rate correct|incorrect|partial")
                elif logger.annotate_last_turn(outcome=arg):
                    print_ok(f"Tagged last reply's outcome as [bold]{arg}[/bold].")
                else:
                    print_warn("No reply yet to rate.")
            elif user_text.startswith("/failure"):
                arg = user_text[len("/failure") :].strip().lower()
                if not arg:
                    print_warn(f"Usage: /failure <mode>, e.g. one of: {', '.join(FAILURE_MODES)}")
                elif logger.annotate_last_turn(failure_mode=arg):
                    print_ok(f"Tagged last reply's failure mode as [bold]{arg}[/bold].")
                else:
                    print_warn("No reply yet to tag.")
            elif user_text in ("/necessary", "/unnecessary"):
                necessary = user_text == "/necessary"
                if logger.annotate_last_turn(tool_call_necessary=necessary):
                    label = "necessary" if necessary else "unnecessary"
                    print_ok(f"Tagged last reply's tool call(s) as [bold]{label}[/bold].")
                else:
                    print_warn("No reply yet to tag.")
            elif user_text.startswith("/note"):
                arg = user_text[len("/note") :].strip()
                if not arg:
                    print_warn("Usage: /note <text>")
                elif logger.annotate_last_turn(note=arg):
                    print_ok("Note added to last reply.")
                else:
                    print_warn("No reply yet to annotate.")
            elif user_text.startswith("/"):
                print_err(f"Unknown command '{user_text}'. Type [bold]/help[/bold] for a list.")
            else:
                full_reply = ""
                streaming_started = False
                answer_live: Live | None = None
                final_meta: dict = {}
                llm_calls: list[dict] = []       # one entry per LLM generation round in this turn
                tool_events: list[dict] = []     # one entry per tool call (native or text-based) in this turn
                malformed_attempts: list[str] = []  # text that looked like a tool call but never parsed into one
                pending_tool_indices: list[int] = []  # tool_events indices awaiting a ToolMessage result
                tool_call_starts: dict[int, float] = {}
                round_had_tool_calls = False
                fake_call_announced = False
                cancelled = False
                error: Exception | None = None
                start_time = time.monotonic()
                spinner = Spinner("dots", text=Text(" thinking...", style=f"italic {ACCENT}"))

                def new_live() -> Live:
                    # A fresh Live object every time, never a restarted one: Live's internal
                    # LiveRender keeps its last-rendered height (_shape) across a stop()+
                    # start() cycle on the same instance, so restarting an already-used Live
                    # makes its next redraw erase based on a stale height from whatever it
                    # rendered before -- corrupting unrelated content printed in between.
                    return Live(spinner, console=console, refresh_per_second=12, transient=True)

                live = new_live()
                live.start()

                thinking_active = False   # currently streaming a live reasoning preview
                thinking_start = 0.0
                thinking_text = ""

                def finish_thinking() -> None:
                    """Collapse the live reasoning preview into a one-line summary, if one was showing."""
                    nonlocal thinking_active
                    if not thinking_active:
                        return
                    elapsed_think = time.monotonic() - thinking_start
                    live.stop()
                    console.print(f"[dim]{GLYPH_THINK} Thought for {elapsed_think:.1f}s[/dim]")
                    thinking_active = False

                def finish_answer() -> None:
                    """Stop the bounded live preview and print the finished reply as
                    rendered Markdown, once, if one was showing."""
                    nonlocal streaming_started, answer_live
                    if not streaming_started:
                        return
                    answer_live.stop()
                    console.print(render_answer(model_name, full_reply))
                    answer_live = None
                    streaming_started = False

                try:
                    for chunk in stream_reply(graph_app, user_text):
                        # response_metadata arrives on its own chunk at the end of every LLM generation
                        # round (tool-deciding rounds included) -- capture it unconditionally, before any
                        # `continue` below, or intermediate rounds' token usage is silently lost.
                        if chunk.response_metadata:
                            usage = extract_usage(chunk.response_metadata)
                            llm_calls.append({
                                "done_reason": chunk.response_metadata.get("done_reason"),
                                **usage,
                                "raw_metrics": chunk.response_metadata,
                            })
                            final_meta = chunk.response_metadata
                            if fake_call_announced and not round_had_tool_calls:
                                parsed = parse_fake_tool_call(full_reply)
                                if parsed:
                                    idx = len(tool_events)
                                    tool_events.append({
                                        "index": idx,
                                        "tool": parsed["name"],
                                        "args": parsed["args"],
                                        "native": False,
                                        "raw_call_text": full_reply,
                                        "result": None,
                                        "result_chars": None,
                                        "looks_like_error": None,
                                        "duration_seconds": None,
                                        "retry_of_index": None,
                                    })
                                    pending_tool_indices.append(idx)
                                    tool_call_starts[idx] = time.monotonic()
                                else:
                                    malformed_attempts.append(full_reply)
                            round_had_tool_calls = False

                        reasoning_piece = (chunk.additional_kwargs or {}).get("reasoning_content")
                        if reasoning_piece:
                            if not thinking_active:
                                thinking_active = True
                                thinking_start = time.monotonic()
                                thinking_text = ""
                                if not live.is_started:
                                    live = new_live()
                                    live.start()
                            thinking_text += reasoning_piece
                            live.update(render_thinking(thinking_text, time.monotonic() - thinking_start))
                            continue

                        tool_calls = getattr(chunk, "tool_calls", None) or []
                        if tool_calls:
                            finish_thinking()
                            finish_answer()
                            round_had_tool_calls = True
                            # Stop the live spinner before printing anything with a plain
                            # console.print(): Rich supports interleaving prints with an
                            # active Live by choreographing around it, but that dance is
                            # exactly the kind of cursor-position-dependent logic that's
                            # fragile under real timing -- simpler and safer to never have
                            # both active in the terminal at once.
                            if live.is_started:
                                live.stop()
                            for call in tool_calls:
                                name = call.get("name")
                                args = call.get("args") or {}
                                idx = len(tool_events)
                                tool_events.append({
                                    "index": idx,
                                    "tool": name,
                                    "args": args,
                                    "native": True,
                                    "raw_call_text": None,
                                    "result": None,
                                    "result_chars": None,
                                    "looks_like_error": None,
                                    "duration_seconds": None,
                                    "retry_of_index": None,
                                })
                                pending_tool_indices.append(idx)
                                tool_call_starts[idx] = time.monotonic()
                                console.print(f"[{ACCENT}]{GLYPH_BULLET}[/{ACCENT}] [bold]{name or 'tool'}[/bold]({_format_call_args(args)})")
                            spinner.update(text=Text(" running...", style=f"italic {ACCENT}"))
                            live = new_live()
                            live.start()
                            live.update(spinner)
                            continue
                        if isinstance(chunk, ToolMessage):
                            finish_thinking()
                            finish_answer()
                            if live.is_started:
                                live.stop()
                            result_name = getattr(chunk, "name", None)
                            match_idx = None
                            if result_name:
                                for i in pending_tool_indices:
                                    if tool_events[i]["tool"] == result_name:
                                        match_idx = i
                                        break
                            if match_idx is None and pending_tool_indices:
                                match_idx = pending_tool_indices[0]
                            if match_idx is not None:
                                pending_tool_indices.remove(match_idx)
                                ev = tool_events[match_idx]
                                ev["result"] = chunk.content
                                ev["result_chars"] = len(chunk.content or "")
                                ev["looks_like_error"] = _looks_like_tool_error(chunk.content)
                                call_start = tool_call_starts.pop(match_idx, None)
                                ev["duration_seconds"] = round(time.monotonic() - call_start, 3) if call_start is not None else None

                                preview = (chunk.content or "").strip().splitlines()[0] if chunk.content else ""
                                if len(preview) > 90:
                                    preview = preview[:87] + "..."
                                result_style = "red" if ev["looks_like_error"] else "dim"
                                console.print(f"  [{result_style}]{GLYPH_RESULT}[/{result_style}]  {preview or '(no output)'}")

                            full_reply = ""
                            fake_call_announced = False
                            round_had_tool_calls = False
                            live = new_live()
                            live.start()
                            live.update(spinner)
                            continue
                        if chunk.content:
                            finish_thinking()
                            full_reply += chunk.content
                        if not full_reply:
                            continue

                        fake_match = FAKE_CALL_PREFIX_RE.match(full_reply)
                        if fake_match:
                            if not fake_call_announced:
                                name = fake_match.group(1)
                                if live.is_started:
                                    live.stop()
                                console.print(f"[{ACCENT}]{GLYPH_BULLET}[/{ACCENT}] [bold]{name}[/bold](…) [dim]as text, no native tool-calling[/dim]")
                                spinner.update(text=Text(" running...", style=f"italic {ACCENT}"))
                                fake_call_announced = True
                            if not live.is_started:
                                live = new_live()
                                live.start()
                            live.update(spinner)
                            continue

                        if not streaming_started:
                            live.stop()
                            answer_live = Live(console=console, refresh_per_second=12, transient=True)
                            answer_live.start()
                            streaming_started = True

                        answer_live.update(render_answer_preview(model_name, full_reply))
                except KeyboardInterrupt:
                    cancelled = True
                except Exception as exc:
                    error = exc
                finally:
                    if live.is_started:
                        live.stop()
                    if answer_live is not None and answer_live.is_started:
                        answer_live.stop()

                # Chain retries: for each tool call, point at the previous call to the same tool in this
                # turn (if any), so a walk back through retry_of_index reconstructs the correction chain.
                last_index_by_tool: dict[str, int] = {}
                for ev in tool_events:
                    ev["retry_of_index"] = last_index_by_tool.get(ev["tool"])
                    if ev["tool"]:
                        last_index_by_tool[ev["tool"]] = ev["index"]

                def _repeated_identical_run(events: list[dict], min_repeat: int = IDENTICAL_CALL_REPEAT_THRESHOLD) -> bool:
                    run_key, run_len = None, 0
                    for e in events:
                        key = (e["tool"], json.dumps(e["args"], sort_keys=True, default=str))
                        run_len = run_len + 1 if key == run_key else 1
                        run_key = key
                        if run_len >= min_repeat:
                            return True
                    return False

                total_input_tokens = sum((c["input_tokens"] or 0) for c in llm_calls) if llm_calls else None
                total_output_tokens = sum((c["output_tokens"] or 0) for c in llm_calls) if llm_calls else None
                total_eval_seconds = sum((c["raw_metrics"].get("eval_duration") or 0) for c in llm_calls) / 1e9
                totals = {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "total_tokens": (total_input_tokens + total_output_tokens) if llm_calls else None,
                    "tokens_per_second": round(total_output_tokens / total_eval_seconds, 1) if total_output_tokens and total_eval_seconds else None,
                    "llm_call_count": len(llm_calls),
                    "tool_call_count": len(tool_events),
                    "unique_tools_used": sorted({e["tool"] for e in tool_events if e["tool"]}),
                    "retry_count": sum(1 for e in tool_events if e["retry_of_index"] is not None),
                }
                auto_flags = {
                    "malformed_tool_call": bool(malformed_attempts),
                    "tool_error": any(e["looks_like_error"] for e in tool_events),
                    "possible_infinite_loop": len(tool_events) >= TOOL_CALL_LOOP_THRESHOLD or _repeated_identical_run(tool_events),
                }

                elapsed = time.monotonic() - start_time
                if error is not None:
                    if streaming_started:
                        console.print()
                    print_err(f"Error talking to model: {error}")
                elif cancelled:
                    if streaming_started:
                        console.print()
                    console.print("[dim]⎋ cancelled[/dim]")
                elif streaming_started:
                    reply_text = full_reply
                    finish_answer()
                    console.print(f"[dim]{format_status(reply_text, elapsed, totals)}[/dim]")
                    state.turn_count += 1
                else:
                    print_reply(model_name, full_reply, elapsed, totals)
                    state.turn_count += 1

                if auto_flags["possible_infinite_loop"] or auto_flags["malformed_tool_call"]:
                    flagged = [k for k, v in auto_flags.items() if v]
                    print_warn(f"auto-flagged: {', '.join(flagged)} — see /failure to confirm/correct.")

                logger.log_turn(
                    task_id=state.task_id,
                    model=model_name,
                    user_message=user_text,
                    assistant_reply=full_reply,
                    elapsed_seconds=round(elapsed, 3),
                    cancelled=cancelled,
                    error=str(error) if error else None,
                    llm_calls=llm_calls,
                    tool_calls=tool_events,
                    malformed_attempts=malformed_attempts,
                    totals=totals,
                    auto_flags=auto_flags,
                    annotation={"outcome": None, "failure_mode": None, "tool_call_necessary": None, "note": None},
                )

            console.print()
    finally:
        unload_model(model_name)
        logger.close()


def main() -> None:
    console.print()
    try:
        with console.status(f"[{ACCENT}]Connecting to Ollama…[/{ACCENT}]", spinner="dots"):
            models = get_installed_models()
    except RuntimeError as exc:
        print_err(str(exc))
        sys.exit(1)

    print_banner(len(models))

    if not models:
        print_warn("No local Ollama models found. Pull one first, e.g. [bold]ollama pull llama3[/bold]")
        sys.exit(1)

    chat_loop(models)


if __name__ == "__main__":
    main()
