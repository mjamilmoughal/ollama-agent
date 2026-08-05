"""A beautiful terminal chat agent for local Ollama models, built with LangGraph."""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import ollama as ollama_client
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from rich import box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

console = Console()
THREAD_ID = "session"
HISTORY_FILE = Path.home() / ".ollama_chat_history"
SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"

# ANSI color names, shared between Rich markup and prompt_toolkit HTML tags,
# so the input line and the panels always agree on a palette.
PT_USER, PT_ACCENT, PT_MUTED = "ansigreen", "ansicyan", "ansibrightblack"

COMMANDS = {
    "/model": "Switch the active Ollama model (name, number, or fuzzy match)",
    "/clear": "Clear conversation history",
    "/help": "Show this help message",
    "/exit": "Quit the chat (also /quit)",
}


class SessionState:
    """Tracks the bits of live session info shown in the input toolbar."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.turn_count = 0
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


def build_app(model_name: str, checkpointer: MemorySaver):
    """Compile a single-node LangGraph chat graph backed by the given Ollama model."""
    llm = ChatOllama(model=model_name)

    def chat_node(state: MessagesState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
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
            Text("🦙  OLLAMA CHAT AGENT", style="bold cyan", justify="center"),
            subtitle=f"[dim]{model_count} model{plural} available · powered by LangGraph[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )


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
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model")
    for i, name in enumerate(models, start=1):
        marker = " [green](current)[/green]" if name == current else ""
        table.add_row(str(i), f"{name}{marker}")
    console.print(table)

    choice = Prompt.ask("[bold yellow]Select a model[/bold yellow] (number or name)").strip()
    matched = match_model(choice, models)
    if matched:
        return matched

    suggestion = suggest_model(choice, models)
    if suggestion:
        console.print(f"[yellow]Unknown model '{choice}'. Did you mean [bold]{suggestion}[/bold]?[/yellow]")
    else:
        console.print(f"[red]Unknown model '{choice}'.[/red]")
    return current or models[0]


def print_help() -> None:
    table = Table(title="Commands", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("Description")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)
    console.print(
        "[dim]Tip: ↑/↓ recalls history · Tab completes commands and model names · "
        "Ctrl+C cancels a reply · Ctrl+D exits.[/dim]"
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
    return HTML(
        f"<{PT_MUTED}>{now}</{PT_MUTED}> <{PT_USER}><b>you</b></{PT_USER}> <{PT_ACCENT}>❯</{PT_ACCENT}> "
    )


def render_toolbar(state: SessionState) -> HTML:
    clock = format_duration(time.monotonic() - state.start_time)
    return HTML(
        f" <b>{state.model_name}</b>  ·  turn {state.turn_count}  ·  {clock}"
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


def render_reply_panel(model_name: str, text: str, elapsed: float, streaming: bool, meta: dict | None = None) -> Panel:
    usage = extract_usage(meta)

    if usage["input_tokens"] is not None and usage["output_tokens"] is not None:
        token_part = f"↑{usage['input_tokens']} ↓{usage['output_tokens']}"
        if usage["tokens_per_second"] is not None:
            token_part += f" · {usage['tokens_per_second']:.0f} tok/s"
    else:
        token_part = f"~{len(text.split())} tok"

    body = Markdown(text) if text.strip() else Text("(empty response)", style="dim")
    status = f"{elapsed:.1f}s · {token_part}"
    subtitle = f"[dim italic]{status} · streaming…[/dim italic]" if streaming else f"[dim]{status}[/dim]"
    return Panel(
        body,
        title=f"Agent: {model_name}",
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style="cyan" if streaming else "magenta",
        box=box.ROUNDED,
    )


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

    console.print(Rule(style="cyan"))
    console.print(f"[dim]Chatting with[/dim] [bold cyan]{model_name}[/bold cyan]  [dim]· /help for commands[/dim]")
    console.print(f"[dim]Logging this session to[/dim] [bold]{logger.path.relative_to(SESSIONS_DIR.parent)}[/bold]")
    console.print(Rule(style="cyan"))

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
                console.print("[yellow]Conversation history cleared.[/yellow]")
            elif user_text == "/model" or user_text.startswith("/model "):
                arg = user_text[len("/model") :].strip()
                matched = match_model(arg, models) if arg else choose_model(models, current=model_name)
                if arg and not matched:
                    suggestion = suggest_model(arg, models)
                    hint = f" Did you mean [bold]{suggestion}[/bold]?" if suggestion else ""
                    console.print(f"[yellow]Unknown model '{arg}'.{hint} Keeping [bold cyan]{model_name}[/bold cyan].[/yellow]")
                else:
                    logger.log_event("model_switch", from_model=model_name, to_model=matched)
                    model_name = matched
                    graph_app = build_app(model_name, checkpointer)
                    state.model_name = model_name
                    console.print(f"[green]Switched to[/green] [bold cyan]{model_name}[/bold cyan]")
            elif user_text.startswith("/"):
                console.print(f"[red]Unknown command[/red] '{user_text}'. Type [bold]/help[/bold] for a list.")
            else:
                full_reply = ""
                final_meta: dict = {}
                cancelled = False
                error: Exception | None = None
                start_time = time.monotonic()
                spinner = Spinner("dots", text=Text(" thinking...", style="italic cyan"))
                try:
                    with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                        for chunk in stream_reply(graph_app, user_text):
                            if chunk.content:
                                full_reply += chunk.content
                            if chunk.response_metadata:
                                final_meta = chunk.response_metadata
                            elapsed = time.monotonic() - start_time
                            live.update(render_reply_panel(model_name, full_reply, elapsed, streaming=True, meta=final_meta))
                except KeyboardInterrupt:
                    cancelled = True
                except Exception as exc:
                    error = exc

                elapsed = time.monotonic() - start_time
                if error is not None:
                    console.print(f"[bold red]Error talking to model:[/bold red] {error}")
                elif cancelled:
                    console.print("[dim]⎋ cancelled[/dim]")
                else:
                    console.print(render_reply_panel(model_name, full_reply, elapsed, streaming=False, meta=final_meta))
                    state.turn_count += 1

                logger.log_turn(
                    model=model_name,
                    user_message=user_text,
                    assistant_reply=full_reply,
                    elapsed_seconds=round(elapsed, 3),
                    cancelled=cancelled,
                    error=str(error) if error else None,
                    ollama_metrics=final_meta,
                    **extract_usage(final_meta),
                )

            console.print()
    finally:
        logger.close()


def main() -> None:
    console.print()
    try:
        with console.status("[cyan]Connecting to Ollama…[/cyan]", spinner="dots"):
            models = get_installed_models()
    except RuntimeError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red", box=box.ROUNDED, title="Connection error"))
        sys.exit(1)

    print_banner(len(models))

    if not models:
        console.print(
            Panel(
                "[bold yellow]No local Ollama models found.[/bold yellow]\n"
                "Pull one first, e.g. [bold]ollama pull llama3[/bold]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        sys.exit(1)

    chat_loop(models)


if __name__ == "__main__":
    main()
