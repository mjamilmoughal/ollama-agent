"""A beautiful terminal chat agent for local Ollama models, built with LangGraph."""

from __future__ import annotations

import sys
from typing import Iterator

import ollama as ollama_client
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
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

COMMANDS = {
    "/model": "Switch the active Ollama model",
    "/clear": "Clear conversation history",
    "/help": "Show this help message",
    "/exit": "Quit the chat (also /quit)",
}


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


def stream_reply(graph_app, user_text: str) -> Iterator[str]:
    """Yield the assistant's reply token by token."""
    config = {"configurable": {"thread_id": THREAD_ID}}
    inputs = {"messages": [HumanMessage(content=user_text)]}
    for message_chunk, _metadata in graph_app.stream(inputs, config, stream_mode="messages"):
        if message_chunk.content:
            yield message_chunk.content


def print_banner() -> None:
    console.print()
    console.print(
        Panel(
            Text("🦙  OLLAMA CHAT AGENT", style="bold cyan", justify="center"),
            subtitle="[dim]powered by LangGraph[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
        )
    )


def choose_model(models: list[str], current: str | None = None) -> str:
    table = Table(box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model")
    for i, name in enumerate(models, start=1):
        marker = " [green](current)[/green]" if name == current else ""
        table.add_row(str(i), f"{name}{marker}")
    console.print(table)

    choice = Prompt.ask("[bold yellow]Select a model[/bold yellow] (number or name)")
    choice = choice.strip()
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    if choice in models:
        return choice
    console.print(f"[red]Unknown model '{choice}', keeping current selection.[/red]")
    return current or models[0]


def print_help() -> None:
    table = Table(title="Commands", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("Description")
    for cmd, desc in COMMANDS.items():
        table.add_row(cmd, desc)
    console.print(table)


def chat_loop(models: list[str]) -> None:
    model_name = choose_model(models)
    checkpointer = MemorySaver()
    graph_app = build_app(model_name, checkpointer)

    console.print(Rule(style="cyan"))
    console.print(f"[dim]Chatting with[/dim] [bold cyan]{model_name}[/bold cyan]  [dim]· /help for commands[/dim]")
    console.print(Rule(style="cyan"))

    while True:
        try:
            user_text = Prompt.ask(f"\n[bold green]you[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_text:
            continue

        if user_text in ("/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/clear":
            checkpointer = MemorySaver()
            graph_app = build_app(model_name, checkpointer)
            console.print("[yellow]Conversation history cleared.[/yellow]")
            continue
        if user_text == "/model":
            model_name = choose_model(models, current=model_name)
            graph_app = build_app(model_name, checkpointer)
            console.print(f"[green]Switched to[/green] [bold cyan]{model_name}[/bold cyan]")
            continue

        full_reply = ""
        spinner = Spinner("dots", text=Text(" thinking...", style="italic cyan"))
        try:
            with Live(spinner, console=console, refresh_per_second=12, transient=True) as live:
                for token in stream_reply(graph_app, user_text):
                    full_reply += token
                    live.update(Text(full_reply))
        except Exception as exc:
            console.print(f"[bold red]Error talking to model:[/bold red] {exc}")
            continue

        console.print(
            Panel(
                Markdown(full_reply) if full_reply.strip() else Text("(empty response)", style="dim"),
                title=f"🤖 {model_name}",
                title_align="left",
                border_style="magenta",
                box=box.ROUNDED,
            )
        )


def main() -> None:
    print_banner()
    try:
        models = get_installed_models()
    except RuntimeError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        sys.exit(1)

    if not models:
        console.print(
            "[bold red]No local Ollama models found.[/bold red] "
            "Pull one first, e.g. [bold]ollama pull llama3[/bold]"
        )
        sys.exit(1)

    chat_loop(models)


if __name__ == "__main__":
    main()
