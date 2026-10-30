"""Command line entry point.

    cagent run "make the failing test in tests/test_dates.py pass"
    cagent run --mode auto --max-cost 1 "add a --verbose flag to cli.py"
    cagent undo            # restore every file the last session touched
    cagent sessions        # list sessions and what they changed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .agent import PRICES, Agent, Events
from .policy import Approver
from .workspace import Workspace

console = Console()


def _short(value: str, n: int = 400) -> str:
    return value if len(value) <= n else value[:n] + f" ... ({len(value) - n} more chars)"


def _events() -> Events:
    def on_tool_call(name: str, inp: dict) -> None:
        if name == "bash":
            console.print(f"[bold cyan]$[/] {inp.get('command', '(restart)')}")
        else:
            console.print(f"[bold magenta]edit[/] {inp.get('command')} {inp.get('path', '')}")

    def on_tool_result(name: str, out: str, is_error: bool) -> None:
        console.print(_short(out), style="red" if is_error else "dim", markup=False, highlight=False)

    return Events(on_text=lambda t: console.print(Markdown(t)), on_tool_call=on_tool_call,
                  on_tool_result=on_tool_result)


def cmd_run(args: argparse.Namespace) -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY is not set[/]")
        return 2
    task = " ".join(args.task) if args.task else sys.stdin.read()
    ws = Workspace(args.dir)
    agent = Agent(ws, model=args.model, effort=args.effort, approver=Approver(args.mode),
                  max_turns=args.max_turns, max_cost_usd=args.max_cost, command_timeout=args.timeout,
                  events=_events())
    console.print(Panel(task, title=f"{args.model} in {ws.root}", border_style="blue"))
    try:
        res = agent.run(task)
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted; `cagent undo` restores the files this session changed[/]")
        return 130
    u = res.usage
    console.print(Panel(
        f"status: {res.status}\nturns: {u.turns}\n"
        f"tokens: {u.input_tokens:,} in, {u.cache_read_tokens:,} cached, {u.output_tokens:,} out\n"
        f"cost: ${u.cost(args.model):.4f}\n"
        f"files changed: {', '.join(res.files_changed) or 'none'}\n"
        f"transcript: {res.transcript_path}",
        title="session", border_style="green" if res.status == "done" else "yellow"))
    return 0 if res.status == "done" else 1


def cmd_undo(args: argparse.Namespace) -> int:
    restored = Workspace.undo(args.dir, args.session)
    if not restored:
        console.print("nothing to undo")
        return 1
    for rel in restored:
        console.print(f"restored {rel}")
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    root = Path(args.dir).resolve() / ".agent" / "sessions"
    if not root.is_dir():
        console.print("no sessions")
        return 0
    for d in sorted(root.iterdir()):
        manifest = d / "manifest.json"
        files = list(json.loads(manifest.read_text(encoding="utf-8"))) if manifest.exists() else []
        console.print(f"{d.name}  {len(files)} file(s)  {' '.join(files)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cagent", description="A coding agent for your terminal.")
    p.add_argument("--dir", default=".", help="workspace root (default: current directory)")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a task")
    r.add_argument("task", nargs="*", help="what to do (read from stdin if omitted)")
    r.add_argument("--model", default="claude-opus-5-5", choices=sorted(PRICES))
    r.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--mode", default="ask", choices=["ask", "auto", "yolo"],
                   help="ask: prompt for commands off the allowlist; auto: refuse them; yolo: run anything")
    r.add_argument("--max-turns", type=int, default=60)
    r.add_argument("--max-cost", type=float, default=5.0, help="stop once the session has cost this many USD")
    r.add_argument("--timeout", type=float, default=180.0, help="seconds per shell command")
    r.set_defaults(fn=cmd_run)

    u = sub.add_parser("undo", help="restore the files changed by a session")
    u.add_argument("session", nargs="?", help="session id (default: the latest)")
    u.set_defaults(fn=cmd_undo)

    s = sub.add_parser("sessions", help="list sessions")
    s.set_defaults(fn=cmd_sessions)

    args = p.parse_args(argv)
    return args.fn(args)
