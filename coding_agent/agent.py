"""The agent loop: send the conversation, run the tools Claude asks for, repeat."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .policy import Approver
from .shell import ShellSession
from .workspace import ToolError, Workspace

TOOLS = [
    # Anthropic-defined tools: declared by type and name only; the schema is built into the model
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 20000},
]

# USD per million tokens (first-party API list prices, October 2026)
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM = """You are a software engineer working in a repository through a bash shell and a file editor.

Workspace root: {root}
Operating system: {os}

How to work:
- Start by looking around: list the directory, read the README and the files related to the task.
- Make the smallest change that fully solves the task, in the style of the surrounding code.
- After changing code, run the relevant tests or the program to check the change actually works. If something fails, read the error and fix the cause.
- Prefer the editor's str_replace for edits; view a file before editing it.
- Do not commit, push, or delete files you did not create unless the task asks for it.
- When the task is done, reply with a short summary of what you changed and how you verified it, and stop calling tools."""


class Client(Protocol):
    """The slice of anthropic.Anthropic the agent uses, so tests can substitute a scripted model."""

    @property
    def messages(self) -> Any: ...


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    turns: int = 0

    def cost(self, model: str) -> float:
        inp, out = PRICES.get(model, (0.0, 0.0))
        # cache writes bill at 1.25x input, cache reads at 0.1x input
        return (self.input_tokens * inp + self.cache_write_tokens * inp * 1.25
                + self.cache_read_tokens * inp * 0.1 + self.output_tokens * out) / 1_000_000


@dataclass
class Events:
    """Callbacks for the UI. All optional."""

    on_text: Callable[[str], None] = lambda text: None
    on_tool_call: Callable[[str, dict], None] = lambda name, inp: None
    on_tool_result: Callable[[str, str, bool], None] = lambda name, out, is_error: None


@dataclass
class Result:
    status: str  # "done", "max_turns", "budget", "refusal", "error"
    final_text: str
    usage: Usage
    files_changed: list[str] = field(default_factory=list)
    transcript_path: str | None = None


def _dump(block) -> dict:
    """Assistant content goes back to the API exactly as received (thinking blocks included)."""
    return block.model_dump(mode="json", exclude_none=True) if hasattr(block, "model_dump") else dict(block)


class Agent:
    def __init__(self, workspace: Workspace, client: Client | None = None, model: str = "claude-opus-5-5",
                 effort: str = "high", approver: Approver | None = None, max_turns: int = 60,
                 max_cost_usd: float | None = 5.0, command_timeout: float = 180.0, events: Events | None = None,
                 transcript: bool = True):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.ws = workspace
        self.client = client
        self.model = model
        self.effort = effort
        self.approver = approver or Approver("ask")
        self.max_turns = max_turns
        self.max_cost = max_cost_usd
        self.events = events or Events()
        self.shell = ShellSession(str(workspace.root), timeout=command_timeout)
        self.usage = Usage()
        self.messages: list[dict] = []
        self.transcript = (workspace.backup_dir / "transcript.jsonl") if transcript else None

    # ------------------------------------------------------------------ tools
    def _run_tool(self, name: str, inp: dict) -> tuple[str, bool]:
        try:
            if name == "bash":
                if inp.get("restart"):
                    return self.shell.restart(), False
                cmd = inp.get("command", "")
                if not cmd.strip():
                    return "empty command", True
                decision = self.approver.approve(cmd)
                if not decision.allowed:
                    return f"command not run: {decision.reason}", True
                out, code = self.shell.run(cmd)
                if code is None:
                    return out, True
                return (out if out.strip() else "(no output)") + ("" if code == 0 else f"\n[exit code {code}]"), code != 0
            if name == "str_replace_based_edit_tool":
                return self.ws.run_editor(inp), False
            return f"unknown tool {name!r}", True
        except ToolError as e:
            return str(e), True
        except Exception as e:  # a bug in a tool handler must not kill the session
            return f"tool failed: {type(e).__name__}: {e}", True

    # ------------------------------------------------------------------ loop
    def _log(self, record: dict) -> None:
        if self.transcript is None:
            return
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcript, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _request(self) -> Any:
        return self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM.format(root=self.ws.root, os=f"{platform.system()} {platform.release()}"),
            tools=TOOLS,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},  # caches the growing conversation prefix turn to turn
        )

    def run(self, task: str) -> Result:
        self.messages.append({"role": "user", "content": task})
        self._log({"t": time.time(), "type": "task", "task": task, "model": self.model})
        final_text = ""
        try:
            for _ in range(self.max_turns):
                if self.max_cost is not None and self.usage.cost(self.model) >= self.max_cost:
                    return self._finish("budget", final_text)
                resp = self._request()
                u = resp.usage
                self.usage.turns += 1
                self.usage.input_tokens += u.input_tokens
                self.usage.output_tokens += u.output_tokens
                self.usage.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
                self.usage.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0
                content = [_dump(b) for b in resp.content]
                self.messages.append({"role": "assistant", "content": content})
                self._log({"t": time.time(), "type": "assistant", "stop_reason": resp.stop_reason, "content": content})

                texts = [b["text"] for b in content if b.get("type") == "text" and b.get("text")]
                for t in texts:
                    self.events.on_text(t)
                if texts:
                    final_text = texts[-1]

                if resp.stop_reason == "refusal":
                    return self._finish("refusal", final_text)
                if resp.stop_reason == "pause_turn":
                    continue  # server paused a long turn; sending the history back resumes it
                tool_uses = [b for b in content if b.get("type") == "tool_use"]
                if not tool_uses:
                    return self._finish("done", final_text)

                results = []
                for tu in tool_uses:
                    if resp.stop_reason == "max_tokens":
                        # the response was cut off, so this call's input may be incomplete: do not run it
                        out, err = "not run: the response hit max_tokens before this call was complete; retry it", True
                    else:
                        self.events.on_tool_call(tu["name"], tu["input"])
                        out, err = self._run_tool(tu["name"], tu["input"])
                        self.events.on_tool_result(tu["name"], out, err)
                    results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": out, "is_error": err})
                # every result for this turn goes back in one user message
                self.messages.append({"role": "user", "content": results})
                self._log({"t": time.time(), "type": "tool_results", "content": results})
            return self._finish("max_turns", final_text)
        finally:
            self.shell.close()

    def _finish(self, status: str, final_text: str) -> Result:
        self._log({"t": time.time(), "type": "end", "status": status, "usage": self.usage.__dict__,
                   "cost_usd": self.usage.cost(self.model)})
        return Result(status, final_text, self.usage, sorted(self.ws.changed),
                      str(self.transcript) if self.transcript else None)
