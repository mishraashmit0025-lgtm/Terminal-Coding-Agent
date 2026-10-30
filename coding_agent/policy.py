"""Which shell commands may run without asking.

Commands come from the model, so they are untrusted. In `auto` mode a command
runs only if every segment of it (split on |, &&, ||, ;) starts with an
allowlisted program and it uses no command substitution, redirection to
absolute paths, or other constructs that could hide a second program. Anything
else is refused, and the model is told why. In `ask` mode the user is asked
instead of refusing. `yolo` allows everything and is meant for a disposable
container."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

SAFE_PROGRAMS = {
    # read-only inspection
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "tree", "pwd", "echo", "printf", "diff", "sort",
    "uniq", "cut", "tr", "basename", "dirname", "realpath", "stat", "file", "which", "true", "false", "test", "[",
    "sed", "awk", "jq", "less", "more", "env", "date", "du",
    # build and test
    "python", "python3", "pip", "pip3", "uv", "pytest", "ruff", "mypy", "black", "node", "npm", "npx", "pnpm",
    "yarn", "tsc", "go", "cargo", "rustc", "make", "cmake", "javac", "java", "mvn", "gradle", "dotnet",
    # files inside the workspace
    "mkdir", "touch", "cp", "mv", "cd",
    # version control, minus anything that talks to a remote (checked below)
    "git",
}

GIT_DENY = {"push", "pull", "fetch", "clone", "remote", "submodule", "config", "credential", "reset", "clean",
            "checkout", "restore", "rebase", "filter-branch", "gc", "reflog"}

FORBIDDEN = [
    (re.compile(r"\$\(|`"), "command substitution"),
    (re.compile(r"(^|[^\w])sudo\b"), "sudo"),
    (re.compile(r">\s*/(?!dev/null)"), "redirecting output to an absolute path"),
    (re.compile(r"(^|\s)\.\.(/|\s|$)"), "paths that leave the workspace"),
    (re.compile(r"\brm\b"), "rm (deletions go through the editor or need approval)"),
    (re.compile(r"\b(curl|wget|ssh|scp|nc|ncat)\b"), "network access"),
]

SEGMENT_SPLIT = re.compile(r"\|\||&&|[|;&\n]")
FD_REDIRECT = re.compile(r"\d*>&\d+")  # 2>&1 and friends, removed before splitting on &


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


def check(command: str) -> Decision:
    for pat, what in FORBIDDEN:
        if pat.search(command):
            return Decision(False, f"not allowed automatically: {what}")
    for seg in SEGMENT_SPLIT.split(FD_REDIRECT.sub(" ", command)):
        seg = seg.strip()
        if not seg:
            continue
        try:
            words = shlex.split(seg)
        except ValueError as e:
            return Decision(False, f"could not parse command: {e}")
        # skip leading VAR=value assignments
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]
        if not words:
            continue
        prog = words[0].rsplit("/", 1)[-1]
        if prog not in SAFE_PROGRAMS:
            return Decision(False, f"{prog!r} is not on the allowlist")
        if prog == "git" and len(words) > 1 and words[1] in GIT_DENY:
            return Decision(False, f"git {words[1]} is not allowed automatically")
        if prog in ("pip", "pip3", "npm", "pnpm", "yarn") and len(words) > 1 and words[1] in ("publish", "login"):
            return Decision(False, f"{prog} {words[1]} is not allowed")
        if prog in ("python", "python3") and "-c" in words and re.search(r"\b(os\.system|subprocess|shutil\.rmtree)", seg):
            return Decision(False, "python -c that spawns processes or deletes trees")
    return Decision(True)


class Approver:
    """Decides whether to run a command: auto (allowlist only), ask, or yolo."""

    def __init__(self, mode: str = "ask", prompt=input):
        if mode not in ("auto", "ask", "yolo"):
            raise ValueError("mode must be auto, ask or yolo")
        self.mode = mode
        self.prompt = prompt
        self.always: set[str] = set()

    def approve(self, command: str) -> Decision:
        if self.mode == "yolo":
            return Decision(True)
        d = check(command)
        if d.allowed or command in self.always:
            return Decision(True)
        if self.mode == "auto":
            return d
        ans = self.prompt(f"\nRun this command? ({d.reason})\n  $ {command}\n[y]es / [n]o / [a]lways: ").strip().lower()
        if ans in ("a", "always"):
            self.always.add(command)
            return Decision(True)
        if ans in ("y", "yes"):
            return Decision(True)
        return Decision(False, "the user declined to run this command")
