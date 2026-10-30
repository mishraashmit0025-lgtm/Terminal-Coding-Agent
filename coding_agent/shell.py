"""A persistent bash session for the bash tool.

The model expects state to carry over between commands (cd, exported variables,
activated virtualenvs), so commands run in one long-lived bash process. Each
command is followed by a sentinel echo carrying the exit code; output is read
until the sentinel appears. A command that runs past its timeout gets the whole
session killed and restarted, and the model is told so."""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
import uuid


def find_bash() -> str:
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe"):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError("bash not found; install Git Bash on Windows")


class ShellSession:
    def __init__(self, cwd: str, timeout: float = 120.0, max_output: int = 30_000, env: dict | None = None):
        self.cwd = cwd
        self.timeout = timeout
        self.max_output = max_output
        self.env = {**os.environ, **(env or {}), "PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb"}
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.start()

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [find_bash(), "--noprofile", "--norc"], cwd=self.cwd, env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.lines = queue.Queue()
        t = threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True)
        t.start()

    @staticmethod
    def _pump(proc: subprocess.Popen, q: queue.Queue) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(5)

    def restart(self) -> str:
        self.close()
        self.start()
        return "shell restarted"

    def _truncate(self, out: str) -> str:
        if len(out) <= self.max_output:
            return out
        half = self.max_output // 2
        cut = len(out) - 2 * half
        return f"{out[:half]}\n... [{cut} characters omitted] ...\n{out[-half:]}"

    def run(self, command: str, timeout: float | None = None) -> tuple[str, int | None]:
        """Run a command; returns (output, exit code). Exit code None means it timed out."""
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        marker = f"__CMD_DONE_{uuid.uuid4().hex}__"
        assert self.proc is not None and self.proc.stdin is not None
        # stdin is closed for the command so a program waiting for input does not hang the session
        self.proc.stdin.write(f"{{ {command}\n}} < /dev/null\necho \"{marker} $?\"\n")
        self.proc.stdin.flush()
        out: list[str] = []
        deadline = timeout or self.timeout
        end = time.monotonic() + deadline
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                self.restart()
                partial = self._truncate("".join(out))
                return f"{partial}\n[command timed out after {deadline:.0f}s; the shell was restarted, so cwd and env were reset]", None
            try:
                line = self.lines.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                continue
            if line is None:
                self.start()
                return self._truncate("".join(out)) + "\n[shell exited; started a new one]", None
            if marker in line:
                before, _, code = line.partition(marker)
                if before:
                    out.append(before)
                try:
                    rc = int(code.strip())
                except ValueError:
                    rc = None
                return self._truncate("".join(out)), rc
            out.append(line)
