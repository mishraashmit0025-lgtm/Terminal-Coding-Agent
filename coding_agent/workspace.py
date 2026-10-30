"""File operations for the text editor tool, confined to one project root.

Every path the model sends is untrusted. It is resolved (following symlinks)
and must stay inside the root; anything else is rejected before a file is
touched. Every write is backed up first so a whole session can be undone."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

SKIP_DIRS = {".git", ".agent", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}


class ToolError(Exception):
    """An error the model should see and recover from (returned with is_error)."""


class Workspace:
    def __init__(self, root: str | Path, session: str | None = None, max_view_chars: int = 20_000):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"workspace {self.root} is not a directory")
        self.session = session or time.strftime("%Y%m%d-%H%M%S")
        self.backup_dir = self.root / ".agent" / "sessions" / self.session
        self.max_view_chars = max_view_chars
        self.changed: dict[str, str | None] = {}  # rel path -> backup file name, None if it did not exist

    # ---------------------------------------------------------------- paths
    def resolve(self, path: str) -> Path:
        if not path or "\x00" in path:
            raise ToolError("path is empty or invalid")
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        try:
            rp = p.resolve()
        except OSError as e:
            raise ToolError(f"cannot resolve {path}: {e}") from e
        if rp != self.root and not rp.is_relative_to(self.root):
            raise ToolError(f"{path} is outside the workspace {self.root}")
        if ".agent" in rp.relative_to(self.root).parts:
            raise ToolError("the .agent directory is reserved for session data")
        return rp

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    # ---------------------------------------------------------------- backup / undo
    def _backup(self, p: Path) -> None:
        rel = self.rel(p)
        if rel in self.changed:
            return  # keep the state from before the session's first edit
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        if p.exists():
            name = f"{len(self.changed):04d}-{p.name}"
            shutil.copy2(p, self.backup_dir / name)
            self.changed[rel] = name
        else:
            self.changed[rel] = None
        (self.backup_dir / "manifest.json").write_text(json.dumps(self.changed, indent=2), encoding="utf-8")

    @staticmethod
    def undo(root: str | Path, session: str | None = None) -> list[str]:
        """Restore every file touched in a session (the latest by default)."""
        sessions = Path(root).resolve() / ".agent" / "sessions"
        if not sessions.is_dir():
            return []
        names = sorted(d.name for d in sessions.iterdir() if (d / "manifest.json").exists())
        if not names:
            return []
        sdir = sessions / (session or names[-1])
        manifest = json.loads((sdir / "manifest.json").read_text(encoding="utf-8"))
        restored = []
        for rel, backup in manifest.items():
            target = Path(root).resolve() / rel
            if backup is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copy2(sdir / backup, target)
            restored.append(rel)
        shutil.rmtree(sdir)
        return restored

    # ---------------------------------------------------------------- editor commands
    def view(self, path: str, view_range: list[int] | None = None) -> str:
        p = self.resolve(path)
        if p.is_dir():
            return self._listing(p)
        if not p.exists():
            raise ToolError(f"{path} does not exist")
        text = self._read(p)
        lines = text.split("\n")
        start, end = 1, len(lines)
        if view_range:
            if len(view_range) != 2:
                raise ToolError("view_range must be [start, end]")
            start, end = view_range
            end = len(lines) if end == -1 else end
            if not (1 <= start <= max(end, 1) <= len(lines)):
                raise ToolError(f"view_range {view_range} is outside 1..{len(lines)}")
        out = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(start, end + 1))
        if len(out) > self.max_view_chars:
            out = out[: self.max_view_chars] + f"\n... truncated; use view_range to see lines past this point"
        return out

    def _listing(self, d: Path, depth: int = 2) -> str:
        rows = []

        def walk(cur: Path, level: int) -> None:
            for child in sorted(cur.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
                if child.name in SKIP_DIRS or child.name.startswith(".") and child.is_dir():
                    continue
                rows.append(f"{'  ' * level}{child.name}{'/' if child.is_dir() else ''}")
                if child.is_dir() and level + 1 < depth:
                    walk(child, level + 1)
                if len(rows) > 500:
                    return

        walk(d, 0)
        return f"{self.rel(d)}/\n" + "\n".join(rows)

    @staticmethod
    def _read(p: Path) -> str:
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            raise ToolError(f"{p.name} looks like a binary file")
        return raw.decode("utf-8", errors="replace")

    def create(self, path: str, file_text: str) -> str:
        p = self.resolve(path)
        if p.is_dir():
            raise ToolError(f"{path} is a directory")
        self._backup(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding="utf-8", newline="")
        return f"wrote {self.rel(p)} ({len(file_text.splitlines())} lines)"

    def str_replace(self, path: str, old_str: str, new_str: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"{path} does not exist")
        text = self._read(p)
        n = text.count(old_str) if old_str else 0
        if n == 0:
            raise ToolError("old_str was not found; view the file and copy the text exactly, including whitespace")
        if n > 1:
            raise ToolError(f"old_str occurs {n} times; include more surrounding lines so it is unique")
        self._backup(p)
        new = text.replace(old_str, new_str, 1)
        p.write_text(new, encoding="utf-8", newline="")
        # show the edited region so the model can check its work
        line = new[: new.index(new_str)].count("\n") + 1 if new_str else text[: text.index(old_str)].count("\n") + 1
        lines = new.split("\n")
        lo, hi = max(1, line - 3), min(len(lines), line + new_str.count("\n") + 3)
        snippet = "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(lo, hi + 1))
        return f"edited {self.rel(p)}:\n{snippet}"

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            raise ToolError(f"{path} does not exist")
        lines = self._read(p).split("\n")
        if not 0 <= insert_line <= len(lines):
            raise ToolError(f"insert_line must be between 0 and {len(lines)}")
        self._backup(p)
        new_lines = insert_text.split("\n")
        lines[insert_line:insert_line] = new_lines
        p.write_text("\n".join(lines), encoding="utf-8", newline="")
        return f"inserted {len(new_lines)} lines after line {insert_line} of {self.rel(p)}"

    def run_editor(self, inp: dict) -> str:
        cmd = inp.get("command")
        try:
            if cmd == "view":
                return self.view(inp["path"], inp.get("view_range"))
            if cmd == "create":
                return self.create(inp["path"], inp.get("file_text", ""))
            if cmd == "str_replace":
                return self.str_replace(inp["path"], inp.get("old_str", ""), inp.get("new_str", ""))
            if cmd == "insert":
                return self.insert(inp["path"], int(inp["insert_line"]), inp.get("insert_text", ""))
        except KeyError as e:
            raise ToolError(f"missing parameter {e.args[0]!r} for {cmd}") from e
        raise ToolError(f"unknown editor command {cmd!r}")
