# harness_hooks

Three fail-open Claude Code hooks (stdlib only, written to stay Python 3.8-compatible (checked statically only; the host python3 is 3.12); any exception prints `{}` and exits 0).

- `bash_guard.py` (PreToolUse, Bash): denies a typed `rg` only when it carries a ripgrep-only option (a plain `rg -n pat dir` runs as `rtk grep` and works; see the probe table below), pytest without `CUDA_VISIBLE_DEVICES=`
  (rule `pytest-gpu`), and starting `scripts/queue_run.sh` while a worker is already running (rule `queue-dup`;
  `status` and `dry-run` are read-only and always allowed). Never rewrites the command. Heredoc bodies and quoted
  text are not scanned; when quoting is ambiguous (unbalanced quotes, or `$(`/`${`/backtick inside double quotes) no rule
  fires, and commands over 100,000 characters skip all rules (false negatives are accepted, false positives are not).
  History: a `py38` rule (deny bare `python3`, premise "bare python3 is 3.8") was built and then removed on
  2026-11-10 after verifier r1 found `/usr/bin/python3` is 3.12.3 here (no 3.8 exists); a bare python3 missing
  numpy/torch fails loudly with ImportError, so it needs no guard.
- `session_brief.py` (SessionStart): <= 6000-char brief: branch, STATE.md freshness (STALE if > 48 h), per-track
  `CODE | status | date | handoff`, queue counts and worker presence, GPUs 4-7 memory.
- `sh_syntax_check.py` (PostToolUse, Write|Edit): `bash -n` on written `.sh` files; blocks with the stderr on error.

## rg rule: probe table (2026-11-10)
A typed `rg ARGS` that starts a command is rewritten by the global rtk hook to `rtk grep ARGS`; a pipe-sink `rg` (`x | rg ARGS`) is not rewritten and can run real ripgrep via the shell function, so the guard may deny a sink `rg` that would work (bypass with `# guard-ok:rg`). `rtk rg ...` and `rtk proxy rg ...` are always denied (no rg binary: rc 127 / rc 1; use `rtk grep` or the Grep tool). Accepted false negatives (r4 m12): `rtk rg` inside `( … )`, after `sudo`/`xargs`, inside `bash -c '…'`, `$(…)` or backticks, and `rtk -v rg` are NOT caught. Each flag was probed as
`rtk grep <flag> "def emit" tools/harness_hooks/ ; echo rc=$?` (a file target where the directory form fails for unrelated reasons, marked F).
Verdict DENY = error or silently wrong; the rule fires on exactly these.

| flag | rc | result | verdict |
|---|---|---|---|
| `--files` | 2 | grep: option '--files' is ambiguous | DENY |
| `--files-with-matches` | 2 | accepted as grep long option (dir: "Is a directory") | excluded by spec (grep has `-l`) |
| `-g '*.py'` | 2 | grep: invalid option -- 'g' | DENY |
| `--glob` | 2 | unrecognized option | DENY |
| `-t py` | 0 | accepted but ignored: `-t md` and `-t rust` return the same .py match | DENY (silently wrong) |
| `--type` | 2 | unrecognized option | DENY |
| `-T py` | 2 | "py" taken as pattern, "def emit: No such file" | DENY |
| `--type-not`, `--type-list`, `--json`, `--multiline`, `--hidden`, `--sort`, `--vimgrep`, `--smart-case`, `--iglob` | 2 | unrecognized option (each) | DENY |
| `-S` | 2 | invalid option -- 'S' | DENY |
| `-U` | 0 (F) | grep `--binary`; multiline not applied | DENY (silently wrong) |
| `--no-ignore` | 0 (F) | abbreviation of grep `--no-ignore-case`; ignore files not affected | DENY (silently wrong) |
| `-uu`, `-uuu` | 0 (F) | grep warns "--unix-byte-offsets (-u) is obsolete"; not "unrestricted" | DENY (silently wrong) |
| `-P`, `--pcre2` | not probed | ambiguous (grep has `-P`) | excluded by spec |

Not denied but also observed (outside the rule, FYI): `rtk grep` handles only `-n` and the bare form with a directory target; any other
short flag with a directory path fell through to `/usr/bin/grep` without `-r` (`rtk grep -i|-l|-w|-C1 "def emit" tools/harness_hooks/` -> rc=2 "Is a directory";
`rtk grep -i "def emit"` with no path -> rc=1, no output). So `rg -i foo dir/` may still fail on this host; the rule does not cover it.
Known gap: typed `rg` with non-`-n` short flags plus a directory (`rg -i foo dir/`, also -l/-w/-c/-C) fails loudly with "Is a directory", and `-v` is silently wrong via `rtk grep`; the guard deliberately does not cover these. Use the Grep tool or `rtk grep`.

## Disable
`HARNESS_HOOKS=0` in the environment, or create `.claude/harness_hooks.disabled` (all hooks then print `{}`).

## Per-rule bypass (bash_guard)
Put the literal marker in the command, e.g. `uv run pytest tests/  # guard-ok:pytest-gpu`.
Markers: `guard-ok:rg`, `guard-ok:pytest-gpu`, `guard-ok:queue-dup`.
Test hook for the queue rule: `HARNESS_GUARD_FAKE_WORKERS=111,222` (empty string = no workers).

## Dry-run
```bash
echo '{"tool_name":"Bash","tool_input":{"command":"rg foo"}}' | python3 tools/harness_hooks/bash_guard.py   # deny (rg)
echo '{"tool_name":"Bash","tool_input":{"command":"python3 -c 1"}}' | python3 tools/harness_hooks/bash_guard.py   # {}
echo '{"hook_event_name":"SessionStart","source":"startup"}' | CLAUDE_PROJECT_DIR=$PWD python3 tools/harness_hooks/session_brief.py
echo '{"tool_input":{"file_path":"scripts/foo.sh"}}' | python3 tools/harness_hooks/sh_syntax_check.py
CUDA_VISIBLE_DEVICES="" uv run pytest tests/test_harness_hooks.py -q
```

## Registration
The hooks are registered in `.claude/settings.json` (since 2026-11-10; same entries as `settings_snippet.json`).
Each registered command has the form `f="$CLAUDE_PROJECT_DIR/tools/harness_hooks/<script>"; [ -f "$f" ] && [ -r "$f" ] || exit 0; python3 "$f"`:
in a checkout without the scripts (worktree etc.) the guard exits 0 instead of letting `python3` fail with exit 2, which would block every Bash call.
Disable with the switches above. A change to these files takes effect on the next hook call (each call is a new process).
