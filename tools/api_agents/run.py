"""Data plane: one Managed Agents session per run directory.

    PY="uv run --no-sync --with anthropic==2.8.0 --with pyyaml python -m tools.api_agents.run"
    $PY prepare <spec.yaml>                          # offline: scan + mount list + kickoff text -> run dir
    with-anthropic-key $PY start <run_dir> --auth "<the user's approval, quoted, with date>"
                                                     # [inline context: sync workers + lead], upload,
                                                     # create session (budget), follow events, collect
    with-anthropic-key $PY watch <run_dir>           # re-attach to the session's stream after a drop
    with-anthropic-key $PY collect <run_dir>         # session usage, threads, output files -> run dir
    with-anthropic-key $PY status <run_dir>
    $PY postcheck <run_dir>                          # offline: locate every quote of outputs/findings.jsonl
    $PY prepare <spec.yaml> --tag r2                 # a second run directory for the same spec on the same day
    $PY merge <run_dir> <run_dir> [...] --out <dir>  # offline: union of the runs' findings (same file list), then postcheck
    with-anthropic-key $PY cleanup <run_dir> [--files] [--session]   # only when the user asks

Run spec (YAML):
    name: <slug>                 task: <tasks/<task>.md stem>     lead: trm-lead      environment: trm-offline
    budget_usd: <session hard cap, list prices>                    deadline_min: <client wall clock>
    files: {list: <lab-relative list file>, select?: [lab-relative paths], limit?: N}
    context: [{path: <lab-relative>, mount: context/<name>, inline?: true}]
    vars: {NAME: value, ...}       # extra {{NAME}} placeholders for the task template
                                   # docaudit: SHARD_TOKENS, MAX_PARALLEL, MAX_FINDINGS_PER_FILE (suggested 5), RUN_NOTE

Built-in placeholders: DATE, N_FILES, EST_TOKENS, CORPUS_ROOT, FILES_LIST, OUTPUT_DIR, CONTEXT_FILE (the first
context entry), WORKER_BRIEFS (what a worker's brief carries besides its own paths; see worker_briefs).

Inline context (opt-in per context entry, `inline: true`): the task template must mark its rules block with the
lines WORKER_RULES_BEGIN / WORKER_RULES_END (HTML comments; stripped from the kickoff). prepare then writes
worker_system.md = the inline context files in spec order, each under a heading naming its mount path, then the
rules block with placeholders filled; it is scanned like any outgoing text and its sha256 goes into corpus.json.
start syncs the lead's rostered workers with that text appended to their system prompt, then the lead (which
pins the new worker versions), so every worker thread of the run shares one cached prefix. The file stays
mounted too (the lead reads it), and the kickoff tells the lead that briefs must not repeat it. Afterwards a run
without inline context needs a plain `sync` first (start refuses while the lock's workers carry shared text).

Mount layout in the container: /mnt/session/uploads/lab/<rel>, /mnt/session/uploads/context/<name>,
/mnt/session/uploads/context/files.txt (the mounted corpus list). Outputs: /mnt/session/outputs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import threading
import time
from pathlib import Path

import yaml

from .common import (LAB_ROOT, RUNS_DIR, TASKS_DIR, Halt, builder, canonical_rel, est_tokens, ledger, now,
                     read_lock, scan_lab_file, scan_text, sha256_bytes)
from .sync import sync as sync_defs

UPLOADS = "/mnt/session/uploads"
OUTPUTS = "/mnt/session/outputs"
MAX_MOUNTS = 900                     # API limit is 999 file resources per session
UPLOAD_TTL_S = 3 * 86400             # uploaded originals expire on their own after 3 days
MA_BETA = "managed-agents-2026-04-01"
WORKER_RULES_BEGIN = "<!-- WORKER_RULES_BEGIN -->"
WORKER_RULES_END = "<!-- WORKER_RULES_END -->"
# No standing authorization: every `start` must quote the user's own approval for that run (--auth). The
# 2026-11-12 approval covered only that day's test; its ledger rows carry a paraphrase of it.


# ------------------------------------------------------------------ prepare (offline)
def read_list(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def render(template: str, values: dict) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", str(v))
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", out)))
    if left:
        raise Halt(f"unfilled template placeholders: {left}")
    return out


def split_worker_rules(template: str) -> tuple[str, str | None]:
    """(template without the marker lines, the rules block between them or None when unmarked)."""
    lines = template.splitlines(keepends=True)
    b = [i for i, ln in enumerate(lines) if ln.strip() == WORKER_RULES_BEGIN]
    e = [i for i, ln in enumerate(lines) if ln.strip() == WORKER_RULES_END]
    if not b and not e and WORKER_RULES_BEGIN not in template and WORKER_RULES_END not in template:
        return template, None
    if len(b) != 1 or len(e) != 1 or b[0] > e[0] or \
            template.count(WORKER_RULES_BEGIN) != 1 or template.count(WORKER_RULES_END) != 1:
        raise Halt("task template: WORKER_RULES_BEGIN / _END must each appear once, on their own lines, BEGIN first")
    rules = "".join(lines[b[0] + 1:e[0]])
    if not rules.strip():
        raise Halt("task template: the marked rules block is empty")
    return "".join(ln for i, ln in enumerate(lines) if i not in (b[0], e[0])), rules


def worker_briefs(context: list[dict]) -> str:
    """{{WORKER_BRIEFS}}: what every worker brief carries besides the worker's own paths."""
    inl = [f"{UPLOADS}/{c['mount']}" for c in context if c.get("inline")]
    if not inl:
        if not context:
            return "Every brief also carries the rules block for every agent, word for word."
        return (f"Every brief also carries the path of the context file ({UPLOADS}/{context[0]['mount']}), which "
                "the worker reads first, and the rules block for every agent, word for word.")
    one = len(inl) == 1
    files = ("the context file " if one else "the context files ") + ", ".join(inl)
    out = (f"Every worker in this run already has {files} and the rules block for every agent in its system "
           f"prompt. Briefs must not repeat them and must not tell workers to read the context file{'' if one else 's'}. "
           "A brief carries only that worker's input paths, its output path and output format, the current date, and "
           "anything specific to its shard.")
    rest = [f"{UPLOADS}/{c['mount']}" for c in context if not c.get("inline")]
    if rest:
        out += (f" The other mounted context files ({', '.join(rest)}) are not in the system prompt; name one in a "
                "brief only when that worker needs it.")
    return out


def worker_system_text(inline: list[tuple[str, str]], rules: str) -> str:
    """The run's shared worker text: each inline context file (mount, text) under a heading naming its mount
    path, in spec order, then the rules block (placeholders already filled)."""
    parts = ["This part is the same for every worker in this run: the project context in full (each file under a "
             "heading naming the path where it is also mounted; you do not need to open it), then the rules every "
             "agent follows. Apply the rules as if your brief contained them."]
    for mount, text in inline:
        body = text.strip("\n")
        parts.append(f"# Context file: {UPLOADS}/{mount}\n\n{body}")      # top level: context files use ## inside
    rules_body = rules.strip("\n")
    parts.append(f"# Rules for every agent\n\n{rules_body}")
    return "\n\n".join(parts) + "\n"


NAME_RE = r"[a-z0-9][a-z0-9-]{2,60}"


def check_spec(spec: dict) -> None:
    for k in ("name", "task", "lead", "environment", "budget_usd", "deadline_min", "files"):
        if k not in spec:
            raise Halt(f"spec: missing {k!r}")
    if not isinstance(spec["name"], str) or not re.fullmatch(NAME_RE, spec["name"]):
        raise Halt("spec: name must be a lowercase slug")
    if float(spec["budget_usd"]) <= 0 or float(spec["deadline_min"]) <= 0:
        raise Halt("spec: budget_usd and deadline_min must be positive")


def prepare(spec_path: Path, runs_dir: Path = RUNS_DIR, today: str | None = None, tag: str | None = None) -> Path:
    raw_spec = Path(spec_path).read_bytes()                 # one read: parsed, copied and hash-pinned
    spec = yaml.safe_load(raw_spec.decode("utf-8"))
    check_spec(spec)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,60}", str(spec["task"])) or not (TASKS_DIR / f"{spec['task']}.md").is_file():
        raise Halt(f"spec: task must name a file tools/api_agents/tasks/<slug>.md ({spec['task']!r})")
    today = today or dt.date.today().isoformat()
    if tag is not None and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,11}", tag):
        raise Halt(f"prepare: --tag must be 1-12 characters of [a-z0-9-] ({tag!r})")
    rd = runs_dir / (f"{today}_{spec['name']}" + (f"-{tag}" if tag else ""))   # a tag lets one spec run twice a day
    if (rd / "session.json").exists():
        raise Halt(f"{rd} already has a session; pick a new name")
    fs = spec["files"]
    if not fs.get("select") and not canonical_rel(fs.get("list")):
        raise Halt("spec: files.list must be a canonical lab-relative path")
    rels = list(fs.get("select") or []) or read_list(LAB_ROOT / fs["list"])
    if fs.get("limit"):
        rels = rels[: int(fs["limit"])]
    b = builder()
    scan_text(b, "spec:name", spec["name"])                 # sent as the session title and metadata
    files, skipped, seen = [], [], set()
    for rel in rels:
        if rel in seen:
            continue
        seen.add(rel)
        ent, skip = scan_lab_file(b, rel)
        (files if ent else skipped).append(ent or skip)
    if not files:
        raise Halt("no file passed the scan")
    if len(files) + len(spec.get("context") or []) + 1 > MAX_MOUNTS:
        raise Halt(f"{len(files)} files exceed the mount cap {MAX_MOUNTS}")
    context, mounts, inline = [], set(), []
    for c in spec.get("context") or []:
        m = c.get("mount")
        if not isinstance(m, str) or not re.fullmatch(r"context/[A-Za-z0-9_-][A-Za-z0-9._-]*", m) or ".." in m \
                or m == "context/files.txt" or m in mounts:
            raise Halt(f"context mount {m!r} must be a unique context/<name> (not files.txt, no '..')")
        mounts.add(m)
        if not isinstance(c.get("inline", False), bool):
            raise Halt(f"context {m}: inline must be true or false")
        ent, skip = scan_lab_file(b, c.get("path"))           # same L1-L4 + IPv4 gate as corpus files
        if not ent:
            raise Halt(f"context file refused: {skip}")
        context.append({"rel": ent["rel"], "mount": m, "sha256": ent["sha256"], "inline": c.get("inline", False)})
        if context[-1]["inline"]:
            raw = (LAB_ROOT / ent["rel"]).read_bytes()
            if sha256_bytes(raw) != ent["sha256"]:
                raise Halt(f"context file {ent['rel']} changed during prepare")
            inline.append((m, raw.decode("utf-8")))
    template, rules = split_worker_rules((TASKS_DIR / f"{spec['task']}.md").read_text(encoding="utf-8"))
    if inline and rules is None:
        raise Halt("inline context needs a task template that marks its rules block (WORKER_RULES_BEGIN / _END)")
    listing = "".join(f"{UPLOADS}/lab/{f['rel']}\n" for f in files)
    briefs = worker_briefs(context)
    values = {"DATE": today, "N_FILES": len(files), "CORPUS_ROOT": f"{UPLOADS}/lab",
              "FILES_LIST": f"{UPLOADS}/context/files.txt", "OUTPUT_DIR": OUTPUTS,
              "CONTEXT_FILE": f"{UPLOADS}/{context[0]['mount']}" if context else "(none)",
              "EST_TOKENS": sum(f["est_tokens"] for f in files), "WORKER_BRIEFS": briefs, **(spec.get("vars") or {})}
    kickoff = render(template, values)
    worker_system = worker_system_text(inline, render(rules, values)) if inline else None
    if worker_system and briefs not in kickoff:
        raise Halt("inline context: the kickoff must carry {{WORKER_BRIEFS}} unchanged (add it to the task template)")
    scan_text(b, "kickoff", kickoff)
    scan_text(b, "files.txt", listing)
    if worker_system:
        scan_text(b, "worker_system", worker_system)          # leaves the machine inside the workers' system prompt
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "spec.yaml").write_bytes(raw_spec)
    (rd / "files.txt").write_text(listing, encoding="utf-8")
    (rd / "kickoff.md").write_text(kickoff, encoding="utf-8")
    if worker_system:
        (rd / "worker_system.md").write_text(worker_system, encoding="utf-8")
    elif (rd / "worker_system.md").exists():
        (rd / "worker_system.md").unlink()                    # left by an earlier prepare of this run dir
    corpus = {"prepared_at": now(), "deny_sha256": b.deny.sha256, "n_files": len(files), "n_skipped": len(skipped),
              "est_tokens": values["EST_TOKENS"], "files": files, "skipped": skipped, "context": context,
              "kickoff_sha256": sha256_bytes(kickoff.encode()), "files_txt_sha256": sha256_bytes(listing.encode()),
              "worker_system_sha256": sha256_bytes(worker_system.encode()) if worker_system else None,
              "spec_sha256": sha256_bytes(raw_spec)}
    (rd / "corpus.json").write_text(json.dumps(corpus, indent=1, ensure_ascii=False) + "\n")
    print(f"prepared {rd}: {len(files)} files (~{values['EST_TOKENS']} tokens), {len(skipped)} skipped")
    if worker_system:
        print(f"  worker_system.md: ~{est_tokens(worker_system)} tokens in every worker's system prompt")
    for s in skipped:
        print(f"  skip {s['rel']}: {s['layer']} {s['reason'][:100]}")
    return rd


# ------------------------------------------------------------------ helpers
def rel_run(rd: Path) -> str:
    try:
        return str(Path(rd).resolve().relative_to(LAB_ROOT.resolve()))
    except ValueError:
        return str(rd)


def load_json(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


def dump_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def as_dict(x) -> dict:
    for m in ("to_dict", "model_dump"):
        if hasattr(x, m):
            try:
                return getattr(x, m)()
            except TypeError:
                pass
    return dict(x)


def planned_items(rd: Path, corpus: dict) -> list[tuple[str, Path, str]]:
    """(mount, source, sha256 of the scanned bytes) for everything the session mounts, from corpus.json only."""
    for f in corpus["files"]:
        if not canonical_rel(f["rel"]):
            raise Halt(f"corpus.json: non-canonical path {f['rel']!r}")
    items = [(f"lab/{f['rel']}", LAB_ROOT / f["rel"], f["sha256"]) for f in corpus["files"]]
    items += [(c["mount"], LAB_ROOT / c["rel"], c["sha256"]) for c in corpus["context"]]
    items += [("context/files.txt", rd / "files.txt", corpus["files_txt_sha256"])]
    if len({m for m, _, _ in items}) != len(items):
        raise Halt("corpus.json: duplicate mount paths")
    return items


def load_pinned(rd: Path) -> tuple[dict, dict]:
    """(spec, corpus) with spec.yaml checked against the hash prepare pinned (title, budget cap, deadline,
    lead and environment all come from it)."""
    corpus = load_json(rd / "corpus.json")
    if not corpus or "spec_sha256" not in corpus:
        raise Halt("corpus.json has no spec_sha256 (missing, or prepared by an older version); run prepare again")
    raw = (rd / "spec.yaml").read_bytes()
    if sha256_bytes(raw) != corpus["spec_sha256"]:
        raise Halt("spec.yaml changed since prepare; run prepare again")
    spec = yaml.safe_load(raw.decode("utf-8"))
    check_spec(spec)
    return spec, corpus


def verified_kickoff(rd: Path, corpus: dict, b) -> str:
    """kickoff.md read once: hash-checked against prepare, scanned again, and these exact bytes are what is sent."""
    if b.deny.sha256 != corpus["deny_sha256"]:
        raise Halt("deny sources changed since prepare; run prepare again")
    kraw = (rd / "kickoff.md").read_bytes()
    if sha256_bytes(kraw) != corpus["kickoff_sha256"]:
        raise Halt("kickoff.md changed since prepare; run prepare again")
    kickoff = kraw.decode("utf-8")
    scan_text(b, "kickoff", kickoff)
    return kickoff


def _worker_system_raw(rd: Path, corpus: dict) -> bytes | None:
    sha = corpus.get("worker_system_sha256")
    if not sha:
        return None
    p = rd / "worker_system.md"
    raw = p.read_bytes() if p.is_file() else b""
    if sha256_bytes(raw) != sha:
        raise Halt("worker_system.md changed or missing since prepare; run prepare again")
    return raw


def verified_worker_system(rd: Path, corpus: dict, b) -> str | None:
    """worker_system.md read once (None when prepare built none): hash-checked, scanned again, and these exact
    bytes are what sync appends to the workers' system prompt."""
    raw = _worker_system_raw(rd, corpus)
    if raw is None:
        return None
    text = raw.decode("utf-8")
    scan_text(b, "worker_system", text)
    return text


def reverify(rd: Path) -> dict:
    """Early whole-set check (start re-checks each buffer again right before its upload)."""
    corpus = load_json(rd / "corpus.json")
    for mount, src, sha in planned_items(rd, corpus):
        if sha256_bytes(src.read_bytes()) != sha:
            raise Halt(f"{mount} changed since prepare; run prepare again")
    if sha256_bytes((rd / "kickoff.md").read_bytes()) != corpus["kickoff_sha256"]:
        raise Halt("kickoff.md changed since prepare; run prepare again")
    _worker_system_raw(rd, corpus)
    return corpus


def upload_all(client, rd: Path, corpus: dict) -> list[dict]:
    """Upload exactly the scanned bytes and return the session resources, built from corpus.json (never from
    whatever uploads.json holds). uploads.json makes a restart skip finished uploads, but only an entry whose
    recorded sha256 equals the scanned sha256 is reused; any mismatch halts."""
    up_path = rd / "uploads.json"
    ups = load_json(up_path, {})
    resources = []
    for mount, src, sha in planned_items(rd, corpus):
        if mount in ups:
            if ups[mount].get("sha256") != sha:
                raise Halt(f"uploads.json entry for {mount} does not match the scanned bytes; start a fresh run dir")
        else:
            raw = src.read_bytes()                              # one read: these bytes are hashed and sent
            if sha256_bytes(raw) != sha:
                raise Halt(f"{mount} changed since prepare; run prepare again")
            mime = "text/markdown" if src.suffix == ".md" else "text/plain"
            fobj = client.files.upload(file=(src.name, raw, mime), expires_in_seconds=UPLOAD_TTL_S)
            ups[mount] = {"file_id": fobj.id, "sha256": sha, "bytes": len(raw)}
            dump_json(up_path, ups)
        resources.append({"type": "file", "file_id": ups[mount]["file_id"], "mount_path": "/" + mount})
    return resources


class SendFailed(Exception):
    """A client event (deny) could not be sent; follow() drops the stream and answers again after replay."""


# ------------------------------------------------------------------ event state machine (pure)
class Follow:
    """Decides what to do with each event. Done = end_turn idle with no open workflow run, terminated,
    budget_reached, or retries_exhausted with no open run. Calls that pause for approval are denied
    (nobody watches an unattended run)."""

    def __init__(self):
        self.open_runs: dict[str, dict] = {}
        self.ended_runs: dict[str, dict] = {}
        self.threads: dict[str, dict] = {}
        self.denied: list[str] = []
        self.asks: dict[str, bool] = {}   # ask event id -> answered (a user.tool_confirmation for it was seen)
        self.done: str | None = None
        self.armed = False          # idles count only after the kickoff user.message (a fresh session starts idle)
        self.n = 0

    def unanswered(self) -> list[str]:
        return [i for i, ok in self.asks.items() if not ok]

    def on(self, d: dict) -> list[tuple]:
        self.n += 1
        t, acts = d.get("type", ""), []
        if t == "user.message":
            self.armed = True
        elif t == "user.tool_confirmation":
            self.asks[d.get("tool_use_id")] = True
        elif t == "workflow_run.created":
            self.open_runs[d["workflow_run_id"]] = {"name": d.get("name"), "phases": d.get("phases")}
        elif t == "workflow_run.status_ended":
            r = self.open_runs.pop(d.get("workflow_run_id"), {})
            self.ended_runs[d.get("workflow_run_id")] = {**r, "result": d.get("result")}
        elif t == "session.thread_created":
            self.threads[d.get("session_thread_id") or d.get("id")] = {"workflow_run_id": d.get("workflow_run_id")}
        elif t in ("agent.tool_use", "agent.mcp_tool_use") and d.get("evaluated_permission") == "ask":
            if d["id"] not in self.asks:
                self.asks[d["id"]] = False
                self.denied.append(d["id"])
                acts.append(("deny", d["id"]))
        elif t == "session.status_terminated":
            self.done = "terminated"
        elif t == "session.status_running" and self.done != "terminated":
            self.done = None                          # resumed (e.g. after a budget raise): not done after all
        elif t == "session.status_idle" and self.armed:
            sr = (d.get("stop_reason") or {}).get("type")
            if sr == "requires_action":
                pass                                  # waiting on us (asks are answered as they arrive)
            elif sr == "budget_reached":
                self.done = "budget_reached"
            elif not self.open_runs:                  # end_turn, retries_exhausted, refusal, or anything new
                self.done = sr or "idle"
        return acts


def short(d: dict) -> str | None:
    t = d.get("type", "")
    if t == "agent.message" and not d.get("session_thread_id"):
        txt = " ".join(b.get("text", "") for b in d.get("content") or [] if b.get("type") == "text")
        return f"lead: {txt[:300]!r}"
    if t.startswith("workflow_run.") or t in ("session.status_idle", "session.status_terminated", "session.error",
                                               "session.usage", "session.thread_created"):
        keep = {k: d.get(k) for k in ("workflow_run_id", "name", "result", "stop_reason", "error",
                                      "workflow_run_phase_id", "session_thread_id") if d.get(k) is not None}
        return f"{t} {json.dumps(keep, ensure_ascii=False, default=str)[:300]}"
    return None


def follow(client, rd: Path, sid: str, kickoff: str | None, deadline_ts: float) -> Follow:
    ev_path = rd / "events.jsonl"
    seen = set()
    if ev_path.exists():
        for line in ev_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    st = Follow()
    log = ev_path.open("a", encoding="utf-8")
    timed_out = threading.Event()

    denied_here: set[str] = set()                      # confirmed sent by this process (survives reconnects)

    def deny(eid):
        """Marked answered only after the send succeeds. A failed send raises SendFailed: the stream is dropped,
        and the reconnect replays history and answers the still-unanswered ask again."""
        if eid in denied_here:
            return
        try:
            client.beta.sessions.events.send(sid, events=[{
                "type": "user.tool_confirmation", "tool_use_id": eid, "result": "deny",
                "deny_message": "Unattended run: calls that need human approval are denied. Continue without it."}])
        except Exception as e:  # noqa: BLE001
            print(f"[{now()}] deny {eid} not sent: {e}", flush=True)
            raise SendFailed(e) from e
        denied_here.add(eid)

    def handle(d):
        if d.get("id") in seen:
            st.on(d)                                  # replayed history: rebuild state, answer nothing here
            return
        seen.add(d.get("id"))
        log.write(json.dumps({"recv": now(), **d}, ensure_ascii=False, default=str) + "\n"); log.flush()
        s = short(d)
        if s:
            print(f"[{now()}] {s}", flush=True)
        for act, eid in st.on(d):
            if act == "deny":
                deny(eid)

    def kickoff_pending() -> bool:
        return kickoff is not None and not (rd / "kickoff_sent").exists()

    connects = 0
    while st.done is None:
        left = deadline_ts - time.time()
        if left <= 0:
            st.done = "deadline"; break
        if connects:
            time.sleep(min(30, 2 * connects, left))    # back off between reconnects, never past the deadline
            if time.time() >= deadline_ts:
                st.done = "deadline"; break
        connects += 1
        try:
            with client.beta.sessions.events.stream(sid) as stream:
                timer = threading.Timer(max(1.0, deadline_ts - time.time()), lambda: (timed_out.set(), stream.close()))
                timer.daemon = True; timer.start()
                try:
                    st = Follow()                               # lossless (re)connect: history first, dedupe by id
                    for ev in client.beta.sessions.events.list(sid):
                        handle(as_dict(ev))
                    if kickoff_pending() and st.armed:          # an earlier send landed but was not recorded
                        (rd / "kickoff_sent").write_text(now() + " (found in history)\n")
                    for eid in st.unanswered():                 # an ask with no answer in history: answer it
                        deny(eid); st.asks[eid] = True
                    if st.done and st.done != "terminated" and \
                            getattr(client.beta.sessions.retrieve(sid), "status", None) == "running":
                        st.done = None                    # history ends on an idle the session has since left
                    if st.done:
                        break
                    if kickoff_pending():                       # marker only after a confirmed send
                        client.beta.sessions.events.send(sid, events=[
                            {"type": "user.message", "content": [{"type": "text", "text": kickoff}]}])
                        (rd / "kickoff_sent").write_text(now() + "\n")
                    for ev in stream:
                        handle(as_dict(ev))
                        if st.done:
                            break
                finally:
                    timer.cancel()
        except Exception as e:  # noqa: BLE001
            if timed_out.is_set():
                st.done = "deadline"; break
            code = getattr(getattr(e, "__cause__", None), "status_code", None) or getattr(e, "status_code", None)
            if isinstance(code, int) and 400 <= code < 500 and code not in (408, 409, 429):
                print(f"[{now()}] permanent client error {code} ({type(e).__name__}: {str(e)[:160]}); stopping",
                      flush=True)
                st.done = f"client_error_{code}"; break      # a 4xx will not succeed on retry
            print(f"[{now()}] stream dropped ({type(e).__name__}: {str(e)[:160]}); reconnecting", flush=True)
    log.close()
    return st


# ------------------------------------------------------------------ network commands
def start(client, rd: Path, auth: str) -> int:
    """`auth` is the user's approval for this metered run, quoted (written to the ledger, never sent)."""
    rd = Path(rd)
    if not isinstance(auth, str) or len(auth.strip()) < 10:
        raise Halt("start needs --auth: quote the user's approval for this run (with its date)")
    if (rd / "session.json").exists():
        raise Halt("session already created for this run; use watch (it also sends a kickoff that was never sent) "
                   "or collect")
    spec, _ = load_pinned(rd)
    lock = read_lock()
    lead = lock["agents"].get(spec["lead"]); env = lock["environments"].get(spec["environment"])
    if not lead or not env:
        raise Halt("lead agent or environment missing from lock.json; run tools.api_agents.sync first")
    corpus = reverify(rd)
    b = builder()
    kickoff = verified_kickoff(rd, corpus, b)
    worker_system = verified_worker_system(rd, corpus, b)
    scan_text(b, "spec:name", spec["name"])
    carried = sorted(k for k, v in lock["agents"].items() if v.get("worker_system_sha256"))
    if worker_system is None and carried:
        raise Halt(f"lock.json: {carried} still carry an earlier run's shared worker text; run tools.api_agents.sync "
                   "first so they go back to their plain system prompts")
    if not any(r.get("event") == "authorized" and r.get("run") == rel_run(rd) for r in _ledger_rows()):
        ledger("authorized", run=rel_run(rd), basis=auth.strip(), budget_usd=spec["budget_usd"])
    if worker_system is not None:                       # workers first, then the lead that pins their new versions
        lock = sync_defs(client, worker_system=worker_system, lead=spec["lead"],
                         log=lambda m: print(f"[{now()}] sync: {m}", flush=True))["lock"]
        lead = lock["agents"][spec["lead"]]; env = lock["environments"][spec["environment"]]
    resources = upload_all(client, rd, corpus)
    ups = load_json(rd / "uploads.json", {})
    ledger("uploaded", run=rel_run(rd), n_files=len(resources),
           bytes=sum(ups[r["mount_path"][1:]]["bytes"] for r in resources))
    cents = str(int(round(float(spec["budget_usd"]) * 100)))
    s = client.beta.sessions.create(
        agent={"type": "agent", "id": lead["id"], "version": lead["version"]},
        environment_id=env["id"], title=f"trm {spec['name']}", resources=resources,
        budget={"type": "limit", "max_list_cost": {"amount": cents, "currency": "USD"}},
        metadata={"run": spec["name"], "lane": "tools/api_agents"})
    dump_json(rd / "session.json", {"id": s.id, "created_at": now(), "agent": lead, "environment": env,
                                     "budget_cents": cents, "worker_system_sha256": corpus.get("worker_system_sha256")})
    ledger("session_created", run=rel_run(rd), session_id=s.id, agent_id=lead["id"], agent_version=lead["version"],
           budget_cents=cents)
    print(f"session {s.id}  https://platform.claude.com/workspaces/default/sessions/{s.id}", flush=True)
    deadline = time.time() + 60 * float(spec["deadline_min"])
    st = follow(client, rd, s.id, kickoff, deadline)
    return _finish(client, rd, s.id, st)


def watch(client, rd: Path) -> int:
    """Re-attach after a drop. If the kickoff was never recorded as sent, follow() replays history first and
    sends it only when history shows no user.message."""
    rd = Path(rd)
    spec, corpus = load_pinned(rd)
    sid = load_json(rd / "session.json")["id"]
    kickoff = None if (rd / "kickoff_sent").exists() else verified_kickoff(rd, corpus, builder())
    deadline = time.time() + 60 * float(spec["deadline_min"])
    st = follow(client, rd, sid, kickoff, deadline)
    return _finish(client, rd, sid, st)


def _finish(client, rd, sid, st: Follow) -> int:
    dump_json(rd / "follow.json", {"done": st.done, "events": st.n, "open_runs": st.open_runs,
                                   "ended_runs": st.ended_runs, "denied": st.denied, "at": now()})
    print(f"[{now()}] follow ended: {st.done}; runs ended {len(st.ended_runs)}, open {len(st.open_runs)}", flush=True)
    collect(client, rd)
    return 0 if st.done == "end_turn" else 6


def _ledger_rows():
    from . import common
    p = common.LEDGER_PATH
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def collect(client, rd: Path) -> dict:
    rd = Path(rd)
    sid = load_json(rd / "session.json")["id"]
    s = as_dict(client.beta.sessions.retrieve(sid))
    dump_json(rd / "session_final.json", s)
    threads = [as_dict(t) for t in client.beta.sessions.threads.list(sid)]
    dump_json(rd / "threads.json", threads)
    lock = read_lock()
    worker_ids = {v["id"]: k for k, v in lock["agents"].items()}
    run_threads = [t for t in threads if t.get("workflow_run_id")]
    by_agent: dict[str, int] = {}
    inline = 0
    for t in threads:
        a = t.get("agent") or {}
        if a.get("type") == "inline":
            inline += 1
        key = worker_ids.get(a.get("id")) or a.get("name") or a.get("type") or "?"   # snapshot name if not in lock
        model = a.get("model")
        model = model.get("id") if isinstance(model, dict) else model
        tag = f"{key}@v{a.get('version')}|{model}|{'run' if t.get('workflow_run_id') else 'thread'}"
        by_agent[tag] = by_agent.get(tag, 0) + 1
    resource_ids = {r.get("file_id") for r in (s.get("resources") or [])}
    out_dir = rd / "outputs"; out_dir.mkdir(exist_ok=True)
    got = []
    for attempt in range(4):
        files = [f for f in client.beta.files.list(scope_id=sid, betas=[MA_BETA]) if f.id not in resource_ids]
        if files:
            break
        time.sleep(3)
    for f in files:
        name = re.sub(r"[^A-Za-z0-9._-]", "_", f.filename) or f.id
        client.files.download(f.id).write_to_file(out_dir / name)
        got.append({"file_id": f.id, "filename": f.filename, "saved": name, "bytes": getattr(f, "size_bytes", None)})
    usage = s.get("usage") or {}
    summary = {"session_id": sid, "status": s.get("status"), "list_cost": usage.get("list_cost"),
               "usage": usage, "stats": s.get("stats"), "n_threads": len(threads), "n_run_threads": len(run_threads),
               "inline_threads": inline, "threads_by_agent": by_agent, "outputs": got, "collected_at": now()}
    dump_json(rd / "summary.json", summary)
    ledger("collected", run=rel_run(rd), session_id=sid, status=s.get("status"), list_cost=usage.get("list_cost"),
           n_threads=len(threads), n_run_threads=len(run_threads), inline_threads=inline, n_outputs=len(got))
    print(json.dumps({k: summary[k] for k in ("status", "list_cost", "n_threads", "n_run_threads", "inline_threads",
                                              "threads_by_agent")}, ensure_ascii=False, default=str, indent=1))
    print("outputs:", [g["saved"] for g in got])
    return summary


def thread_events(client, rd: Path) -> dict:
    """Save every workflow thread's own event list to <run_dir>/thread_events/<thread_id>.jsonl. Read-only and not
    model inference. For a run that stopped before the lead combined the results (2026-11-17: the workspace usage
    limit was reached in the check phase): the workers' result files exist only in the session's container, but
    each worker's `write` call, with the file's content, is in its thread's events."""
    rd = Path(rd)
    sid = load_json(rd / "session.json")["id"]
    out = rd / "thread_events"; out.mkdir(exist_ok=True)
    n = {}
    for t in load_json(rd / "threads.json") or []:
        if not t.get("workflow_run_id"):
            continue
        rows = [as_dict(e) for e in client.beta.sessions.threads.events.list(t["id"], session_id=sid, betas=[MA_BETA])]
        with (out / f"{t['id']}.jsonl").open("w", encoding="utf-8") as fh:
            for e in rows:
                fh.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
        n[t["id"]] = len(rows)
    ledger("thread_events", run=rel_run(rd), session_id=sid, n_threads=len(n), n_events=sum(n.values()))
    print(json.dumps({"threads": len(n), "events": sum(n.values())}))
    return n


def set_budget(client, rd: Path, usd: float) -> None:
    """Change the session's cap (must exceed the consumed list cost). A session paused at budget_reached resumes."""
    rd = Path(rd)
    sid = load_json(rd / "session.json")["id"]
    cents = str(int(round(usd * 100)))
    s = client.beta.sessions.update(sid, budget={"type": "limit", "max_list_cost": {"amount": cents, "currency": "USD"}})
    ledger("budget_changed", run=rel_run(rd), session_id=sid, budget_cents=cents,
           consumed=(as_dict(s).get("usage") or {}).get("list_cost"))
    print(f"budget -> {cents} cents; status {getattr(s, 'status', None)}")


def status(client, rd: Path) -> None:
    sid = load_json(Path(rd) / "session.json")["id"]
    s = as_dict(client.beta.sessions.retrieve(sid))
    print(json.dumps({"status": s.get("status"), "list_cost": (s.get("usage") or {}).get("list_cost"),
                      "budget": s.get("budget")}, default=str))


def cleanup(client, rd: Path, files: bool, session: bool) -> None:
    rd = Path(rd)
    if files:
        ups = load_json(rd / "uploads.json", {})
        n = 0
        for m, u in ups.items():
            try:
                client.files.delete(u["file_id"]); n += 1
            except Exception as e:  # noqa: BLE001
                print(f"delete {m} ({u['file_id']}): {e}")
        ledger("files_deleted", run=rel_run(rd), n=n)
        print(f"deleted {n} uploaded files")
    if session:
        sid = load_json(rd / "session.json")["id"]
        client.beta.sessions.delete(sid)
        ledger("session_deleted", run=rel_run(rd), session_id=sid)
        print(f"deleted session {sid}")


# ------------------------------------------------------------------ postcheck (offline)
def postcheck(rd: Path) -> dict:
    from tools.api_batch.docaudit import locate
    rd = Path(rd)
    fp = rd / "outputs" / "findings.jsonl"
    rows = [json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
    prefix = f"{UPLOADS}/lab/"

    def norm(p):
        p = (p or "").strip()
        return p[len(prefix):] if p.startswith(prefix) else p.lstrip("/")

    cache: dict[str, str | None] = {}

    def text(rel):
        if rel not in cache:
            p = LAB_ROOT / rel
            cache[rel] = p.read_text(encoding="utf-8") if rel and p.is_file() else None
        return cache[rel]

    corpus = load_json(rd / "corpus.json")
    mounted = {f["rel"] for f in corpus["files"]}
    ctx = {c["mount"]: c["rel"] for c in corpus.get("context") or []}   # shared context is mounted outside lab/
    res, counts = [], {"n": len(rows), "quote_found": 0, "quote_not_found": 0, "file_not_mounted": 0,
                       "counter_checked": 0, "counter_found": 0}
    for r in rows:
        rel = norm(r.get("file"))
        t = text(rel)
        line, how = locate(t, r.get("quote", "")) if t is not None else (0, "no-file")
        if rel not in mounted:
            counts["file_not_mounted"] += 1
        counts["quote_found" if line else "quote_not_found"] += 1
        cline = chow = None
        if r.get("counter_quote") and r.get("counter_file"):
            crel = norm(r["counter_file"])
            crel = ctx.get(crel[len(UPLOADS):].lstrip("/") if crel.startswith(UPLOADS.lstrip("/")) else crel, crel)
            ct = text(crel)
            counts["counter_checked"] += 1
            cline, chow = locate(ct, r["counter_quote"]) if ct is not None else (0, "no-file")
            counts["counter_found"] += bool(cline)
        res.append({"file": rel, "line": line, "how": how, "counter_line": cline, "counter_how": chow})
    out = {"counts": counts, "rows": res, "at": now()}
    dump_json(rd / "postcheck.json", out)
    print(json.dumps(counts))
    return out


_SEVERITY = {"high": 3, "medium": 2, "low": 1}


def _same_finding(m: dict, rel: str, line: int, quote: str) -> bool:
    """Two findings are one when they are in the same file and quote the same located line, or one quote contains
    the other, or the quotes are nearly identical. Deliberately strict: an unmerged duplicate costs the reader a
    moment, a wrong merge hides a finding."""
    if m["rel"] != rel:
        return False
    if line and m["line"] and line == m["line"]:
        return True
    a, b = (m["finding"].get("quote") or "").strip(), (quote or "").strip()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def merge(rds: list[Path], out: Path) -> dict:
    """Offline union of the kept findings of several runs over the SAME file list (the C1 configuration finds a
    different subset on each run; RECORD_2026-11-17.md sec. 8). Writes <out>/outputs/findings.jsonl (each finding
    plus `found_in` and `n_runs`), <out>/corpus.json, <out>/merge.json, then runs postcheck on the result."""
    from tools.api_batch.docaudit import locate
    rds, out = [Path(r) for r in rds], Path(out)
    if len(rds) < 2 or len({r.resolve() for r in rds}) != len(rds):
        raise Halt("merge: give at least two different run directories")
    corpora = [load_json(rd / "corpus.json") for rd in rds]
    if any(c is None for c in corpora) or len({c.get("files_txt_sha256") for c in corpora}) != 1:
        raise Halt("merge: the runs do not cover the same file list (corpus.json files_txt_sha256 differs or is missing)")
    if (out / "outputs" / "findings.jsonl").exists():
        raise Halt(f"merge: {out} already holds merged findings")
    prefix = f"{UPLOADS}/lab/"

    def norm(p):
        p = (p or "").strip()
        return p[len(prefix):] if p.startswith(prefix) else p.lstrip("/")

    cache: dict[str, str | None] = {}

    def text(rel):
        if rel not in cache:
            p = LAB_ROOT / rel
            cache[rel] = p.read_text(encoding="utf-8") if rel and p.is_file() else None
        return cache[rel]

    merged, sources = [], []
    for rd in rds:
        fp = rd / "outputs" / "findings.jsonl"
        if not fp.is_file():
            raise Halt(f"merge: {fp} is missing (collect the run first)")
        rows = [json.loads(x) for x in fp.read_text(encoding="utf-8").splitlines() if x.strip()]
        cost = ((load_json(rd / "summary.json", {}) or {}).get("list_cost") or {}).get("amount")
        sources.append({"run": rd.name, "n_findings": len(rows), "list_cost_cents": int(cost) if cost else None})
        for r in rows:
            rel = norm(r.get("file"))
            t = text(rel)
            line = locate(t, r.get("quote", ""))[0] if t is not None else 0
            hit = next((m for m in merged if _same_finding(m, rel, line, r.get("quote", ""))), None)
            if hit is None:
                merged.append({"rel": rel, "line": line, "finding": r, "runs": [rd.name]})
            elif rd.name not in hit["runs"]:
                hit["runs"].append(rd.name)
                if _SEVERITY.get(r.get("severity"), 0) > _SEVERITY.get(hit["finding"].get("severity"), 0):
                    hit["finding"] = r                       # keep the statement that rated it higher
    merged.sort(key=lambda m: (m["rel"], m["line"]))
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    with (out / "outputs" / "findings.jsonl").open("w", encoding="utf-8") as fh:
        for m in merged:
            fh.write(json.dumps({**m["finding"], "found_in": m["runs"], "n_runs": len(m["runs"])},
                                ensure_ascii=False) + "\n")
    dump_json(out / "corpus.json", {"files": corpora[0]["files"], "files_txt_sha256": corpora[0]["files_txt_sha256"],
                                    "context": corpora[0].get("context") or [],
                                    "merged_from": [rd.name for rd in rds]})
    costs = [s["list_cost_cents"] for s in sources]
    info = {"sources": sources, "n_merged": len(merged),
            "n_in_every_run": sum(len(m["runs"]) == len(rds) for m in merged),
            "n_in_one_run": sum(len(m["runs"]) == 1 for m in merged),
            "list_cost_cents_total": sum(costs) if all(c is not None for c in costs) else None, "at": now()}
    dump_json(out / "merge.json", info)
    print(json.dumps({k: v for k, v in info.items() if k != "sources"}))
    postcheck(out)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.api_agents.run")
    sp = ap.add_subparsers(dest="cmd", required=True)
    pr = sp.add_parser("prepare"); pr.add_argument("spec")
    pr.add_argument("--tag", help="suffix for the run directory, so one spec can be prepared twice on a day (e.g. r2)")
    mg = sp.add_parser("merge"); mg.add_argument("run_dirs", nargs="+"); mg.add_argument("--out", required=True)
    st_ = sp.add_parser("start"); st_.add_argument("run_dir")
    st_.add_argument("--auth", required=True, help="the user's approval for this metered run, quoted, with date")
    for c in ("watch", "collect", "status", "postcheck", "thread-events"):
        sp.add_parser(c).add_argument("run_dir")
    bu = sp.add_parser("budget"); bu.add_argument("run_dir"); bu.add_argument("--usd", type=float, required=True)
    cl = sp.add_parser("cleanup"); cl.add_argument("run_dir")
    cl.add_argument("--files", action="store_true"); cl.add_argument("--session", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(Path(a.spec), tag=a.tag); return 0
    if a.cmd == "merge":
        merge([Path(x) for x in a.run_dirs], Path(a.out)); return 0
    if a.cmd == "postcheck":
        postcheck(Path(a.run_dir)); return 0
    import anthropic
    client = anthropic.Anthropic()
    if a.cmd == "start":
        return start(client, Path(a.run_dir), a.auth)
    if a.cmd == "watch":
        return watch(client, Path(a.run_dir))
    if a.cmd == "collect":
        collect(client, Path(a.run_dir)); return 0
    if a.cmd == "status":
        status(client, Path(a.run_dir)); return 0
    if a.cmd == "thread-events":
        thread_events(client, Path(a.run_dir)); return 0
    if a.cmd == "budget":
        set_budget(client, Path(a.run_dir), a.usd); return 0
    if a.cmd == "cleanup":
        cleanup(client, Path(a.run_dir), a.files, a.session); return 0
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Halt as e:
        print(f"HALT: {e}", file=sys.stderr)
        sys.exit(3)
