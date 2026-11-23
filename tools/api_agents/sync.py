"""Control plane: push defs/ to the API project, record ids + versions in lock.json.

    with-anthropic-key uv run --no-sync --with anthropic==2.8.0 --with pyyaml python -m tools.api_agents.sync [--dry-run]
    with-anthropic-key ... python -m tools.api_agents.sync --retire --basis "<the user's request, quoted, with date>"

Idempotent. An agent whose resolved parameters hash equals the lock entry is left alone; a changed one is
updated in place (new version, same id), never re-created. A key missing from the lock is first looked up
remotely by metadata trm_key (adopted if found) before anything is created. Workers sync before the
coordinators that roster them, so a coordinator always pins the workers' current versions.

run.py start calls sync(..., worker_system=<text>, lead=<spec lead>) for a run whose spec inlines context: the
lead's rostered workers get that text appended to their system prompt (common.system_text), their lock entries
record its sha256, and the lead re-pins them. A later plain sync puts the workers back to their plain bodies.
"""
from __future__ import annotations

import argparse
import sys

from .common import (LOCK_PATH, METADATA_KEY, Halt, agent_params, builder, env_params, ledger, load_agent_defs,
                     load_env_defs, now, params_sha, read_lock, roster, scan_text, sha256_bytes, system_text,
                     worker_order, write_lock)


def _remote_by_key(items, key):
    for it in items:
        md = getattr(it, "metadata", None) or {}
        if md.get(METADATA_KEY) == key and not getattr(it, "archived_at", None):
            return it
    return None


def sync(client, dry_run: bool = False, lock_path=LOCK_PATH, log=print, worker_system: str | None = None,
         lead: str | None = None) -> dict:
    """worker_system (with lead): appended to the system prompt of every agent the lead rosters."""
    lock = read_lock(lock_path)
    adefs, edefs = load_agent_defs(), load_env_defs()
    extra: dict[str, str] = {}
    ws_sha = None
    if worker_system:
        if lead not in adefs or not roster(adefs[lead]):
            raise Halt(f"worker_system needs a coordinator with a roster; lead={lead!r}")
        extra = {k: worker_system for k in roster(adefs[lead])}
        ws_sha = sha256_bytes(worker_system.encode())
    b = builder()
    for k, d in adefs.items():                       # egress scan of every free-text field we are about to send
        scan_text(b, f"agent:{k}:system", system_text(d, extra.get(k)))
        scan_text(b, f"agent:{k}:description", str(d["description"]))
        scan_text(b, f"agent:{k}:name", str(d["name"]))
    for k, d in edefs.items():
        scan_text(b, f"env:{k}:name", str(d["name"]))
        scan_text(b, f"env:{k}:description", str(d.get("description") or ""))
    remote_agents = remote_envs = None
    changes = []

    for k, d in edefs.items():
        p = env_params(d)
        h = params_sha(p)
        ent = lock["environments"].get(k)
        if ent and ent.get("sha256") == h:
            log(f"env   {k}: unchanged ({ent['id']})"); continue
        if dry_run:
            log(f"env   {k}: would {'update' if ent else 'create'}"); changes.append(("env", k)); continue
        if ent is None:
            if remote_envs is None:
                remote_envs = list(client.beta.environments.list(limit=100))
            hit = _remote_by_key(remote_envs, k)
            ent = {"id": hit.id} if hit else None
        if ent:
            e = client.beta.environments.update(ent["id"], **p); action = "updated"
        else:
            e = client.beta.environments.create(**p); action = "created"
        lock["environments"][k] = {"id": e.id, "name": e.name, "sha256": h, "synced_at": now()}
        write_lock(lock, lock_path)
        ledger("env_" + action, key=k, id=e.id, sha256=h)
        log(f"env   {k}: {action} {e.id}"); changes.append(("env", k))

    for k in worker_order(adefs):
        try:
            p = agent_params(adefs[k], lock, extra.get(k))
        except Exception:
            if dry_run:
                log(f"agent {k}: would create (after its workers)"); changes.append(("agent", k)); continue
            raise
        h = params_sha(p)
        ent = lock["agents"].get(k)
        if ent and ent.get("sha256") == h:
            log(f"agent {k}: unchanged ({ent['id']} v{ent['version']})"); continue
        if dry_run:
            log(f"agent {k}: would {'update' if ent else 'create'}"); changes.append(("agent", k)); continue
        if ent is None:
            if remote_agents is None:
                remote_agents = list(client.beta.agents.list(limit=100))
            hit = _remote_by_key(remote_agents, k)
            ent = {"id": hit.id, "version": hit.version} if hit else None
        if ent:
            a = client.beta.agents.update(ent["id"], version=ent["version"], **p); action = "updated"
        else:
            a = client.beta.agents.create(**p); action = "created"
        tag = {"worker_system_sha256": ws_sha} if k in extra else {}
        lock["agents"][k] = {"id": a.id, "version": a.version, "name": a.name, "model": p["model"],
                             "sha256": h, "synced_at": now(), **tag}
        write_lock(lock, lock_path)
        ledger("agent_" + action, key=k, id=a.id, version=a.version, sha256=h, model=p["model"], **tag)
        log(f"agent {k}: {action} {a.id} v{a.version}"); changes.append(("agent", k))
    return {"changes": changes, "lock": lock}


def retire(client, basis: str, lock_path=LOCK_PATH, log=print) -> dict:
    """Archive every agent and environment in the lock, then empty the lock. Only on the user's request: `basis`
    is that request, quoted with its date, and goes to the ledger. Archived agents and environments stay
    readable (run dirs keep their ids as history) but cannot start new sessions; a later sync creates fresh
    ones. The lock entry of an object is removed only after its archive call returned."""
    if len((basis or "").strip()) < 10:
        raise Halt("retire: --basis must quote the user's request (at least 10 characters)")
    lock = read_lock(lock_path)
    previous = {"agents": dict(lock["agents"]), "environments": dict(lock["environments"])}
    done = []
    for k in sorted(lock["agents"]):
        r = client.beta.agents.archive(lock["agents"][k]["id"])
        ledger("agent_archived", key=k, id=r.id, archived_at=str(getattr(r, "archived_at", None)), basis=basis)
        log(f"agent {k}: archived {r.id}"); done.append(("agent", k))
        del lock["agents"][k]; write_lock(lock, lock_path)
    for k in sorted(lock["environments"]):
        r = client.beta.environments.archive(lock["environments"][k]["id"])
        ledger("env_archived", key=k, id=r.id, archived_at=str(getattr(r, "archived_at", None)), basis=basis)
        log(f"environment {k}: archived {r.id}"); done.append(("environment", k))
        del lock["environments"][k]; write_lock(lock, lock_path)
    ledger("lock_cleared", previous=previous, basis=basis)
    return {"archived": done, "lock": lock}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.api_agents.sync")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; no network writes")
    ap.add_argument("--retire", action="store_true",
                    help="archive every agent and environment in the lock and empty it (user request only)")
    ap.add_argument("--basis", help="with --retire: the user's request, quoted with its date")
    a = ap.parse_args(argv)
    import anthropic
    client = anthropic.Anthropic()
    if a.retire:
        r = retire(client, a.basis or "")
        print(f"{len(r['archived'])} archived")
        return 0
    r = sync(client, dry_run=a.dry_run)
    print(f"{len(r['changes'])} change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
