#!/usr/bin/env python3
"""AD-7: claim-ledger regeneration checker (DRY-RUN BY DEFAULT).

Parses the "Claim ledger" section of a RESULTS markdown file: shell variable
definitions in the preceding fenced block and the table column
"regeneration command" (or "재생성"). Default prints rows (class, expanded
command, expected-hash hint). `--execute` runs ONLY rows classed "pure".
Exit 0 ok, 1 mismatch / nonzero rc, 2 no ledger table found.
Classes: gpu | writes | pure | nocmd (no backtick command in the cell).
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CMD_PREFIXES = ("cd ", "uv ", "PYTHON", "sha256sum", "python")
GPU_RE = re.compile(r"CUDA_VISIBLE_DEVICES|nvidia|torch|measure_rho|pretrain\.py")
WRITE_WORDS = re.compile(r"덮어씀|overwrite|\btee\b|--save|--out\b", re.I)
HASH_HINT = re.compile(r"sha256(?![a-z])[\s:=`'\"~≈]*([0-9a-fA-F]{8,64})")
HEX64 = re.compile(r"\b[0-9a-fA-F]{64}\b")
VAR_DEF = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
VAR_REF = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)")
PIPE = re.compile(r"(?<!\\)\|")


def expand(text, env):
    for _ in range(10):
        new = VAR_REF.sub(lambda m: env.get(m.group(1) or m.group(2), m.group(0)), text)
        if new == text:
            break
        text = new
    return text


def split_row(line):
    cells = PIPE.split(line.strip())
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1].strip() == "":
        cells = cells[:-1]
    return [c.replace("\\|", "|").strip() for c in cells]


def pick_command(cell):
    spans = re.findall(r"`([^`\n]+)`", cell)
    if not spans:
        return None
    for s in spans:
        if s.startswith(CMD_PREFIXES):
            return s
    return max(spans, key=len)


def classify(cmd, row_text):
    if cmd is None:
        return "nocmd"
    if GPU_RE.search(cmd) or GPU_RE.search(row_text):
        return "gpu"
    redirect = re.sub(r"\d*>&\d*|&>|\d*>\s*/dev/null", "", cmd)
    if ">" in redirect or WRITE_WORDS.search(cmd) or WRITE_WORDS.search(row_text):
        return "writes"
    return "pure"


def parse_ledger(text):
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.lstrip().startswith("#") and "claim ledger" in l.lower()), None)
    if start is None:
        return None
    level = len(lines[start]) - len(lines[start].lstrip("#"))
    end = len(lines)
    fence = False
    for i in range(start + 1, len(lines)):
        if lines[i].lstrip().startswith("```"):
            fence = not fence
        if fence:
            continue
        m = re.match(r"(#+)\s", lines[i])
        if m and len(m.group(1)) <= level:
            end = i
            break
    sec = lines[start + 1:end]
    env, in_fence, hdr_i, col = {}, False, None, None
    for i, l in enumerate(sec):
        if l.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            m = VAR_DEF.match(l)
            if m and not l.lstrip().startswith("#"):
                v = m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                    v = v[1:-1]
                env[m.group(1)] = expand(v, env)
            continue
        if l.lstrip().startswith("|") and hdr_i is None:
            cells = split_row(l)
            for j, c in enumerate(cells):
                if "regeneration command" in c.lower() or "재생성" in c:
                    hdr_i, col = i, j
                    break
    if hdr_i is None:
        return None
    rows = []
    for l in sec[hdr_i + 2:]:
        if not l.lstrip().startswith("|"):
            break
        cells = split_row(l)
        if col >= len(cells):
            continue
        cmd = pick_command(cells[col])
        row_text = " | ".join(cells)
        h = HASH_HINT.search(row_text)
        rows.append({
            "row": cells[0] if cells else str(len(rows) + 1),
            "class": classify(cmd, row_text),
            "command": expand(cmd, env) if cmd else None,
            "expected_hash_prefix": h.group(1).lower() if h else None,
        })
    return {"env": env, "rows": rows}


def execute_row(r, env, timeout=600):
    renv = dict(os.environ)
    renv.update(env)
    try:
        p = subprocess.run(r["command"], shell=True, executable="/bin/bash", cwd=REPO, env=renv,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "rc": None}
    res = {"rc": p.returncode}
    # rc of the whole ledger command is informational for compound commands like
    # `...; echo rc=$?`; a nonzero final rc is a failure per contract.
    if p.returncode != 0:
        res["status"] = "rc_nonzero"
        return res
    exp = r["expected_hash_prefix"]
    toks = [t.lower() for t in HEX64.findall(p.stdout)]
    if exp and toks:
        res["status"] = "match" if any(t.startswith(exp) for t in toks) else "mismatch"
    else:
        res["status"] = "ran_no_hash"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_md")
    ap.add_argument("--dry-run", action="store_true", help="default; print only")
    ap.add_argument("--execute", action="store_true", help="run ONLY pure rows")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)
    try:
        text = open(a.results_md, encoding="utf-8").read()
    except OSError as e:
        print(f"unreadable: {e}", file=sys.stderr)
        return 2
    led = parse_ledger(text)
    if led is None or not led["rows"]:
        print("no claim-ledger table with a regeneration-command column found", file=sys.stderr)
        return 2
    execute = a.execute and not a.dry_run
    bad = False
    for r in led["rows"]:
        r["result"] = None
        if execute and r["class"] == "pure":
            r["result"] = execute_row(r, led["env"], a.timeout)
            if r["result"]["status"] in ("rc_nonzero", "mismatch", "timeout"):
                bad = True
    if a.json:
        print(json.dumps({"mode": "execute" if execute else "dry-run", **led}, ensure_ascii=False, indent=2))
    else:
        print(f'# mode={"execute" if execute else "dry-run"} rows={len(led["rows"])} vars={sorted(led["env"])}')
        for r in led["rows"]:
            cmd = (r["command"] or "").replace("\n", " ")
            cmd = cmd if len(cmd) <= 120 else cmd[:117] + "..."
            extra = f' hash~{r["expected_hash_prefix"]}' if r["expected_hash_prefix"] else ""
            res = f' => {r["result"]["status"]}' if r["result"] else ""
            print(f'{r["row"]}\t{r["class"]}\t{cmd}{extra}{res}')
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
