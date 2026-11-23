#!/usr/bin/env python3
"""AD-4: read-only, warning-only pre-enqueue guard for an experiment card.

Arg: EXP id (EXP-026) or a card path. Checks: (1) literature_link frontmatter (fallback:
a "## 문헌"/"literature" section containing a lab/28_literature path), (2) EXP-NNN_FREEZE.sha256
exists, (3) EXP-NNN_FREEZE_WITNESS.md exists, (4) `sha256sum -c --quiet` passes (run in the
experiments dir). Exit 0 always (WARN: lines on stderr); --strict exits 1 on any missing item.
Exit 2: card not found.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXPDIR = os.path.join(REPO, "lab/21_experiments/11_lab-experiments")
EXCLUDE = re.compile(r"_(RESULTS|WANDB_RUNS|FREEZE_WITNESS|RETROFIT|VERIFY)")


def find_card(arg, expdir):
    if os.path.isfile(arg):
        return os.path.realpath(arg)
    m = re.fullmatch(r"EXP-(\d+)", arg)
    if not m:
        return None
    cands = [p for p in sorted(glob.glob(os.path.join(expdir, f"EXP-{m.group(1)}_*.md")))
             if not EXCLUDE.search(os.path.basename(p))]
    return cands[0] if cands else None


def check_literature(text):
    fm = re.match(r"---\n(.*?)\n---", text, re.S)
    if fm:
        m = re.search(r"^literature_link:[ \t]*(.*)$", fm.group(1), re.M)
        if m and m.group(1).strip().strip("'\"") not in ("", "~", "null", "[]"):
            return True, "frontmatter literature_link"
    for sec in re.split(r"^(?=#{1,6} )", text, flags=re.M):
        head = sec.split("\n", 1)[0]
        if re.match(r"#{1,6} .*(문헌|literature)", head, re.I) and "lab/28_literature" in sec:
            return True, "literature section"
    return False, None


def run(arg, expdir=EXPDIR):
    card = find_card(arg, expdir)
    if card is None:
        return None
    base = os.path.basename(card)
    m = re.match(r"(EXP-\d+)_", base)
    exp = m.group(1) if m else arg
    d = os.path.dirname(card)
    text = open(card, encoding="utf-8", errors="replace").read()
    lit, via = check_literature(text)
    sha = os.path.join(d, f"{exp}_FREEZE.sha256")
    wit = os.path.join(d, f"{exp}_FREEZE_WITNESS.md")
    r = {"card": card, "exp": exp, "literature_link": lit, "literature_via": via,
         "freeze_sha256_exists": os.path.isfile(sha), "freeze_witness_exists": os.path.isfile(wit),
         "freeze_check": None}
    if r["freeze_sha256_exists"]:
        # manifest paths may be relative to the experiments dir or to the lab root two levels up
        rd = os.path.realpath(d)
        for cwd in (d, os.path.dirname(rd), os.path.dirname(os.path.dirname(rd))):
            p = subprocess.run(["sha256sum", "-c", "--quiet", sha], cwd=cwd,
                               capture_output=True, text=True)
            r["freeze_check_cwd"] = cwd
            if p.returncode == 0:
                break
        r["freeze_check"] = "pass" if p.returncode == 0 else "fail"
        if p.returncode:
            r["freeze_check_detail"] = (p.stdout + p.stderr).strip()[:500]
    r["warnings"] = [w for ok, w in [
        (lit, "no literature_link (frontmatter) or lab/28_literature path in a literature section"),
        (r["freeze_sha256_exists"], f"{exp}_FREEZE.sha256 missing"),
        (r["freeze_witness_exists"], f"{exp}_FREEZE_WITNESS.md missing"),
        (r["freeze_check"] != "fail", "sha256sum -c failed"),
    ] if not ok]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="EXP-NNN or card path")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any item missing")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expdir", default=EXPDIR)
    a = ap.parse_args(argv)
    r = run(a.target, a.expdir)
    if r is None:
        print(f"card not found: {a.target}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f'card: {r["card"]}')
        print(f'literature_link: {r["literature_link"]} (via {r["literature_via"]})')
        print(f'FREEZE.sha256 exists: {r["freeze_sha256_exists"]}')
        print(f'FREEZE_WITNESS.md exists: {r["freeze_witness_exists"]}')
        print(f'sha256sum -c: {r["freeze_check"]}')
    for w in r["warnings"]:
        print("WARN: " + w, file=sys.stderr)
    return 1 if (a.strict and r["warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
