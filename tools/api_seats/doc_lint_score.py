"""Blind-grading kit and scorer for the doc_lint pilot.

  # 1) after doc_lint.py: anonymize model identities per label (A/B, random), write
  #    grading packets into RUN_DIR/grading/ and the key OUTSIDE the run dir
  uv run python -I tools/api_seats/doc_lint_score.py blind --run RUN_DIR --labels LABELS.yaml \
      --returns-dir RETURNS_DIR --key-out KEY.json --seed-file SEED.txt
  # 2) after the grader wrote RUN_DIR/grading/<label>.grades.yaml: score
  uv run --with pyyaml python -I tools/api_seats/doc_lint_score.py score --run RUN_DIR \
      --labels LABELS.yaml --key KEY.json --out METRICS.json

Metric definitions are the frozen ones in the pilot's 00_PREREG.md §4; this file
implements them and nothing else.
"""
from __future__ import annotations

import argparse, hashlib, itertools, json, random, secrets, sys
from collections import defaultdict
from pathlib import Path

import yaml

LABEL_TO_VERSION = {"015_v_r1": "EXP-015_RESULTS.v_r1.md", "015_v_r2": "EXP-015_RESULTS.v_r2.md",
                    "015_v_r3": "EXP-015_RESULTS.v_r3.md", "017_v_r1": "EXP-017_RESULTS.v_r1.md"}
RETURN_OF = {"015_v_r1": "EXP-015_RESULTS_VERIFY-r1.return.yaml", "015_v_r2": "EXP-015_RESULTS_VERIFY-r2.return.yaml",
             "015_v_r3": "EXP-015_RESULTS_VERIFY-r3.return.yaml", "017_v_r1": "EXP-017_RESULTS_VERIFY-r1.return.yaml"}
OUTCOMES = {"match", "new_true", "false_positive", "unclear"}


def load_labels(p: Path) -> dict:
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def check_returns(returns_dir: Path, pinned: dict) -> None:
    # verifier_original is the grading criterion, so every return must equal its pinned
    # hash before anything (seed, packets, key) is written.
    for label, name in RETURN_OF.items():
        got = hashlib.sha256((returns_dir / name).read_bytes()).hexdigest()
        if got != pinned[name]:
            sys.exit(f"{name}: sha256 {got} != pinned {pinned[name]}")


def verifier_text(returns_dir: Path, label: str) -> dict:
    d = yaml.safe_load((returns_dir / RETURN_OF[label]).read_text(encoding="utf-8"))
    out = {}
    for k in ("blocking", "non_blocking", "blocking_findings", "non_blocking_findings"):
        for it in d.get(k) or []:
            out[it["id"]] = {kk: it[kk] for kk in it if kk != "id"}
    return out


def cmd_blind(a) -> int:
    labels = load_labels(a.labels)
    check_returns(a.returns_dir, {b["return"]: b["return_sha256"] for b in labels["benchmark"]})
    seed = secrets.token_hex(16)
    a.seed_file.write_text(seed + "\n")
    rng = random.Random(seed)
    key, gdir = {}, a.run / "grading"
    gdir.mkdir(exist_ok=True)
    for label in LABEL_TO_VERSION:
        ldir = a.run / label
        if not ldir.exists():
            continue
        models = sorted(p.name for p in ldir.iterdir() if p.is_dir())
        letters = ["A", "B"][:len(models)]
        rng.shuffle(models)
        key[label] = dict(zip(letters, models))
        flags = []
        for letter, model in zip(letters, models):
            for parsed in sorted((ldir / model).glob("s*.parsed.json")):
                s = parsed.name.split(".")[0]
                d = json.loads(parsed.read_text(encoding="utf-8"))
                for i, f in enumerate(d["findings"], 1):
                    flags.append({"flag_id": f"{letter}.{s}.f{i}", **{k: f[k] for k in
                                  ("category", "severity", "location", "quote_source", "quote", "description", "suggested_fix")},
                                  "quote_exact_in_source": d["quote_checks"][i - 1]["exact"]})
                flags.append({"flag_id": f"{letter}.{s}.SAMPLE", "status": d["status"],
                              "n_findings": len(d["findings"])})
        vt = verifier_text(a.returns_dir, label)
        gt = [{"id": f["id"], "class": f["class"], "doc_partial": f["doc_partial"], "summary": f["summary"],
               "verifier_original": vt.get(f["id"].split("-", 1)[1])}
              for f in labels["findings"] if f["present_in"] == LABEL_TO_VERSION[label]]
        # The answer list is what the verifier found, not every defect; known residual
        # defects go to the grader so flags on them are graded new_true, not false_positive.
        gaps = [{k: g[k] for k in ("id", "location", "text", "defect")}
                for g in labels.get("known_gaps") or [] if LABEL_TO_VERSION[label] in g["present_in"]]
        packet = {"label": label, "version_file": f"benchmark/{LABEL_TO_VERSION[label]}",
                  "ground_truth_findings": gt, "residual_defects_not_in_answer_set": gaps, "flags": flags}
        (gdir / f"{label}.packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=1), encoding="utf-8")
        print(label, "flags", sum(1 for f in flags if "category" in f), "gt", len(gt))
    # The grader never sees which model is which: the key goes outside the run dir.
    a.key_out.write_text(json.dumps(key, indent=1))
    return 0


def jaccard(x: set, y: set) -> float:
    return 1.0 if not x and not y else len(x & y) / len(x | y)


def cmd_score(a) -> int:
    labels = load_labels(a.labels)
    key = json.loads(a.key.read_text())
    cls = {f["id"]: f for f in labels["findings"]}
    weak = set(labels.get("weak_salience") or [])
    per_model = defaultdict(lambda: {"hits_any": set(), "hits_maj": set(), "samples": 0, "failed": 0,
                                     "fp": 0, "new_true": 0, "unclear": 0, "flags": 0, "quote_exact": 0,
                                     "jacc": [], "doc_den": 0, "weak_den": 0, "by_class_any": defaultdict(set),
                                     "partial_any": set(), "blocking_doc_hit": {}})
    for label, letters in key.items():
        g = yaml.safe_load((a.run / "grading" / f"{label}.grades.yaml").read_text(encoding="utf-8"))
        grades = {r["flag_id"]: r for r in g["grades"]}
        packet = json.loads((a.run / "grading" / f"{label}.packet.json").read_text(encoding="utf-8"))
        gt_doc = {f["id"] for f in packet["ground_truth_findings"] if f["class"] == "DOC"}
        for letter, model in letters.items():
            m = per_model[model]
            m["doc_den"] += len(gt_doc)
            m["weak_den"] += len(gt_doc & weak)
            sample_sets = defaultdict(set)
            marks = [fl for fl in packet["flags"] if fl["flag_id"].startswith(letter + ".")
                     and fl["flag_id"].endswith(".SAMPLE")]
            sids = sorted(fl["flag_id"].split(".")[1] for fl in marks)
            m["failed"] += sum(fl["status"] != "ok" for fl in marks)
            for fl in packet["flags"]:
                if not fl["flag_id"].startswith(letter + ".") or fl["flag_id"].endswith(".SAMPLE"):
                    continue
                s = fl["flag_id"].split(".")[1]
                m["flags"] += 1
                m["quote_exact"] += bool(fl["quote_exact_in_source"])
                r = grades[fl["flag_id"]]
                assert r["outcome"] in OUTCOMES, r
                if r["outcome"] == "match":
                    for fid in r["matched_ids"]:
                        sample_sets[s].add(fid)
                elif r["outcome"] == "false_positive":
                    m["fp"] += 1
                elif r["outcome"] == "new_true":
                    m["new_true"] += 1
                else:
                    m["unclear"] += 1
            m["samples"] += len(sids)
            counts = defaultdict(int)
            for s in sids:
                for fid in sample_sets[s]:
                    counts[fid] += 1
            for fid, c in counts.items():
                c_cls = cls[fid]["class"]
                m["by_class_any"][c_cls].add(fid)
                if cls[fid]["doc_partial"]:
                    m["partial_any"].add(fid)
                if fid in gt_doc:
                    m["hits_any"].add(fid)
                    if c >= 2:
                        m["hits_maj"].add(fid)
            for f in gt_doc:
                if cls[f]["verifier_severity"] == "blocking":
                    m["blocking_doc_hit"][f] = counts.get(f, 0)
            # Stability over successful samples only; a pair where both samples hit no DOC
            # finding is not counted as agreement (it is excluded, not scored 1).
            ok_sids = [fl["flag_id"].split(".")[1] for fl in marks if fl["status"] == "ok"]
            docsets = [sample_sets[s] & gt_doc for s in ok_sids]
            m["jacc"] += [jaccard(x, y) for x, y in itertools.combinations(docsets, 2) if x | y]
    out = {}
    expected = len(LABEL_TO_VERSION) * a.n
    for model, m in per_model.items():
        ok = m["samples"] - m["failed"]
        missing = expected - m["samples"]
        out[model] = {
            "R_any": round(len(m["hits_any"]) / m["doc_den"], 4), "R_maj": round(len(m["hits_maj"]) / m["doc_den"], 4),
            # descriptive only (00_PREREG.md §4): R_any with the weak-salience DOC items left out
            "R_any_excl_weak": (round(len(m["hits_any"] - weak) / (m["doc_den"] - m["weak_den"]), 4)
                                if m["doc_den"] > m["weak_den"] else None),
            # integer counts: the section 5 comparisons are made on these, not on rounded ratios
            "doc_hits_any_count": len(m["hits_any"]), "doc_hits_maj_count": len(m["hits_maj"]),
            "doc_hits_any": sorted(m["hits_any"]), "doc_hits_maj": sorted(m["hits_maj"]), "doc_den": m["doc_den"],
            "blocking_doc_hit_count_of_n": m["blocking_doc_hit"],
            "FP_per_sample": round(m["fp"] / ok, 3) if ok else None,
            "new_true_per_sample": round(m["new_true"] / ok, 3) if ok else None,
            "unclear_total": m["unclear"], "flags_total": m["flags"],
            "G_quote_exact": round(m["quote_exact"] / m["flags"], 4) if m["flags"] else None,
            "J_mean": round(sum(m["jacc"]) / len(m["jacc"]), 4) if m["jacc"] else None,
            "samples": m["samples"], "failed_samples": m["failed"],
            # a call that wrote no parsed.json (skipped_budget / api_error / connection_error)
            # is a failed sample for the section 5 gate; its kind is in ledger.jsonl
            "missing_samples": missing, "failed_for_rule": m["failed"] + missing, "expected_samples": expected,
            "incidental_hits_any": {c: sorted(v) for c, v in m["by_class_any"].items() if c != "DOC"},
            "doc_partial_hits_any": sorted(m["partial_any"]),
        }
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("blind")
    b.add_argument("--run", type=Path, required=True); b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--returns-dir", type=Path, required=True); b.add_argument("--key-out", type=Path, required=True)
    b.add_argument("--seed-file", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True); s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--key", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n", type=int, default=3, help="samples per (version, model); must equal doc_lint.py --n")
    a = ap.parse_args()
    return cmd_blind(a) if a.cmd == "blind" else cmd_score(a)


if __name__ == "__main__":
    sys.exit(main())
