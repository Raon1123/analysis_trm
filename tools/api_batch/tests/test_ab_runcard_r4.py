"""r4: structural post-check classes (character allow-list, number words / glued suffixes, hypothesis talk, run-status
vocabulary, EXP/H ids only in see_also, fixed attribution sentence), the verifier's adversarial variants
(17_STAGE_B_VERIFY-r2.return.yaml: B1/B2/B3/B4/multi/status/number/invisible classes, rebuilt on synthetic fixture cards),
honest outputs and an honest-sentence corpus, the gate's fail-open fix and the script-rendered attribution."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runcard_fixture import good_output, make_card  # noqa: E402
from tools.api_batch.common import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tools/wandb_run_index"))
import pilot_gate as PG  # noqa: E402
import prose_check as PC  # noqa: E402
import prose_inject as PI  # noqa: E402

KNOWN = ({"aaaa1111", "bbbb2222", "abc12345", "4s34eck3", "abcdefgh", "zabrylpt"},
         {"a_k4_s1", "demo_k4_s1", "other_run_alpha", "TRM_halt_one"})

# card kind -> (make_card kwargs, good_output kwargs)
KINDS = {
    "plain": ({}, {}),
    "nosum": ({"status": "no-summary"}, {"status": "no-summary", "extra_caveats": ("status",)}),
    "trunc": ({"truncated": True}, {"truncated": True, "extra_caveats": ("truncated_history",)}),
    "snap": ({"snapshot": True, "status": "running?"}, {"status": "running?", "snapshot": True, "extra_caveats": ("status", "snapshot")}),
    "multi": ({"multi": True}, {"multi": True, "extra_caveats": ("multi_attribution",)}),
}
FIELDS = {"ws": "what_this_run_is", "ns": "numbers_summary", "at": "attribution"}


def run(kind, field, add=None, replace=None):
    ck, ok = KINDS[kind]
    o = good_output(**ok)
    if field == "cv":
        o["caveats"].append({"kind": "other", "text": add})
    elif field == "see":
        o["see_also"] = [add]
    else:
        f = FIELDS[field]
        o[f] = replace if replace is not None else o[f] + " " + add
    return PC.check_output(make_card(**ck), o, KNOWN)


def test_baselines_pass_for_every_kind():
    for k, (ck, ok) in KINDS.items():
        assert PC.check_output(make_card(**ck), good_output(**ok), KNOWN) == [], k


# ---- rebuilt verifier variants: (kind, field, text). EVERY one must be rejected.
B1 = [("plain", "ws", s) for s in (
    "Related to run 4s34eck3.", "Related to ZABRYLPT.", "Related to zabrylpt.", "See runs/zabrylpt/files.", "See `zabrylpt`.", "See run-zabrylpt.",
    "See zabrylpt_s1.", "See id:zabrylpt.", "See zabrylpt.md.", "Config at runs/4s34eck3/files.", "Related `4s34eck3`.", "Run 4S34ECK3 is related.",
    "Similar to TRM_halt_one.", "Similar to trm_halt_one.", "`TRM_halt_one` is a sibling.", "run-4s34eck3 mentioned", "Run 4s34еck3 related.",
    "Run 4s34e​ck3 related.", "Run 4s34eck3ab exists.", "Run abcdefgh exists.", "Sibling a_k4_s1.", "Sibling A_K4_S1.", "Run zx9y8w7v exists.")]
B2 = [("nosum", "ns", s) for s in (
    "The run was a success.", "This run succeeds at the task.", "The model learned the permutation.", "It is robust to seed.", "The runs agree with each other.",
    "This corroborates the earlier finding.", "The result is inconclusive.", "The training was unstable.", "The model clearly solves the task.",
    "This outperforms its sibling.", "It fully solves k6.", "The approach works.", "The trend is encouraging.", "This favors the hypothesis.",
    "This favours the idea.", "Data point against the claim.", "This is in line with H-023.", "The run is healthy.", "Performance is excellent.",
    "It is superior.", "The model memorised the data.", "It overfits.", "A negative result.", "a clean win", "The hypothesis holds.",
    "Results are compatible with the theory.", "This bolsters the case", "This lends credence", "Training diverged.", "Training was successful.",
    "The run proves it.", "Evidence-based", "supports that", "SUPPORTS", "un-supported", "loss improves", "non-improving", "The run has failed to learn",
    "Fail.", "The experiment validates", "validated", "fully-converged", "unconverged", "unlearned", "CoNfIrMs the idea",
    "This is consistent\nwith H-023.", "Das Ergebnis bestaetigt die Hypothese", "Le resultat prouve l hypothese", "Die Hypothese stimmt",
    "The result refutes it.", "A strong result.", "A weak result.", "strong evidence", "weak evidence", "It contradicts the earlier run.",
    "The theory predicts this.", "We expect this.", "A conjecture holds.", "The claim is plausible.", "It is stable.")]
B2_NONASCII = [("nosum", "ns", s) for s in (
    "이 결과는 가설을 입증한다.", "この結果は仮説を支持する。", "Это доказывает гипотезу.",
    "结果证明了假设。", "Das Ergebnis bestätigt.", "Le résultat prouve l'hypothèse.", "The run is confіrmed",
    "The run is sup​ported", "ｓｕｃｃｅｓｓ", "The model suc­ceeds", "prov​een", "s u c c e s s",
    "‮", "✓ done", "→ rises", "≈ 0.5", "Δ is large", "− sign", "\U0001f600", "step ２６４", "step ٢٦٤",
    "fin​al values", "These are the ﬁnal values.", "The run was compl​eted.")]
B3 = [("trunc", f, s) for f, s in (
    ("ws", "The table's last row is where the run ended up."), ("ws", "The last row is where training stopped."),
    ("ns", "These are the values at the close of training."), ("ns", "The run ends there."), ("ns", "The concluding value is shown."),
    ("ns", "This was the terminal value."), ("ws", "The run is over."), ("cv", "The reported last values reflect the end state of the run."),
    ("ns", "The final value at the end of the run was reported."), ("ns", "FiNaL values"), ("ns", "Though these are effectively the end values."),
    ("ws", "Nevertheless the run wrapped up there."), ("ns", "The run finished."), ("ns", "Training completed."), ("ns", "The loss converged."))]
B4 = [("snap", "ws", s) for s in (
    "The run is over.", "Training wrapped up.", "The run has stopped logging.", "The run terminated normally.", "The run concluded.", "The run is done.",
    "The run exited cleanly.", "The run reached its end.", "The training has wrapped up.", "The run finished and completed training.",
    "These are the final values.", "Training ended.", "The run converged.", "The run is not ended.")]
MULTI = [("multi", f, s) for f, s in (
    ("ws", "The run belongs to EXP-015."), ("ws", "This is an EXP-015 run."), ("ws", "It sits under EXP-015."), ("ns", "EXP-015 is the correct experiment."),
    ("ws", "The experiment is EXP-015."), ("ws", "This run is part of EXP-015."), ("ws", "This run was made for EXP-015."),
    ("cv", "EXP-015 is listed first, which is the main one."), ("cv", "EXP-015 fits best."), ("ws", "It is one of the EXP-015 runs."),
    ("cv", "The run really pertains to EXP-015."), ("ws", "Only EXP-015 applies."), ("ws", "It belongs to exp015."), ("ws", "Linked to EXP015."),
    ("ws", "Linked to H-023."), ("ws", "Linked to H023."), ("cv", "See h-4."))]
STATUS = [(k, f, s) for k, f, s in (
    ("plain", "ws", "The run crashed."), ("plain", "ws", "It was killed early."), ("plain", "ws", "The run is still running."),
    ("nosum", "ws", "The run completed."), ("nosum", "ws", "The run finished normally."), ("nosum", "ws", "The run ended cleanly."),
    ("plain", "ws", "The run aborted."), ("plain", "ws", "The run stalled."), ("plain", "ws", "The run hung."), ("plain", "ws", "The run is ongoing."),
    ("plain", "ws", "The run is in progress."), ("plain", "ws", "The run succeeded."), ("plain", "ws", "The run is done."),
    ("plain", "ws", "The run terminated."), ("plain", "ws", "The run diverged."), ("plain", "ws", "The run exited."), ("plain", "ws", "The run is over."),
    ("plain", "cv", "The run finished fine."), ("plain", "ns", "It ended at step 244100."))]
NUM = [("trunc", "ns", s) for s in (
    "The loss is two point zero nine.", "There are twelve rows.", "The 3rd row.", "about a thousand steps", "reached 10k steps", "reached 244k steps.",
    "1e3", "-0.5", "step 3.916", "The value is 33%.", "about 2.", "roughly 3.", "approx 1.", "a 3x change", "10M steps", "half of 244100", "twice 4882",
    "It doubled.", "one run", "step zero", "a dozen rows", "hundreds of steps", "once at step 4882", "1/2 of the rows", "a 5-fold change")]
INVIS = [("plain", "ns", s) for s in ("<script>x</script>", "<b>b</b>", "[x](y)", "![i](http://x)", "```", "&amp; more", "&#x41;", "a > b", "a < b")]
URLS = [("nosum", "ws", s) for s in ("See https://wandb.ai/x/y.", "[run](https://wandb.ai/x)", "ftp://x.y/z", "See wandb.ai/raon1123/Sigma_k/runs/abc")]
SEE = [("multi", "see", s) for s in ("EXP-099_other.md", "4s34eck3", "see here please", "https://wandb.ai/x", "a" * 130)]

ALL = B1 + B2 + B2_NONASCII + B3 + B4 + MULTI + STATUS + NUM + INVIS + URLS + SEE


def test_variant_count_is_large():
    assert len(ALL) >= 150, len(ALL)


@pytest.mark.parametrize("kind,field,text", ALL, ids=[f"{i:03d}" for i in range(len(ALL))])
def test_adversarial_variant_is_rejected(kind, field, text):
    assert run(kind, field, text), (kind, field, text)


# ---- structural rules: one assertion per class
def test_character_allow_list_is_applied_to_every_string():
    o = good_output()
    o["caveats"] = [{"kind": "other", "text": "Plain."}]
    for path in (("what_this_run_is",), ("attribution",), ("numbers_summary",), ("caveats", 0, "text"), ("what_the_numbers_show", 0, "key"),
                 ("what_the_numbers_show", 1, "value"), ("see_also", 0)):
        for bad in ("​", "­", "‮", "а", "é", "가", "\t"):
            q = json.loads(json.dumps(o))
            cur = q
            for p in path[:-1]:
                cur = cur[p]
            cur[path[-1]] = cur[path[-1]] + bad
            assert PC.check_output(make_card(), q, KNOWN), (path, repr(bad))
    assert PC.bad_chars("plain text\nline") == "" and PC.bad_chars("aа") != ""


def test_ids_are_compared_casefolded():
    assert run("plain", "ws", "Related to AAAA1111.") and run("plain", "ws", "Related to Aaaa1111.")
    assert run("plain", "ws", "Run DEMO_K4_S1 here.")                # even the card's own name must be written as printed
    assert run("plain", "ws", "Compare OTHER_RUN_ALPHA.")


def test_exp_and_hypothesis_ids_only_in_see_also():
    assert run("plain", "see", "EXP-010_demo.md") == []
    assert run("multi", "see", "EXP-010_demo.md") == []
    assert run("plain", "ws", "Attributed to EXP-010.")
    assert run("plain", "ws", "Attributed to H1.")
    # the script sentences still carry the ids on a multi card
    assert PC.check_output(make_card(multi=True), good_output(multi=True, extra_caveats=("multi_attribution",)), KNOWN) == []


def test_status_words_only_in_script_sentences_or_as_the_status_value():
    assert run("snap", "ws", None, replace="Run demo_k4_s1 in project Demo, status running?.") == []
    assert run("plain", "ws", None, replace="Run demo_k4_s1 in project Demo, status ended.") == []
    assert run("plain", "ws", None, replace="Run demo_k4_s1 in project Demo, it ended.")        # not the "status <value>" form
    assert run("plain", "ws", None, replace="Run demo_k4_s1 in project Demo, status failed.")   # not the card's value
    assert run("nosum", "ws", None, replace="Run demo_k4_s1 in project Demo, status ended.")    # card status is no-summary
    # the status caveat is the exact script sentence
    o = good_output(status="no-summary", extra_caveats=("status",), texts={"status": "The run was possibly killed."})
    assert any("verbatim" in f for f in PC.check_output(make_card(status="no-summary"), o, KNOWN))


def test_number_forms():
    assert run("plain", "ns", None, replace="probe/test_exact was 0.00390625 at step 4882 and 1 at step 244100.") == []
    assert run("plain", "ns", "244100 is printed in the card.") == []                          # a printed number, as digits, is fine
    assert run("plain", "ns", "5e-08 is the lr minimum.") == []
    assert run("plain", "ns", "at step 244k")
    assert run("plain", "ns", "3rd")
    assert run("plain", "ns", "ten")
    assert PC.GLUE_RE.search("244k") and not PC.GLUE_RE.search("k2_s1")


def test_forbidden_file_has_the_new_sections():
    sec = PC.load_sections()
    assert sec["status"] and sec["numwords"] and any("hypothes" in x for x in sec["en"])
    assert any(x.startswith("crash") for x in sec["status"])


# ---- honest outputs: 6 outputs of different card kinds + a sentence corpus
def test_six_honest_outputs_accepted():
    cases = [
        ({}, {}),
        ({"cohort": "legacy", "truncated": True, "tier": "constructed-unverified", "duplicates": ("zzzz0000",)},
         {"cohort": "legacy", "truncated": True, "extra_caveats": ("truncated_history", "legacy_cohort", "duplicate_run_name", "link_tier"),
          "texts": {"link_tier": "Tier constructed-unverified; the URL is not api-confirmed.", "duplicate_run_name": "Other runs share this name: zzzz0000."}}),
        ({"truncated": True}, {"truncated": True, "extra_caveats": ("truncated_history",)}),
        ({"snapshot": True, "status": "running?"}, {"status": "running?", "snapshot": True, "extra_caveats": ("status", "snapshot")}),
        ({"multi": True}, {"multi": True, "extra_caveats": ("multi_attribution",)}),
        ({"status": "no-summary", "has_metadata": False, "cohort": "legacy", "tier": "no-remote"},
         {"status": "no-summary", "cohort": "legacy", "extra_caveats": ("status", "no_metadata", "legacy_cohort", "link_tier"),
          "texts": {"link_tier": "The tier is no-remote; the URL is not api-confirmed.", "no_metadata": "The card has no metadata file."}}),
    ]
    for ck, ok in cases:
        o = good_output(**ok)
        assert PC.check_output(make_card(**ck), o, KNOWN) == [], (ck, PC.check_output(make_card(**ck), o, KNOWN))


HONEST = [
    "probe/test_exact reads 0.00390625 at step 4882.", "probe/test_exact reads 1 at step 244100.",
    "probe/test_exact logs 0.00390625 at step 4882 and 1 at step 244100.", "probe/test_exact rises to 1 at step 244100.",
    "train/lm_loss drops to 0.002931 at step 244100.", "train/lm_loss is 2.54363 at step 1.", "train/lm_loss has its max 7.58577 at step 66745.",
    "train/lm_loss has its min 0.000471849 at step 105258.", "train/lr reaches 0.0001 at step 244100.", "train/lr stays at 5e-08 at step 1.",
    "The table records train/lm_loss as 0.002931 at step 244100.", "The table logs probe/test_exact with first 0.00390625 at step 4882.",
    "probe/test_exact has max 1 at step 43938.", "probe/test_exact has min 0 at step 63466.",
    "The last value of train/lm_loss is 0.002931 at step 244100.", "The first value of train/lm_loss is 2.54363 at step 1.",
    "The lowest train/lm_loss in the table is 0.000471849 at step 105258.", "The highest train/lm_loss in the table is 7.58577 at step 66745.",
    "probe/test_exact is 0 at step 63466 and 1 at step 43938.", "The table has 244100 rows for train/lm_loss.",
    "train/lr min is 5e-08 at step 1 and max is 0.0001 at step 2000.", "The stats table lists probe/test_exact, train/lm_loss and train/lr.",
    "probe/test_exact rises from 0.00390625 at step 4882 to 1 at step 244100.", "train/lm_loss falls from 2.54363 at step 1 to 0.002931 at step 244100.",
    "train/lm_loss peaks at 7.58577 at step 66745.", "probe/test_exact starts at 0.00390625 at step 4882.", "train/lr starts at 5e-08 at step 1.",
    "The max of train/lr is 0.0001 at step 2000.", "probe/test_exact has n 50.", "train/lm_loss has n 244100.", "probe/test_exact stays at 0 at step 63466.",
    "The log for train/lm_loss runs from step 1 to step 244100.", "train/lm_loss goes down to 0.000471849 at step 105258.",
    "The table shows train/lm_loss 0.002931 at step 244100.", "probe/test_exact is recorded as 1 at step 244100.", "The nonfinite count is 0 for each key.",
    "train/lm_loss is 0.002931 at the last step 244100.", "probe/test_exact is 1 at step 43938 and 1 at step 244100.",
    "The first row of probe/test_exact is 0.00390625 at step 4882.", "train/lr is 0.0001 at step 244100 and 5e-08 at step 1.",
    "The key probe/test_exact has values 0 and 1 for min and max.", "train/lm_loss min and max are 0.000471849 and 7.58577.",
    "The table says train/lm_loss is 2.54363 at step 1.", "train/lr is 5e-08 at step 1.",
    # verifier's two false positives, kept to measure: both stay rejected by design (status/evidence wording)
    "The table shows that the value is 1.", "The run failed to log.",
]
HONEST_MIN = 40


def test_honest_sentence_acceptance_at_least_40_of_46():
    assert len(HONEST) == 46
    from tools.api_batch import runcard as RC
    card = make_card()
    ents = [{"key": k, "stat": st, "value": v[0], "step": v[1]} for k, row in RC.stats_table(card).items() for st, v in row.items() if st in RC.STATS]
    ents = ents[:12]

    def honest(sent):                                    # every table cell is a listed entry, so only the wording is under test
        o = good_output()
        o["what_the_numbers_show"], o["numbers_summary"] = ents, sent
        return PC.check_output(card, o, KNOWN)
    ok = [s for s in HONEST if honest(s) == []]
    rej = [s for s in HONEST if s not in ok]
    print("honest accepted", len(ok), "of", len(HONEST), "rejected:", rej)
    assert len(ok) >= HONEST_MIN, rej
    assert "The table shows that the value is 1." in rej and "The run failed to log." in rej      # decided: stay banned


# ---- rendered attribution (the gate fail-open test is in test_ab_runcard.py)
def test_render_attribution_is_script_made_from_the_card():
    card = make_card(multi=True)
    L = PI.render_attribution(card)
    assert L == ["- exps: EXP-010 (source map, confidence registry); EXP-015 (source map, confidence registry); EXP-023 (source map, confidence registry)"]
    body = PI.render_model(good_output(multi=True, extra_caveats=("multi_attribution",)), "m", "b", "c", card)
    assert "unresolved here" in body and L[0] in body
    assert PI.render_attribution(make_card(exps=())) == []
