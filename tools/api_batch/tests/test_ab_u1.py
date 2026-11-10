"""U1 (design 6.2): map = one request per cell from curve CSVs + cell summaries, reduce = second batch over collected
items. Synthetic fixture only (no real experiment data); the mechanism, not the content, is tested."""
import json

import pytest

from tools.api_batch import ops, u1
from tools.api_batch.common import Halt
from tools.api_batch.ledger import Ledger
from tools.api_batch.pack import PackBuilder
from tools.api_batch.spec import SpecError, validate_spec

CELLS = [("g7", 3, 1), ("g7", 3, 2), ("g7", 5, 1), ("fullgrad", 3, 1)]
MAP_SPEC = """name: u1-test-map
use: U1
model: claude-sonnet-5-5
effort: medium
max_tokens: 8000
cache_ttl: 5m
header: shared/u1_header.md
prompt: {{system_template: tools/api_batch/prompts/u1_map.md}}
output_schema: tools/api_batch/schemas/review_note.json
data:
  train_curves: 31_demo/curves/train_curves.csv
  decode_curves: 31_demo/curves/decode_curves.csv
  cell_stats: 21_experiments/run/cell_stats.csv
  tstar: 21_experiments/run/tstar.csv
budget: {{cap_usd_hard_worst: {cap}}}
"""
RED_SPEC = """name: u1-test-reduce
use: U1-reduce
model: claude-sonnet-5-5
effort: high
max_tokens: 8000
digest_max_tokens: 16000
cache_ttl: none
header: shared/u1_header.md
prompt: {system_template: tools/api_batch/prompts/u1_reduce.md}
map_batch: 38_api-batch/11_batches/map
group_by: [arm, k]
digest: true
budget: {cap_usd_hard_worst: 1.5}
"""
HEADER = "# SHARED HEADER\nDefinitions only; nothing denied here.\n"


def rn(a, k, s):
    return f"pp_ltf_{a}_lay2_k{k}_s{s}"


def write_data(lab, repo, *, bad_row=None, short_decode=False):
    cur = lab / "31_demo/curves"
    cur.mkdir(parents=True)
    tr = ["run_name,arm,k,seed,frozen_class,step,train_loss,probe_test_exact"]
    dc = ["run_name,arm,k,seed,cell_class_Z,t,a_t,v_t"]
    cs = ["exp_id,unit,run_name,arm,k,seed,frozen_class,split,readout_kind,J,Z"]
    ts = ["exp_id,unit,run_name,split,gcd_class,n_rows,median"]
    for a, k, s in CELLS:
        n = rn(a, k, s)
        arm = "full" if a == "fullgrad" else a
        tr += [f"{n},{arm},{k},{s},G,{i * 100 + 1},0.{i:03d},{'1.0' if i % 5 == 0 else ''}" for i in range(59)]
        dc += [f"{n},{arm},{k},{s},jump,{t},0.{t:02d},0.5" for t in range(1, 42 if short_decode else 43)]
        cs += [f"EXP-022,U__{n},{n},{arm},{k},{s},G,test,{rk},20,jump" for rk in ("ltf", "dgrid")]
        ts += [f"EXP-022,U__{n},{n},test,{g},1000,14.0" for g in ("all", "1", ">1")]
    if bad_row:
        cs.append(bad_row)
    (cur / "train_curves.csv").write_text("\n".join(tr) + "\n", encoding="utf-8")
    (cur / "decode_curves.csv").write_text("\n".join(dc) + "\n", encoding="utf-8")
    d = lab / "21_experiments/run"
    d.mkdir(parents=True)
    (d / "cell_stats.csv").write_text("\n".join(cs) + "\n", encoding="utf-8")
    (d / "tstar.csv").write_text("\n".join(ts) + "\n", encoding="utf-8")


@pytest.fixture
def env(tmp_path, deny_files, lab):
    root = lab / "38_api-batch"
    mbd, rbd = root / "11_batches/map", root / "11_batches/reduce"
    for bd, spec in ((mbd, MAP_SPEC.format(cap=3.5)), (rbd, RED_SPEC)):
        (bd / "shared").mkdir(parents=True)
        (bd / "spec.yaml").write_text(spec, encoding="utf-8")
        (bd / "shared/u1_header.md").write_text(HEADER, encoding="utf-8")
    write_data(lab, None)

    def builder(bd):
        return PackBuilder.from_files(deny_files, lab_root=lab, batch_dir=bd)
    return type("E", (), dict(lab=lab, root=root, mbd=mbd, rbd=rbd, builder=builder, led=Ledger(root / "ledger.jsonl"),
                              deny_files=deny_files))


def test_map_plan_one_request_per_cell(env):
    d = ops.plan(env.mbd, env.builder(env.mbd), env.led)
    assert d["n_requests"] == len(CELLS) and d["use"] == "U1" and d["est_usd_hard_worst"] <= d["cap_usd"]
    rows, _ = __import__("tools.api_batch.manifest", fromlist=["x"]).load_manifest(env.mbd / "manifest.jsonl")
    assert {r["meta"]["cell"] for r in rows} == {rn(*c) for c in CELLS}
    r = rows[0]
    assert r["custom_id"].startswith("u1m-") and r["params"]["model"] == "claude-sonnet-5-5"
    assert r["params"]["output_config"]["effort"] == "medium" and r["params"]["max_tokens"] == 8000
    blocks = r["params"]["messages"][0]["content"]
    assert blocks[0]["type"] == "document" and blocks[0]["cache_control"] and blocks[1]["type"] == "text"   # header cached, cell is the tail
    assert blocks[1]["text"].startswith("# CELL pp_ltf_")
    assert r["params"]["output_config"]["format"]["schema"]["required"] == ["scope_id", "cell_summary", "notes"]
    assert d["u1"]["n_cells"] == len(CELLS)
    cell = [p for p in d["scan"] if p["name"].startswith("cell-")][0]
    assert cell["mode"] == "verify-only" and cell["kind"] == "u1_data"
    assert {p["kind"] for p in d["scan"]} >= {"u1_data", "spec_text"}
    assert env.led.last("planned")["n_requests"] == len(CELLS)


def test_cell_slice_is_verbatim_rows_without_identity_columns():
    texts = {"train_curves": "run_name,arm,k,seed,frozen_class,step,train_loss,probe_test_exact\n" + "".join(
        f"pp_ltf_g7_lay2_k3_s1,g7,3,1,G,{i},0.{i:02d},\n" for i in range(1, 51)),
        "decode_curves": "run_name,arm,k,seed,cell_class_Z,t,a_t,v_t\n" + "".join(f"pp_ltf_g7_lay2_k3_s1,g7,3,1,jump,{t},0.5,1\n" for t in range(1, 43)),
        "cell_stats": "exp_id,unit,run_name,arm,k,seed,split,Z\nEXP-022,U,pp_ltf_g7_lay2_k3_s1,g7,3,1,test,jump\nEXP-022,U,pp_ltf_g7_lay2_k3_s1,g7,3,1,train,jump\n",
        "tstar": "exp_id,unit,run_name,split,gcd_class,median\n" + "".join(f"EXP-022,U,pp_ltf_g7_lay2_k3_s1,test,{g},14.0\n" for g in ("all", "1", ">1"))}
    (c,) = u1.build_cells(texts)
    assert "split,Z\ntest,jump\ntrain,jump" in c["text"] and "pp_ltf_g7_lay2_k3_s1,g7" not in c["text"].split("\n", 1)[1]
    assert "\n17,0.17,\n" in c["text"] and c["rows"] == {"train_curves": 50, "decode_curves": 42, "cell_stats": 2, "tstar": 3}


def test_deny_hit_in_any_data_file_refuses_whole_run(env):
    (env.lab / "21_experiments/run/cell_stats.csv").write_text(
        (env.lab / "21_experiments/run/cell_stats.csv").read_text() + "x,y,pp_ltf_g7_lay2_k3_s1,g7,3,1,G,test,ltf,20,EXP-016\n", encoding="utf-8")
    with pytest.raises(Halt, match="L3b"):
        ops.plan(env.mbd, env.builder(env.mbd), env.led)
    assert not (env.mbd / "manifest.jsonl").exists() and not env.led.rows("planned")


def test_deny_hit_in_header_or_spec_refuses(env):
    (env.mbd / "shared/u1_header.md").write_text(HEADER + "see per_call.csv\n", encoding="utf-8")
    with pytest.raises(Halt):
        ops.plan(env.mbd, env.builder(env.mbd), env.led)


def test_l1_refuses_unlisted_curve_location_and_shape_change(env):
    b = env.builder(env.mbd)
    b._l1("u1_data", "repo", "reports/2026-10-23_exp022-data/curves/train_curves.csv")          # the user-permitted location
    for bad in ("reports/notes/x.csv", "reports/2026-10-23_exp022-data/curves/x.json", "reports/2026-10-23_a-data/other/x.csv",
                "reports/2026-10-23_a-data/curves/sub/x.csv", "tools/x.csv"):
        with pytest.raises(Halt, match="L1"):
            b._l1("u1_data", "repo", bad)
    b._l1("spec_text", "batch", "shared/u1_header.md")
    with pytest.raises(Halt, match="L1"):
        b._l1("spec_text", "batch", "shared/sub/x.md")
    b._l1("derived", "lab", "38_api-batch/11_batches/map/items/a.md")
    with pytest.raises(Halt, match="L1"):
        b._l1("derived", "lab", "38_api-batch/11_batches/map/raw/a.md")


def test_short_decode_halts(tmp_path, deny_files, lab):
    write_data(lab, None, short_decode=True)
    with pytest.raises(Halt, match="decode_curves has 41 rows"):
        u1.build_cells({n: (lab / f"31_demo/curves/{n}.csv").read_text() if n.endswith("curves") else (lab / f"21_experiments/run/{n}.csv").read_text()
                        for n in ("train_curves", "decode_curves", "cell_stats", "tstar")})


def test_over_cap_refused(env):
    (env.mbd / "spec.yaml").write_text(MAP_SPEC.format(cap=0.01))
    with pytest.raises(Halt, match="plan refused"):
        ops.plan(env.mbd, env.builder(env.mbd), env.led)


def test_spec_validation():
    s = __import__("yaml").safe_load(MAP_SPEC.format(cap=3.5))
    validate_spec(s)
    for key in ("header", "output_schema", "data"):
        bad = {k: v for k, v in s.items() if k != key}
        with pytest.raises(SpecError):
            validate_spec(bad)
    with pytest.raises(SpecError):
        validate_spec({**s, "data": {"train_curves": "x"}})
    with pytest.raises(SpecError):
        validate_spec({**s, "effort": None})
    r = __import__("yaml").safe_load(RED_SPEC)
    validate_spec(r)
    with pytest.raises(SpecError):
        validate_spec({**r, "group_by": ["k"]})


def _fake_collect(env, *, skip=None, status="succeeded", l5="pass"):
    rows = [json.loads(ln) for ln in (env.mbd / "manifest.jsonl").read_text().splitlines()]
    (env.mbd / "items").mkdir(exist_ok=True)
    inbox = []
    for i, r in enumerate(rows):
        if r["custom_id"] == skip:
            continue
        cid = r["custom_id"]
        (env.mbd / "items" / f"{cid}.md").write_text(
            f"---\nstatus: DRAFT (api-batch output — not evidence)\nuse: U1\ncustom_id: {cid}\nl5: {l5}\n---\n"
            + json.dumps({"scope_id": r["meta"]["cell"], "cell_summary": "flat", "notes": []}) + "\n", encoding="utf-8")
        inbox.append({"custom_id": cid, "batch_id": "msgbatch_x", "use": "U1", "result_status": status, "item": f"items/{cid}.md"})
    (env.root / "INBOX.jsonl").write_text("".join(json.dumps(x) + "\n" for x in inbox), encoding="utf-8")
    return rows


def test_reduce_groups_and_digest(env):
    ops.plan(env.mbd, env.builder(env.mbd), env.led)
    _fake_collect(env)
    d = ops.plan(env.rbd, env.builder(env.rbd), env.led)
    # groups (arm,k): fullgrad k3, g7 k3, g7 k5  + digest
    assert d["use"] == "U1-reduce" and d["n_requests"] == 4 and set(d["u1"]["scopes"]) == {"fullgradk3", "g7k3", "g7k5", "digest"}
    rows = [json.loads(ln) for ln in (env.rbd / "manifest.jsonl").read_text().splitlines()]
    by = {r["meta"]["scope"]: r for r in rows}
    assert by["g7k3"]["meta"]["n_items"] == 2 and by["digest"]["meta"]["n_items"] == 4
    assert by["digest"]["params"]["max_tokens"] == 16000 and by["g7k3"]["params"]["max_tokens"] == 8000
    assert all(r["custom_id"].startswith("u1r-") and r["params"]["output_config"]["effort"] == "high" for r in rows)
    assert all("format" not in r["params"]["output_config"] for r in rows)
    docs = by["g7k3"]["params"]["messages"][0]["content"]
    assert docs[0]["title"] == "u1_header" and all(b["type"] in ("document", "text") for b in docs) and docs[-1]["type"] == "text"
    assert d["est_usd_hard_worst"] <= 1.5


def test_reduce_scopes_subset(env):
    ops.plan(env.mbd, env.builder(env.mbd), env.led)
    _fake_collect(env)
    spec = env.rbd / "spec.yaml"
    base = spec.read_text()
    spec.write_text(base + "scopes: [g7k3, nope]\n")
    with pytest.raises(Halt, match="nope"):
        ops.plan(env.rbd, env.builder(env.rbd), env.led)
    spec.write_text(base + "scopes: [g7k3, digest]\n")
    d = ops.plan(env.rbd, env.builder(env.rbd), env.led)
    assert d["n_requests"] == 2 and d["u1"]["scopes"] == ["g7k3", "digest"]


def test_reduce_requires_complete_clean_map(env):
    ops.plan(env.mbd, env.builder(env.mbd), env.led)
    rows = _fake_collect(env, skip=json.loads((env.mbd / "manifest.jsonl").read_text().splitlines()[0])["custom_id"])
    with pytest.raises(Halt, match="not fully collected"):
        ops.plan(env.rbd, env.builder(env.rbd), env.led)
    _fake_collect(env, status="refusal")
    with pytest.raises(Halt, match="not fully collected"):
        ops.plan(env.rbd, env.builder(env.rbd), env.led)
    _fake_collect(env, l5="quarantined")                    # derived Pack needs l5: pass
    with pytest.raises(Halt, match="l5"):
        ops.plan(env.rbd, env.builder(env.rbd), env.led)
    assert not (env.rbd / "manifest.jsonl").exists()


def test_reduce_refuses_edited_map_manifest(env):
    ops.plan(env.mbd, env.builder(env.mbd), env.led)
    _fake_collect(env)
    (env.mbd / "manifest.jsonl").write_text((env.mbd / "manifest.jsonl").read_text().replace("pp_ltf_g7", "pp_ltf_g8", 1))
    with pytest.raises(Halt):
        ops.plan(env.rbd, env.builder(env.rbd), env.led)
