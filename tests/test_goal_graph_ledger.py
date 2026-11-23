from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / ".agents/skills/goal-graph-orchestrator/scripts/ledger.py"


def load_ledger_module():
    spec = importlib.util.spec_from_file_location("goal_graph_ledger_test_module", LEDGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def trusted_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_ledger_module()
    mount = tmp_path / "mnt"
    user_root = mount / "ayp"
    parent = user_root / "agents"
    parent.mkdir(parents=True)
    for anchor in (mount, user_root, parent):
        anchor.chmod(0o755)
    trusted = parent / "trm"
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", trusted)
    return module, trusted


def run(*args: object, ok: bool = True, cwd: Path = ROOT) -> dict:
    completed = subprocess.run(
        [sys.executable, str(LEDGER), *map(str, args)],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if ok:
        assert completed.returncode == 0, completed.stdout + completed.stderr
    else:
        assert completed.returncode != 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def scaffold(
    tmp_path: Path, *, goal_effort: int = 100, terra_effort: int = 60, terra_reserve: int = 12,
) -> tuple[Path, Path]:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-test",
        "--objective", "exercise engine", "--wall-minutes", 60,
        "--effort-minutes", goal_effort, "--verification-reserve", 20,
        "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-test"
    run(
        "add-terra", goal, "--subproject-id", "SP-01", "--owner", "terra-author",
        "--objective", "own intermediate project", "--wall-minutes", 30,
        "--effort-minutes", terra_effort, "--verification-reserve", terra_reserve,
        "--concurrency", 2, "--write-scope", "outputs/goal-graph-test",
    )
    return goal, goal / "subprojects/SP-01/ledger.yaml"


def add_task(terra: Path, task_id: str, output: str, *dependencies: str) -> None:
    args: list[object] = [
        "add-task", terra, "--task-id", task_id, "--objective", f"produce {output}",
        "--parent-criterion", f"{output} is valid", "--criterion", "artifact is valid",
        "--evidence-path", output, "--timebox", 15, "--write-scope", output,
        "--output", output,
    ]
    for dependency in dependencies:
        args += ["--depends-on", dependency]
    run(*args)


def start(terra: Path, task_id: str = "SP-01-L01", *, agent: str = "worker-1", model: str = "gpt-5.6-terra") -> Path:
    return Path(run("start-attempt", terra, "--task-id", task_id, "--agent-id", agent, "--actual-model", model)["attempt_dir"])


def pass_review(attempt: Path, *, review_id: str = "fresh", agent: str = "verifier-1") -> dict:
    """A PASS review must carry direct evidence for every frozen criterion."""
    return run(
        "record-review", attempt, "--review-id", review_id, "--seat", "luna-verifier",
        "--agent-id", agent, "--actual-model", "gpt-5.6-terra", "--verdict", "PASS",
        "--criterion-pass", f"C1={attempt / 'result.yaml'}", "--form-check", "schema",
        "--intent-check", "outcome", "--evidence", attempt / "result.yaml",
    )


def record_feedback(attempt: Path, *, ok: bool = True) -> dict:
    return run(
        "record-feedback", attempt, "--failed-item", "C1", "--observed", "artifact failed",
        "--expected", "artifact passes", "--evidence", attempt / "result.yaml",
        "--likely-cause", "implementation defect", "--minimal-repair", "repair artifact",
        "--rerun-scope", "outputs/goal-graph-test/a.txt", "--rerun-command", "rtk uv run pytest",
        "--owner", "terra-author", ok=ok,
    )


def fail_review(
    attempt: Path, *, review_id: str = "fresh-fail", agent: str = "verifier-1", ok: bool = True,
) -> dict:
    """An independent verifier refusing the worker's own PASS self-report."""
    return run(
        "record-review", attempt, "--review-id", review_id, "--seat", "luna-verifier",
        "--agent-id", agent, "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL",
        "--criterion", f"C1=FAIL={attempt / 'result.yaml'}", "--form-check", "schema",
        "--intent-check", "criterion not met", "--evidence", attempt / "result.yaml", ok=ok,
    )


def pass_result(attempt: Path, *, agent: str = "worker-1") -> None:
    evidence = attempt.parent.parent / "orders/001.yaml"
    run(
        "record-result", attempt, "--agent-id", agent, "--actual-model", "gpt-5.6-terra",
        "--verdict", "PASS", "--elapsed-minutes", 1, "--evidence", evidence,
        "--form-check", "schema", "--intent-check", "worker asserts the predicate holds",
    )


def init_goal(tmp_path: Path, goal_id: str, objective: str) -> Path:
    """A bare goal with no terra under it."""
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", goal_id, "--objective", objective,
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test",
    )
    return state / goal_id


def edit_yaml(path: Path, mutate) -> None:
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def mark_first_child_pass(goal: Path) -> None:
    edit_yaml(goal / "subprojects/SP-01/ledger.yaml", lambda d: d["children"][0].update(status="PASS"))


def sha_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_happy_path_requires_independent_review_and_resumes(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra, "SP-01-L02", "outputs/goal-graph-test/b.txt", "SP-01-L01")
    assert run("ready", terra)["ready"] == ["SP-01-L01"]
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    evidence = attempt.parent.parent / "orders/001.yaml"
    run("record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS", "--elapsed-minutes", 1, "--evidence", evidence, "--form-check", "schema", "--intent-check", "outcome")
    run("transition", terra, "--node", "SP-01-L01", "--to", "VERIFYING")
    failed = run("record-review", attempt, "--review-id", "self", "--seat", "luna-verifier", "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS", "--form-check", "schema", "--intent-check", "outcome", "--evidence", attempt / "result.yaml", ok=False)
    assert "differ" in failed["error"]
    review = pass_review(attempt)
    run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", "--evidence", review["review"])
    assert run("ready", terra)["ready"] == ["SP-01-L02"]
    assert run("validate", goal)["error_count"] == 0
    assert any(node["id"] == "SP-01-L01" and node["status"] == "PASS" for node in run("status", goal)["nodes"])


def test_pass_without_evidence_and_illegal_transition_are_rejected(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    assert "illegal transition" in run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", ok=False)["error"]
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    assert "illegal transition" in run("transition", terra, "--node", "SP-01-L01", "--to", "PASS", ok=False)["error"]


def test_validate_rejects_cycle_and_active_scope_overlap(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/shared")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/shared/child")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["dependencies"] = ["SP-01-L02"]
    terra["children"][1]["dependencies"] = ["SP-01-L01"]
    terra["children"][0]["status"] = "READY"
    terra["children"][1]["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("cycle" in error for error in result["errors"])
    assert any("write-scope overlap" in error for error in result["errors"])


def test_validate_rejects_budget_reserve_consumption(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["graph"]["nodes"][0]["allocation"]["agent_effort_budget_minutes"] = 81
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("consume verification reserve" in error for error in result["errors"])


def test_immutable_order_and_optimistic_revision(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    duplicate = run("add-task", terra, "--task-id", "SP-01-L01", "--objective", "different", "--parent-criterion", "same", "--criterion", "same", "--evidence-path", "outputs/goal-graph-test/a.txt", "--write-scope", "outputs/goal-graph-test/a.txt", ok=False)
    assert "duplicate task" in duplicate["error"]
    stale = run("transition", terra, "--node", "SP-01-L01", "--to", "READY", "--expected-revision", 0, ok=False)
    assert "revision mismatch" in stale["error"]


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    result = run("init", "--state-root", tmp_path, "--goal-id", "G-20260802-escape", "--objective", "escape", "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "/etc", ok=False)
    assert "escapes workspace" in result["error"]


def test_trusted_external_scope_is_canonical_and_stable_when_root_appears(trusted_scope) -> None:
    module, trusted = trusted_scope
    requested = trusted / "future/artifact"
    before = module.safe_scope(str(requested))
    trusted.mkdir(mode=0o700)
    after = module.safe_scope(str(requested))
    assert before == after == requested.as_posix()

    target = trusted / "canonical"
    target.mkdir()
    (trusted / "alias").symlink_to(target, target_is_directory=True)
    assert module.safe_scope(str(trusted / "alias/artifact")) == (target / "artifact").as_posix()


def test_trusted_external_scope_rejects_broad_prefixes_traversal_and_relative_alias(trusted_scope) -> None:
    module, trusted = trusted_scope
    rejected = [
        trusted.parent,
        trusted.parent.parent,
        trusted.parent.parent / "agents-not-approved",
        trusted.parent.parent / "trm-not-approved",
        trusted.with_name("trm2"),
        trusted / ".." / "trm2",
    ]
    for path in rejected:
        with pytest.raises(module.LedgerError, match="escapes workspace"):
            module.safe_scope(str(path))

    relative_alias = os.path.relpath(trusted / "artifact", module.WORKSPACE)
    with pytest.raises(module.LedgerError, match="must be absolute"):
        module.safe_scope(relative_alias)


def test_workspace_root_does_not_contain_the_external_trusted_root(trusted_scope) -> None:
    module, trusted = trusted_scope
    assert not module.contained(str(trusted / "artifact"), ".")
    assert not module.overlaps(str(trusted / "artifact"), ".")
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        module.validate_delegated_scope([str(trusted / "artifact")], ["."], [])


def test_trusted_external_anchor_rejects_unsafe_permissions_and_symlinks(trusted_scope) -> None:
    module, trusted = trusted_scope
    parent = module.TRUSTED_EXTERNAL_ANCHORS[-1]
    parent.chmod(0o777)
    with pytest.raises(module.LedgerError, match=r"parent permissions are unsafe \(0777\)"):
        module.safe_scope(str(trusted / "artifact"))

    parent.chmod(0o755)
    outside = parent.parent / "outside"
    outside.mkdir()
    trusted.symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.LedgerError, match="trusted root must be a real directory"):
        module.safe_scope(str(trusted / "artifact"))

    trusted.unlink()
    trusted.mkdir(mode=0o700)
    trusted.chmod(0o777)
    with pytest.raises(module.LedgerError, match=r"trusted root permissions are unsafe \(0777\)"):
        module.safe_scope(str(trusted / "artifact"))
    trusted.chmod(0o700)
    (trusted / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(module.LedgerError, match="symlink escapes trusted root"):
        module.safe_scope(str(trusted / "escape/artifact"))


def test_trusted_external_anchor_rejects_a_symlinked_parent(trusted_scope, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = trusted_scope
    mount, user_root, _ = module.TRUSTED_EXTERNAL_ANCHORS
    real_parent = user_root / "real-agents"
    real_parent.mkdir()
    linked_parent = user_root / "agents-link"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, linked_parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", linked_parent / "trm")
    with pytest.raises(module.LedgerError, match="anchor must be a real directory"):
        module.safe_scope(str(linked_parent / "trm/artifact"))


def test_trusted_external_anchor_rejects_missing_or_wrong_owner(trusted_scope, monkeypatch: pytest.MonkeyPatch) -> None:
    module, trusted = trusted_scope
    mount, user_root, parent = module.TRUSTED_EXTERNAL_ANCHORS
    missing_parent = user_root / "missing-agents"
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, missing_parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", missing_parent / "trm")
    with pytest.raises(module.LedgerError, match="anchor is missing"):
        module.safe_scope(str(missing_parent / "trm/artifact"))

    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ANCHORS", (mount, user_root, parent))
    monkeypatch.setattr(module, "TRUSTED_EXTERNAL_ROOT", trusted)
    actual_uid = os.geteuid()
    monkeypatch.setattr(module.os, "geteuid", lambda: actual_uid + 1)
    with pytest.raises(module.LedgerError, match="parent must be owned by uid"):
        module.safe_scope(str(trusted / "artifact"))


def test_missing_dependency_is_rejected(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["dependencies"] = ["SP-01-L99"]
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("depends on missing SP-01-L99" in error for error in result["errors"])


def test_tampered_ready_node_cannot_bypass_dependency_gate(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/b.txt", "SP-01-L01")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][1]["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("before dependency SP-01-L01 PASS" in error for error in result["errors"])
    rejected = run(
        "start-attempt", terra_path, "--task-id", "SP-01-L02", "--agent-id", "worker-2",
        "--actual-model", "gpt-5.6-terra", ok=False,
    )
    assert "dependencies must PASS" in rejected["error"]


def test_failure_local_retry_preserves_order_and_prior_attempt(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = Path(run("start-attempt", terra, "--task-id", "SP-01-L01", "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra")["attempt_dir"])
    run("record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema", "--intent-check", "failed predicate")
    record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    frozen_hash = hashlib.sha256((first / "result.yaml").read_bytes() + order.read_bytes()).hexdigest()
    run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", first / "feedback.yaml")
    second = Path(run("start-attempt", terra, "--task-id", "SP-01-L01", "--agent-id", "worker-2", "--actual-model", "gpt-5.6-terra")["attempt_dir"])
    assert second.name == "002"
    assert hashlib.sha256((first / "result.yaml").read_bytes() + order.read_bytes()).hexdigest() == frozen_hash


def test_missing_immutable_order_is_rejected(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    order.rename(order.with_suffix(".missing"))
    result = run("validate", goal, ok=False)
    assert any("immutable order is missing" in error for error in result["errors"])


def test_active_concurrency_quota_is_enforced(tmp_path: Path) -> None:
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra_path, "SP-01-L02", "outputs/goal-graph-test/b.txt")
    terra = yaml.safe_load(terra_path.read_text())
    terra["schedule"]["concurrency_quota"] = 1
    for child in terra["children"]:
        child["status"] = "READY"
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("active tasks exceed concurrency quota" in error for error in result["errors"])


def test_result_identity_must_match_frozen_assignment(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra, agent="worker-1", model="gpt-5.6-terra")
    mismatch = run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-sol",
        "--verdict", "PASS", "--elapsed-minutes", 1, ok=False,
    )
    assert "agent/model" in mismatch["error"]


def test_fail_result_cannot_be_promoted_by_a_pass_review(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    rejected = run(
        "record-review", attempt, "--review-id", "fresh", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "PASS",
        "--criterion-pass", f"C1={attempt / 'result.yaml'}", "--form-check", "schema",
        "--intent-check", "outcome", "--evidence", attempt / "result.yaml", ok=False,
    )
    assert "non-PASS" in rejected["error"]


def test_unsafe_ids_and_delegated_scope_violations_are_rejected(tmp_path: Path) -> None:
    unsafe_goal = run(
        "init", "--state-root", tmp_path, "--goal-id", "G-20260802/bad", "--objective", "bad",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test", ok=False,
    )
    assert "unsafe goal" in unsafe_goal["error"]
    goal, terra = scaffold(tmp_path)
    unsafe_task = run(
        "add-task", terra, "--task-id", "SP-01-L01/escape", "--objective", "bad",
        "--parent-criterion", "bad", "--criterion", "bad", "--evidence-path", "proof",
        "--write-scope", "outputs/goal-graph-test/a.txt", ok=False,
    )
    assert "unsafe task" in unsafe_task["error"]
    escaped_scope = run(
        "add-terra", goal, "--subproject-id", "SP-02", "--owner", "terra-author",
        "--objective", "outside", "--wall-minutes", 10, "--effort-minutes", 10,
        "--verification-reserve", 2, "--write-scope", "outputs/outside", ok=False,
    )
    assert "exceeds parent scope" in escaped_scope["error"]
    data = yaml.safe_load(terra.read_text())
    data["contract"]["forbidden_paths"] = ["outputs/goal-graph-test/denied"]
    terra.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    forbidden = run(
        "add-task", terra, "--task-id", "SP-01-L01", "--objective", "bad",
        "--parent-criterion", "bad", "--criterion", "bad", "--evidence-path", "proof",
        "--write-scope", "outputs/goal-graph-test/denied", ok=False,
    )
    assert "forbidden" in forbidden["error"]


def test_validate_rejects_tampered_goal_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, _ = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["contract"]["write_scope"] = [str(trusted.with_name("trm2"))]
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    errors = module.validate_root(goal)
    assert any("goal contract has unsafe scope" in error and "escapes workspace" in error for error in errors)


def test_validate_rejects_tampered_terra_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra_path = scaffold(tmp_path)
    terra = yaml.safe_load(terra_path.read_text())
    terra["contract"]["write_scope"] = [str(trusted / "terra")]
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")
    errors = module.validate_root(goal)
    assert any("SP-01 has unsafe contract scope" in error and "exceeds parent scope" in error for error in errors)


def test_tampered_order_external_scope_fails_validation_and_start(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra_path = scaffold(tmp_path)
    add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    order_path = terra_path.parent / "tasks/SP-01-L01/orders/001.yaml"
    order = yaml.safe_load(order_path.read_text())
    order["scope"]["write_scope"] = [str(trusted / "task")]
    order_path.write_text(yaml.safe_dump(order, sort_keys=False), encoding="utf-8")
    terra = yaml.safe_load(terra_path.read_text())
    terra["children"][0]["order_sha256"] = hashlib.sha256(order_path.read_bytes()).hexdigest()
    terra_path.write_text(yaml.safe_dump(terra, sort_keys=False), encoding="utf-8")

    errors = module.validate_root(goal)
    assert any("unsafe order scope" in error and "exceeds parent scope" in error for error in errors)
    run("transition", terra_path, "--node", "SP-01-L01", "--to", "READY")
    args = module.parser().parse_args([
        "start-attempt", str(terra_path), "--task-id", "SP-01-L01",
        "--agent-id", "worker-1", "--actual-model", "test-model",
    ])
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        args.func(args)


def test_tampered_feedback_external_scope_fails_validation_and_reissue(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    feedback = Path(record_feedback(attempt)["feedback"])
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    packet = yaml.safe_load(feedback.read_text())
    packet["rerun_scope"] = [str(trusted / "rerun")]
    feedback.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")

    errors = module.validate_root(goal)
    assert any("unsafe feedback scope" in error and "exceeds parent scope" in error for error in errors)
    args = module.parser().parse_args([
        "reissue-order", str(terra), "--task-id", "SP-01-L01", "--feedback", str(feedback),
    ])
    with pytest.raises(module.LedgerError, match="exceeds parent scope"):
        args.func(args)


def test_reissue_rejects_tampered_root_or_terra_external_scope(tmp_path: Path, trusted_scope) -> None:
    module, trusted = trusted_scope
    for tamper in ("root", "terra"):
        goal, terra_path = scaffold(tmp_path / tamper)
        add_task(terra_path, "SP-01-L01", "outputs/goal-graph-test/a.txt")
        run("transition", terra_path, "--node", "SP-01-L01", "--to", "READY")
        attempt = start(terra_path)
        run(
            "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
            "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
            "--intent-check", "failed predicate",
        )
        feedback = Path(record_feedback(attempt)["feedback"])
        run("transition", terra_path, "--node", "SP-01-L01", "--to", "FAIL")
        target = goal / "goal-ledger.yaml" if tamper == "root" else terra_path
        data = yaml.safe_load(target.read_text())
        data["contract"]["write_scope"] = [
            str(trusted.with_name("trm2")) if tamper == "root" else str(trusted / "terra")
        ]
        target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        args = module.parser().parse_args([
            "reissue-order", str(terra_path), "--task-id", "SP-01-L01", "--feedback", str(feedback),
        ])
        expected = "escapes workspace" if tamper == "root" else "exceeds parent scope"
        with pytest.raises(module.LedgerError, match=expected):
            args.func(args)


def test_reissue_requires_immutable_feedback_and_preserves_prior_order(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path, terra_effort=50)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    missing = run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL", ok=False)
    assert "feedback" in missing["error"]
    prior_order = terra.parent / "tasks/SP-01-L01/orders/001.yaml"
    frozen = prior_order.read_bytes()
    missing = run("reissue-order", terra, "--task-id", "SP-01-L01", ok=False)
    assert "feedback" in missing["error"]
    feedback = record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    reissued = run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", feedback["feedback"])
    assert Path(reissued["order"]).name == "002.yaml"
    assert prior_order.read_bytes() == frozen
    terra_data = yaml.safe_load(terra.read_text())
    child = terra_data["children"][0]
    assert child["order_version"] == 2
    assert child["current_attempt"] == 1


def test_retry_cannot_consume_terra_verification_reserve(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path, terra_effort=30, terra_reserve=12)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "failed predicate",
    )
    feedback = record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    rejected = run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", feedback["feedback"], ok=False)
    assert "reserve" in rejected["error"]


def test_ordinary_first_failure_cannot_fall_back_to_sol(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    first = start(terra)
    run(
        "record-result", first, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema",
        "--intent-check", "local defect",
    )
    record_feedback(first)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    rejected = run("transition", terra, "--node", "SP-01-L01", "--to", "ESCALATED", ok=False)
    assert "escalation packet" in rejected["error"]
    packet = run(
        "record-escalation", first, "--root-contract-field", "authorization",
        "--observed", "required operation is not authorized", "--decision-required", "approve or deny operation",
        "--evidence", first / "result.yaml", "--requested-by", "terra-author",
    )
    run("transition", terra, "--node", "SP-01-L01", "--to", "ESCALATED", "--escalation", packet["escalation"])


def test_nonpass_review_must_score_every_frozen_criterion(tmp_path: Path) -> None:
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    run(
        "record-result", attempt, "--agent-id", "worker-1", "--actual-model", "gpt-5.6-terra",
        "--verdict", "FAIL", "--elapsed-minutes", 1, "--form-check", "schema", "--intent-check", "defect",
    )
    rejected = run(
        "record-review", attempt, "--review-id", "negative-empty", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL", ok=False,
    )
    assert "score every frozen criterion" in rejected["error"]
    run(
        "record-review", attempt, "--review-id", "negative-complete", "--seat", "luna-verifier",
        "--agent-id", "verifier-1", "--actual-model", "gpt-5.6-terra", "--verdict", "FAIL",
        "--criterion", f"C1=FAIL={attempt / 'result.yaml'}",
    )


def test_terra_review_author_is_bound_to_frozen_owner(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    rejected = run(
        "record-terra-review", terra.parent, "--review-id", "forged", "--author-agent-id", "fake-author",
        "--agent-id", "terra-author", "--actual-model", "gpt-5.6-sol", "--verdict", "PASS",
        "--form-check", "schema", "--intent-check", "outcome", "--evidence", terra, ok=False,
    )
    assert "frozen owner" in rejected["error"]


def test_terra_fallback_requires_root_contract_packet(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    goal_ledger = goal / "goal-ledger.yaml"
    run("transition", goal_ledger, "--node", "SP-01", "--to", "BLOCKED")
    rejected = run("transition", goal_ledger, "--node", "SP-01", "--to", "ESCALATED", ok=False)
    assert "Terra Sol fallback" in rejected["error"]
    packet = run(
        "record-terra-escalation", terra.parent, "--root-contract-field", "shared_file_owner",
        "--observed", "shared file has no owner", "--decision-required", "assign one owner",
        "--evidence", terra, "--requested-by", "terra-author",
    )
    run("transition", goal_ledger, "--node", "SP-01", "--to", "ESCALATED", "--escalation", packet["escalation"])
    assert run("validate", goal)["error_count"] == 0


def test_recursive_child_terra_is_structurally_valid_and_scope_bounded(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    child = run(
        "add-child-terra", terra, "--subproject-id", "SP-01-C01", "--owner", "terra-child",
        "--objective", "independent child criterion", "--wall-minutes", 10, "--effort-minutes", 10,
        "--verification-reserve", 2, "--concurrency", 1,
        "--write-scope", "outputs/goal-graph-test/child",
    )
    assert Path(child["ledger"]).is_file()
    assert run("validate", goal)["error_count"] == 0


def test_finish_requires_every_global_gate_to_pass(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-gate", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-gate"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    rejected = run("finish", goal, ok=False)
    assert "global gate" in rejected["error"]
    passed = run("pass-gate", goal, "--gate-id", "GATE-01", "--evidence", goal / "goal-ledger.yaml")
    assert passed["status"] == "PASS"
    assert run("finish", goal)["status"] == "PASS"


def test_usage_recording_preserves_the_verification_reserve(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path, goal_effort=100)
    rejected = run(
        "record-usage", goal / "goal-ledger.yaml", "--agent-id", "worker-1", "--seat", "luna-worker",
        "--category", "execution", "--wall-minutes", 1, "--effort-minutes", 81,
        "--evidence", goal / "goal-ledger.yaml", ok=False,
    )
    assert "verification reserve" in rejected["error"]


def test_usage_category_cannot_impersonate_a_verifier(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path, goal_effort=100)
    rejected = run(
        "record-usage", goal / "goal-ledger.yaml", "--agent-id", "worker-1", "--seat", "luna-worker",
        "--category", "verification", "--wall-minutes", 1, "--effort-minutes", 95,
        "--evidence", goal / "goal-ledger.yaml", ok=False,
    )
    assert "cannot record verification" in rejected["error"]


def test_validate_rejects_phantom_global_gate_evidence(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run(
        "init", "--state-root", state, "--goal-id", "G-20260802-phantom", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test",
    )
    goal = state / "G-20260802-phantom"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["global_gates"][0]["status"] = "PASS"
    data["global_gates"][0]["evidence"] = ["does/not/exist.txt"]
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = run("validate", goal, ok=False)
    assert any("evidence path does not exist" in error for error in result["errors"])
    assert "validation failed" in run("finish", goal, ok=False)["error"]


def test_status_summary_exposes_gates_and_unresolved_without_the_node_dump(tmp_path: Path) -> None:
    """The old status omitted gates and unresolved — the two things `finish` is gated on."""
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")

    full = run("status", goal)
    assert [gate["gate_id"] for gate in full["gates"]] == ["GATE-01"]
    assert full["unresolved"] == [] and full["verdict"] is None
    assert "nodes" in full

    summary = run("status", goal, "--summary")
    assert "nodes" not in summary, "the summary must not re-dump every node"
    assert summary["node_total"] == 2 and summary["node_counts"]["DRAFT"] == 2
    assert [gate["gate_id"] for gate in summary["gates"]] == ["GATE-01"]
    assert summary["attention"] == [] and summary["ready_total"] == 2  # SP-01 and SP-01-L01


def test_doctor_flags_a_finishable_goal_that_was_never_finished(tmp_path: Path) -> None:
    goal = init_goal(tmp_path, "G-20260802-stranded", "gate")
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    assert not any(f["check"] == "goal-not-closed" for f in run("doctor", goal)["findings"])

    run("pass-gate", goal, "--gate-id", "GATE-01", "--evidence", goal / "goal-ledger.yaml")
    stranded = run("doctor", goal, ok=False)
    assert stranded["finding_counts"]["error"] == 1
    assert any(f["check"] == "goal-not-closed" for f in stranded["findings"])

    run("finish", goal)
    assert run("doctor", goal)["finding_counts"]["error"] == 0


def test_doctor_flags_a_pass_whose_frozen_evidence_path_does_not_exist(tmp_path: Path) -> None:
    """A criterion can name an evidence_path that is prose, not a path; PASS then rests on assertion."""
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    edit_yaml(
        goal / "subprojects/SP-01/tasks/SP-01-L01/orders/001.yaml",
        lambda d: d["success_criteria"][0].update(evidence_path="감사 출력(result payload에 수치 기록)"),
    )
    mark_first_child_pass(goal)

    findings = run("doctor", goal, ok=False)["findings"]
    phantom = [f for f in findings if f["check"] == "evidence-path-missing"]
    assert phantom and phantom[0]["sample"] == ["SP-01-L01:C1"]


def test_doctor_is_read_only_and_fail_on_sets_the_exit_code(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    before = sha_tree(goal)

    clean = run("doctor", goal)
    assert clean["finding_counts"]["error"] == 0
    assert run("doctor", goal, "--fail-on", "never")["ok"] is True
    tripped = run("doctor", goal, "--fail-on", "warn", ok=False)
    assert tripped["finding_counts"]["warn"] >= 1

    assert sha_tree(goal) == before, "doctor must never mutate the ledger tree"


def test_doctor_resolves_evidence_paths_independently_of_the_caller_cwd(tmp_path: Path) -> None:
    """A workspace-relative evidence path must not read as missing just because the reader moved."""
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    real = ROOT / "outputs/goal-graph-test/a.txt"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("evidence", encoding="utf-8")
    mark_first_child_pass(goal)
    try:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        for cwd in (ROOT, elsewhere):
            findings = run("doctor", goal, "--fail-on", "never", cwd=cwd)["findings"]
            assert not [f for f in findings if f["check"] == "evidence-path-missing"], f"from {cwd}"
    finally:
        real.unlink()


def test_record_and_resolve_unresolved_gate_the_finish(tmp_path: Path) -> None:
    """An unresolved item is the honest way to keep a caveat on a goal whose nodes passed."""
    goal = init_goal(tmp_path, "G-20260901-unres", "caveat")
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "final audit")
    run("pass-gate", goal, "--gate-id", "GATE-01", "--evidence", goal / "goal-ledger.yaml")

    added = run(
        "record-unresolved", goal, "--item-id", "U-01", "--owner", "sol",
        "--statement", "deliverable is not on disk", "--evidence", goal / "goal-ledger.yaml",
    )
    assert added["unresolved_count"] == 1
    assert "duplicate unresolved item" in run(
        "record-unresolved", goal, "--item-id", "U-01", "--owner", "sol",
        "--statement", "again", ok=False,
    )["error"]

    assert "unresolved items prevent finish" in run("finish", goal, ok=False)["error"]
    assert [i["item_id"] for i in run("status", goal, "--summary")["unresolved"]] == ["U-01"]
    assert any(f["check"] == "unresolved-open" for f in run("doctor", goal, "--fail-on", "never")["findings"])
    assert run("validate", goal)["error_count"] == 0

    assert "unknown unresolved item" in run(
        "resolve-unresolved", goal, "--item-id", "NOPE", "--resolution", "x", ok=False,
    )["error"]
    assert run("resolve-unresolved", goal, "--item-id", "U-01", "--resolution", "closed")["unresolved_count"] == 0
    assert run("finish", goal)["status"] == "PASS"


def test_record_unresolved_rejects_phantom_evidence(tmp_path: Path) -> None:
    goal = init_goal(tmp_path, "G-20260901-unres2", "caveat")
    rejected = run(
        "record-unresolved", goal, "--item-id", "U-01", "--owner", "sol",
        "--statement", "s", "--evidence", "does/not/exist.md", ok=False,
    )
    assert "evidence path does not exist" in rejected["error"]
    assert not yaml.safe_load((goal / "goal-ledger.yaml").read_text())["unresolved"]


def test_independent_verifier_fail_overrides_a_worker_pass_self_report(tmp_path: Path) -> None:
    """The case the verification architecture exists for: worker self-reports PASS,
    the independent verifier finds a frozen criterion unmet. Before this was fixed the
    node was stranded in VERIFYING with every exit refused (observed live on
    SP-06-L01 of G-20260830-paper-position, 2026-09-01)."""
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    pass_result(attempt)
    fail_review(attempt)

    # PASS must still be refused -- the gate itself was never the problem.
    assert run(
        "transition", terra, "--node", "SP-01-L01", "--to", "VERIFYING",
        "--evidence", attempt / "result.yaml",
    )["new"] == "VERIFYING"
    blocked = run(
        "transition", terra, "--node", "SP-01-L01", "--to", "PASS",
        "--evidence", attempt / "result.yaml", ok=False,
    )
    assert "independent passing" in blocked["error"]

    # The verifier's FAIL must open the repair loop.
    record_feedback(attempt)
    run("transition", terra, "--node", "SP-01-L01", "--to", "FAIL")
    run("reissue-order", terra, "--task-id", "SP-01-L01", "--feedback", attempt / "feedback.yaml")
    second = start(terra, agent="worker-2")
    assert second.name == "002"


def test_worker_pass_with_only_passing_reviews_still_cannot_be_failed(tmp_path: Path) -> None:
    """The new demotion path must not become a general escape hatch."""
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    pass_result(attempt)
    pass_review(attempt)
    rejected = record_feedback(attempt, ok=False)
    assert "PASS attempts" in rejected["error"]


def test_worker_cannot_self_demote_with_its_own_fail_review(tmp_path: Path) -> None:
    """Demotion requires a DISTINCT reviewer, mirroring the promotion path."""
    _, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--node", "SP-01-L01", "--to", "READY")
    attempt = start(terra)
    pass_result(attempt)
    rejected = fail_review(attempt, review_id="self-fail", agent="worker-1", ok=False)
    assert "must differ from author" in rejected["error"]
    # No review exists, so the demotion path stays shut.
    still_shut = record_feedback(attempt, ok=False)
    assert "PASS attempts" in still_shut["error"]


# ---------------------------------------------------------------------------
# amend-schedule (DU-01 (a1), PPC 2026-11-03): schedule-amend verb with a durable record
# ---------------------------------------------------------------------------

def _amend(target: Path, *extra: object, ok: bool = True) -> dict:
    return run("amend-schedule", target, *extra, ok=ok)


def test_amend_schedule_terra_updates_both_books_and_records(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    result = _amend(terra, "--expected-revision", 0, "--expected-parent-revision", 1,
                    "--effort-minutes", 70, "--verification-reserve", 14, "--reason", "round 2 sizing")
    assert result["revision"] == 1 and result["parent"]["revision_to"] == 2
    t = yaml.safe_load(terra.read_text())
    g = yaml.safe_load((goal / "goal-ledger.yaml").read_text())
    assert t["schedule"]["agent_effort_budget_minutes"] == 70
    assert t["schedule"]["verification_reserve_minutes"] == 14
    assert g["graph"]["nodes"][0]["allocation"]["agent_effort_budget_minutes"] == 70
    (record,) = t["schedule_amendments"]
    assert record["fields"]["agent_effort_budget_minutes"] == {"from": 60, "to": 70}
    assert record["reason"] == "round 2 sizing" and record["revision_from"] == 0 and record["revision_to"] == 1
    assert record["parent"] == {"ledger_id": "G-20260802-test", "revision_from": 1, "revision_to": 2}
    run("validate", goal)


def test_amend_schedule_goal_level_records_on_goal(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path)
    result = _amend(goal, "--expected-revision", 1, "--effort-minutes", 150, "--verification-reserve", 30,
                    "--deadline", "2026-12-01T00:00:00+00:00", "--reason", "goal headroom")
    assert result["parent"] is None
    g = yaml.safe_load((goal / "goal-ledger.yaml").read_text())
    assert g["schedule"]["agent_effort_budget_minutes"] == 150 and g["revision"] == 2
    assert g["schedule_amendments"][0]["fields"]["deadline_at"]["to"] == "2026-12-01T00:00:00+00:00"
    run("validate", goal)


def test_amend_schedule_revision_guards_write_nothing(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    before = {p: p.read_bytes() for p in (terra, goal / "goal-ledger.yaml")}
    stale = _amend(terra, "--expected-revision", 5, "--expected-parent-revision", 1,
                   "--effort-minutes", 70, "--reason", "x", ok=False)
    assert "revision mismatch" in stale["error"]
    missing = _amend(terra, "--expected-revision", 0, "--effort-minutes", 70, "--reason", "x", ok=False)
    assert "--expected-parent-revision" in missing["error"]
    no_rev = _amend(terra, "--effort-minutes", 70, "--reason", "x", ok=False)
    assert "--expected-revision" in no_rev["error"]
    blank = _amend(terra, "--expected-revision", 0, "--deadline", "2026-12-01T00:00:00+00:00", "--reason", " ", ok=False)
    assert "reason" in blank["error"]
    same = _amend(terra, "--expected-revision", 0, "--expected-parent-revision", 1,
                  "--effort-minutes", 60, "--reason", "x", ok=False)
    assert "no field changes" in same["error"]
    assert all(p.read_bytes() == data for p, data in before.items())
    # deadline-only Terra amendment touches no mirrored allocation, so no parent revision is needed
    _amend(terra, "--expected-revision", 0, "--deadline", "2026-12-01T00:00:00+00:00", "--reason", "x")
    assert (goal / "goal-ledger.yaml").read_bytes() == before[goal / "goal-ledger.yaml"]


@pytest.mark.parametrize("extra, fragment", [
    (("--effort-minutes", 100, "--verification-reserve", 20), "subproject effort allocations consume verification reserve"),
    (("--verification-reserve", 5), "verification reserve is below 20%"),
])
def test_amend_schedule_refuses_new_validate_errors_and_restores(tmp_path: Path, extra, fragment) -> None:
    goal, terra = scaffold(tmp_path)
    before = {p: p.read_bytes() for p in (terra, goal / "goal-ledger.yaml")}
    refused = _amend(terra, "--expected-revision", 0, "--expected-parent-revision", 1, *extra,
                     "--reason", "over-allocate", ok=False)
    assert "files restored" in refused["error"] and fragment in refused["error"]
    assert all(p.read_bytes() == data for p, data in before.items())


def test_amend_schedule_refuses_terra_deadline_past_goal(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    _amend(goal, "--expected-revision", 1, "--deadline", "2026-10-01T00:00:00+00:00", "--reason", "goal deadline")
    before = terra.read_bytes()
    refused = _amend(terra, "--expected-revision", 0, "--deadline", "2026-10-22T00:00:00+00:00", "--reason", "late", ok=False)
    assert "deadline exceeds goal" in refused["error"]
    assert terra.read_bytes() == before


def test_amend_schedule_refuses_pass_node(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["graph"]["nodes"][0]["status"] = "PASS"
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    refused = _amend(terra, "--expected-revision", 0, "--deadline", "2026-12-01T00:00:00+00:00", "--reason", "x", ok=False)
    assert "frozen" in refused["error"]


def test_amend_schedule_clears_child_overflow_rehearsal(tmp_path: Path) -> None:
    """The PPC round-2 shape: orders overflow a Terra's execution budget; the amendment clears it."""
    goal, terra = scaffold(tmp_path, terra_effort=40, terra_reserve=8)
    goal_path = goal / "goal-ledger.yaml"
    for index in range(1, 4):
        add_task(terra, f"SP-01-L0{index}", f"outputs/goal-graph-test/{index}.txt")
    overflow = run("validate", goal, ok=False)
    assert any("child effort consumes verification reserve" in e for e in overflow["errors"])
    t_rev = yaml.safe_load(terra.read_text())["revision"]
    g_rev = yaml.safe_load(goal_path.read_text())["revision"]
    result = _amend(terra, "--expected-revision", t_rev, "--expected-parent-revision", g_rev,
                    "--effort-minutes", 60, "--verification-reserve", 12, "--reason", "fit round-2 orders")
    assert any("child effort consumes verification reserve" in e for e in result["cleared_errors"])
    run("validate", goal)


def test_amend_schedule_input_guards(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    before = {p: p.read_bytes() for p in (terra, goal / "goal-ledger.yaml")}
    zero = _amend(terra, "--expected-revision", 0, "--verification-reserve", 0, "--reason", "x", ok=False)
    assert ">= 1" in zero["error"]
    bad = _amend(terra, "--expected-revision", 0, "--deadline", "next tuesday", "--reason", "x", ok=False)
    assert "invalid ISO-8601" in bad["error"]
    stale_parent = _amend(terra, "--expected-revision", 0, "--expected-parent-revision", 9,
                          "--effort-minutes", 70, "--reason", "x", ok=False)
    assert "revision mismatch" in stale_parent["error"]
    assert all(p.read_bytes() == data for p, data in before.items())


def test_amend_schedule_refuses_pass_goal(tmp_path: Path) -> None:
    goal, _ = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    data = yaml.safe_load(goal_path.read_text())
    data["status"] = "PASS"
    goal_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    refused = _amend(goal, "--expected-revision", 1, "--effort-minutes", 150, "--reason", "x", ok=False)
    assert "frozen" in refused["error"]


def test_amend_schedule_nested_child_terra_mirrors_parent_terra(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    run("add-child-terra", terra, "--subproject-id", "SP-01-T01", "--owner", "terra-child",
        "--objective", "nested", "--wall-minutes", 20, "--effort-minutes", 20, "--verification-reserve", 4,
        "--write-scope", "outputs/goal-graph-test")
    child = next(goal.glob("subprojects/**/SP-01-T01/ledger.yaml"))
    p_rev = yaml.safe_load(terra.read_text())["revision"]
    goal_before = (goal / "goal-ledger.yaml").read_bytes()
    result = _amend(child, "--expected-revision", 0, "--expected-parent-revision", p_rev,
                    "--effort-minutes", 25, "--verification-reserve", 5, "--reason", "nested")
    assert result["parent"]["ledger_id"] == "SP-01"
    parent = yaml.safe_load(terra.read_text())
    entry = next(c for c in parent["children"] if c.get("subproject_id") == "SP-01-T01")
    assert entry["allocation"]["agent_effort_budget_minutes"] == 25
    assert (goal / "goal-ledger.yaml").read_bytes() == goal_before


def test_amend_schedule_restores_both_files_when_parent_write_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    goal, terra = scaffold(tmp_path)
    goal_path = goal / "goal-ledger.yaml"
    before = {p: p.read_bytes() for p in (terra, goal_path)}
    module = load_ledger_module()
    real = module.atomic_yaml
    calls = {"n": 0}

    def flaky(path, data, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated parent write failure")
        return real(path, data, **kw)

    monkeypatch.setattr(module, "atomic_yaml", flaky)
    args = module.parser().parse_args([
        "amend-schedule", str(terra), "--expected-revision", "0", "--expected-parent-revision", "1",
        "--effort-minutes", "70", "--verification-reserve", "14", "--reason", "x"])
    with pytest.raises(OSError):
        args.func(args)
    assert all(p.read_bytes() == data for p, data in before.items())


# ---------------------------------------------------------------------------
# supersede + amend-gate (PPC 2026-11-05, user ruling "D-27 중단, DU-06 (i), P1로 간다"):
# spec §13 SKIPPED made reachable with a durable record; gate criterion amendment with a record.
# Only PASS satisfies a dependency (spec §13) — unchanged.
# ---------------------------------------------------------------------------

def _supersede(terra: Path, node: str, *extra: object, rev: int = 1, ok: bool = True) -> dict:
    return run("supersede", terra, "--expected-revision", rev, "--node", node,
               "--reason", "user ruling", "--impact", "no downstream output", *extra, ok=ok)


def test_supersede_draft_task_records_and_validates(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    result = _supersede(terra, "SP-01-L01", "--evidence", goal / "goal-ledger.yaml")
    assert result["old"] == "DRAFT" and result["new"] == "SKIPPED" and result["revision"] == 2
    t = yaml.safe_load(terra.read_text())
    assert t["children"][0]["status"] == "SKIPPED"
    (rec,) = t["supersessions"]
    assert rec["node_id"] == "SP-01-L01" and rec["from_status"] == "DRAFT" and rec["node_type"] == "luna"
    assert rec["superseded_by"] is None and rec["revision_from"] == 1 and rec["revision_to"] == 2
    assert rec["impact"] == "no downstream output"
    assert run("validate", goal)["error_count"] == 0


def test_supersede_guards_write_nothing(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    before = terra.read_bytes()
    ev = ("--evidence", goal / "goal-ledger.yaml")
    assert "revision" in _supersede(terra, "SP-01-L01", *ev, rev=7, ok=False)["error"]
    assert "--evidence" in _supersede(terra, "SP-01-L01", ok=False)["error"]
    assert "unknown child" in _supersede(terra, "SP-01-L99", *ev, ok=False)["error"]
    assert "itself" in _supersede(terra, "SP-01-L01", *ev, "--superseded-by", "SP-01-L01", ok=False)["error"]
    assert "missing" in _supersede(terra, "SP-01-L01", *ev, "--superseded-by", "SP-01-L77", ok=False)["error"]
    blank = run("supersede", terra, "--expected-revision", 1, "--node", "SP-01-L01", "--reason", " ",
                "--impact", "x", *ev, ok=False)
    assert "--reason" in blank["error"]
    assert terra.read_bytes() == before


def test_supersede_refuses_started_or_finished_children(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    run("transition", terra, "--expected-revision", 1, "--node", "SP-01-L01", "--to", "READY")
    start(terra)
    rejected = _supersede(terra, "SP-01-L01", "--evidence", goal / "goal-ledger.yaml", rev=3, ok=False)
    assert "only DRAFT/READY/BLOCKED" in rejected["error"]


def test_skipped_never_satisfies_a_dependency(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra, "SP-01-L02", "outputs/goal-graph-test/b.txt", "SP-01-L01")
    _supersede(terra, "SP-01-L01", "--evidence", goal / "goal-ledger.yaml", rev=2)
    assert "SP-01-L02" not in run("ready", terra)["ready"]
    refused = run("transition", terra, "--expected-revision", 3, "--node", "SP-01-L02", "--to", "READY", ok=False)
    assert "dependencies must PASS" in refused["error"]


def test_supersede_with_replacement_records_successor(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra, "SP-01-L01R", "outputs/goal-graph-test/a2.txt")
    _supersede(terra, "SP-01-L01", "--evidence", goal / "goal-ledger.yaml", "--superseded-by", "SP-01-L01R", rev=2)
    t = yaml.safe_load(terra.read_text())
    assert t["supersessions"][0]["superseded_by"] == "SP-01-L01R"
    refused = _supersede(terra, "SP-01-L01R", "--evidence", goal / "goal-ledger.yaml",
                         "--superseded-by", "SP-01-L01", rev=3, ok=False)
    assert "SKIPPED" in refused["error"]
    assert run("validate", goal)["error_count"] == 0


def test_validate_flags_skipped_without_record_and_orphan_record(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    add_task(terra, "SP-01-L02", "outputs/goal-graph-test/b.txt")
    t = yaml.safe_load(terra.read_text())
    t["children"][0]["status"] = "SKIPPED"
    t["supersessions"] = [{
        "superseded_at": "2026-11-05T00:00:00+00:00", "node_id": "SP-01-L02", "node_type": "luna",
        "from_status": "DRAFT", "reason": "r", "impact": "i", "evidence": ["x"], "superseded_by": None,
        "revision_from": 1, "revision_to": 2,
    }]
    terra.write_text(yaml.safe_dump(t, sort_keys=False))
    errors = run("validate", goal, ok=False)["errors"]
    assert any("SP-01-L01 is SKIPPED without a supersession record" in e for e in errors)
    assert any("supersession record for SP-01-L02 does not match" in e for e in errors)


def test_subproject_pass_accepts_recorded_skip_only(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    gl = goal / "goal-ledger.yaml"
    for rev, to in ((1, "READY"), (2, "RUNNING"), (3, "VERIFYING")):
        run("transition", gl, "--expected-revision", rev, "--node", "SP-01", "--to", to)
    run("record-terra-review", terra.parent, "--review-id", "r1", "--agent-id", "terra-verifier-1",
        "--actual-model", "m", "--verdict", "PASS", "--form-check", "f", "--intent-check", "i", "--evidence", gl)
    review = terra.parent / "reports/terra-review-r1.yaml"
    blocked = run("transition", gl, "--expected-revision", 4, "--node", "SP-01", "--to", "PASS",
                  "--evidence", review, ok=False)
    assert "every child" in blocked["error"]
    _supersede(terra, "SP-01-L01", "--evidence", gl)
    assert run("transition", gl, "--expected-revision", 4, "--node", "SP-01", "--to", "PASS",
               "--evidence", review)["new"] == "PASS"
    assert run("validate", goal)["error_count"] == 0


def test_supersede_restores_file_when_validate_gains_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    module = load_ledger_module()
    real = module.validate_root
    calls = {"n": 0}

    def fake(root):
        calls["n"] += 1
        return real(root) + (["injected"] if calls["n"] > 1 else [])

    monkeypatch.setattr(module, "validate_root", fake)
    before = terra.read_bytes()
    args = module.parser().parse_args(["supersede", str(terra), "--expected-revision", "1", "--node", "SP-01-L01",
                                       "--reason", "r", "--impact", "i", "--evidence", str(goal / "goal-ledger.yaml")])
    with pytest.raises(module.LedgerError, match="injected"):
        args.func(args)
    assert terra.read_bytes() == before


def test_amend_gate_records_and_keeps_dependencies(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run("init", "--state-root", state, "--goal-id", "G-20260802-gate", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test")
    goal = state / "G-20260802-gate"
    gl = goal / "goal-ledger.yaml"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "first")
    run("add-gate", goal, "--gate-id", "GATE-02", "--criterion", "check ran once", "--depends-on", "GATE-01")
    rev = yaml.safe_load(gl.read_text())["revision"]
    result = run("amend-gate", goal, "--expected-revision", rev, "--gate-id", "GATE-02",
                 "--criterion", "stop recorded", "--reason", "user ruling", "--evidence", gl)
    g = yaml.safe_load(gl.read_text())
    gate = next(x for x in g["global_gates"] if x["gate_id"] == "GATE-02")
    assert gate["criterion"] == "stop recorded" and gate["status"] == "DRAFT" and gate["dependencies"] == ["GATE-01"]
    (rec,) = g["gate_amendments"]
    assert rec["criterion"] == {"from": "check ran once", "to": "stop recorded"} and rec["revision_to"] == result["revision"]
    assert "dependencies must PASS" in run("pass-gate", goal, "--expected-revision", result["revision"],
                                           "--gate-id", "GATE-02", "--evidence", gl, ok=False)["error"]
    assert run("validate", goal)["error_count"] == 0


def test_amend_gate_guards(tmp_path: Path) -> None:
    state = tmp_path / "state"
    run("init", "--state-root", state, "--goal-id", "G-20260802-gate", "--objective", "gate",
        "--effort-minutes", 100, "--verification-reserve", 20, "--write-scope", "outputs/goal-graph-test")
    goal = state / "G-20260802-gate"
    gl = goal / "goal-ledger.yaml"
    run("add-gate", goal, "--gate-id", "GATE-01", "--criterion", "first")
    rev = yaml.safe_load(gl.read_text())["revision"]
    before = gl.read_bytes()
    base = ("amend-gate", goal, "--gate-id", "GATE-01", "--reason", "r")
    assert "--evidence" in run(*base, "--expected-revision", rev, "--criterion", "x", ok=False)["error"]
    assert "unchanged" in run(*base, "--expected-revision", rev, "--criterion", "first", "--evidence", gl, ok=False)["error"]
    assert "unknown global gate" in run("amend-gate", goal, "--gate-id", "GATE-09", "--reason", "r", "--expected-revision", rev,
                                        "--criterion", "x", "--evidence", gl, ok=False)["error"]
    assert gl.read_bytes() == before
    run("pass-gate", goal, "--expected-revision", rev, "--gate-id", "GATE-01", "--evidence", gl)
    assert "only a DRAFT gate" in run(*base, "--expected-revision", rev + 1, "--criterion", "x", "--evidence", gl, ok=False)["error"]


def test_supersede_refuses_a_node_with_attempt_history(tmp_path: Path) -> None:
    goal, terra = scaffold(tmp_path)
    add_task(terra, "SP-01-L01", "outputs/goal-graph-test/a.txt")
    t = yaml.safe_load(terra.read_text())
    t["children"][0]["status"] = "READY"
    t["children"][0]["current_attempt"] = 1
    terra.write_text(yaml.safe_dump(t, sort_keys=False))
    rejected = _supersede(terra, "SP-01-L01", "--evidence", goal / "goal-ledger.yaml", ok=False)
    assert "attempt history" in rejected["error"]
