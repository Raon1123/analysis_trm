import json
from pathlib import Path

import pytest

from research_mcp.index import ResearchError, ResearchIndex
from research_mcp.server import MCPServer


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    exp = tmp_path / "lab" / "experiments"
    exp.mkdir(parents=True)
    (exp / "EXP-001_demo.md").write_text("---\nid: EXP-001\nstatus: running\n---\n# Demo\n\nAccuracy is 0.95.\n", encoding="utf-8")
    (tmp_path / "lab" / "registry").mkdir()
    (tmp_path / "lab" / "registry" / "EXPERIMENTS.md").write_text("# Registry\nEXP-001 running\n", encoding="utf-8")
    return tmp_path


def test_search_and_experiment(repo: Path) -> None:
    index = ResearchIndex(repo)
    assert index.search("Accuracy")["matches"][0]["line"] == 7
    record = index.experiment("1")
    assert record["experiment_id"] == "EXP-001"
    assert record["frontmatter"]["status"] == "running"


def test_search_ignores_stopwords_and_vendor_cache(repo: Path) -> None:
    cached = repo / "lab" / "theory" / ".lake" / "packages" / "x"
    cached.mkdir(parents=True)
    (cached / "README.md").write_text("EXP-001 inconclusive " * 20, encoding="utf-8")
    result = ResearchIndex(repo).search("Why was EXP-001 running?")
    assert result["matches"][0]["path"] == "lab/experiments/EXP-001_demo.md"
    assert all(".lake" not in match["path"] for match in result["matches"])


def test_explicit_experiment_id_prioritizes_owning_document(repo: Path) -> None:
    other = repo / "lab" / "reports" / "other.md"
    other.parent.mkdir()
    other.write_text(("EXP-001 inconclusive\n" * 20), encoding="utf-8")
    result = ResearchIndex(repo).search("Why was EXP-001 inconclusive?")
    assert result["matches"][0]["path"] == "lab/experiments/EXP-001_demo.md"


def test_path_escape_is_blocked(repo: Path) -> None:
    with pytest.raises(ResearchError):
        ResearchIndex(repo).read("../secret.md")


def test_manifest_comparison(repo: Path) -> None:
    index = ResearchIndex(repo)
    manifest = index.manifest()
    assert index.compare_manifest(manifest)["in_sync"]
    manifest["files"]["lab/experiments/EXP-001_demo.md"]["sha256"] = "changed"
    assert index.compare_manifest(manifest)["changed"] == ["lab/experiments/EXP-001_demo.md"]


def test_mcp_initialize_and_tools(repo: Path) -> None:
    server = MCPServer(repo)
    response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
    assert response["result"]["protocolVersion"] == "2025-11-25"
    assert response["result"]["capabilities"]["tools"] == {"listChanged": False}
    called = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_experiment", "arguments": {"experiment_id": "EXP-001"}}})
    assert called["result"]["structuredContent"]["experiment_id"] == "EXP-001"


def test_discussion_write_is_opt_in(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRM_MCP_ALLOW_WRITES", raising=False)
    with pytest.raises(ResearchError):
        ResearchIndex(repo).record_discussion(topic="x", message="y", author="z")
