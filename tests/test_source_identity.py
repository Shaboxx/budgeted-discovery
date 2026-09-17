"""Source provenance must never inherit an enclosing repository identity."""
import json
import subprocess
import pytest
from frontier_bench import runner


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "standalone"
    package = root / "src" / "frontier_bench"
    package.mkdir(parents=True)
    module = package / "runner.py"
    module.write_text("# independent source\n")
    (root / "source-provenance.json").write_text(json.dumps({
        "revision": "standalone-uncommitted", "dirty": True
    }))
    monkeypatch.setattr(runner, "__file__", str(module))
    return root


def test_source_identity_uses_own_repository(source_tree, monkeypatch):
    def git(command, **kwargs):
        assert kwargs["cwd"] == source_tree
        return {
            ("rev-parse", "--show-toplevel"): str(source_tree),
            ("rev-parse", "HEAD"): "own-revision",
            ("status", "--porcelain"): "",
        }[tuple(command[1:])]
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    result = runner.source_identity()
    assert result["revision"] == "own-revision"
    assert result["dirty"] is False
    assert len(result["source_hash"]) == 64


def test_source_identity_rejects_enclosing_repository(source_tree, monkeypatch):
    calls = []
    def git(command, **kwargs):
        calls.append(command)
        assert command == ["git", "rev-parse", "--show-toplevel"]
        return str(source_tree.parent)
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    result = runner.source_identity()
    assert result["revision"] == "standalone-uncommitted"
    assert len(calls) == 1


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.CalledProcessError(128, "git")])
def test_source_identity_without_git_or_commit(source_tree, monkeypatch, failure):
    def unavailable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(runner.subprocess, "check_output", unavailable)
    assert runner.source_identity()["revision"] == "standalone-uncommitted"
