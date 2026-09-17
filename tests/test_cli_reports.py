"""Independent actual CLI/report/import checks on portable, bounded control worlds."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest
import yaml

from frontier_bench.reporting import report
from frontier_bench.runner import compare, read_jsonl, run_one
from frontier_bench.schemas import canonical
from frontier_bench.world import load_world
from test_runner_integrity import fixture_run, load
from test_stateful_integrity import audit_world


def cli(*args):
    return subprocess.run([sys.executable, "-m", "frontier_bench.cli", *map(str, args)],
                          text=True, capture_output=True, timeout=40)


def config_file(tmp_path, **experiment):
    world, cfg = fixture_run(tmp_path, "query")
    cfg["experiment"].update(world_path=str(world), seeds=[91], arrangements=["weak"],
                             conditions=["evolving_restricted"], policies=["fixed", "query"],
                             max_runs=2)
    cfg["experiment"].update(experiment)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return world, cfg, path


def comparison_fixture(tmp_path):
    _, cfg, _ = config_file(tmp_path)
    output = tmp_path / "comparison"
    assert compare(cfg, output)["status"] == "complete"
    return output


def test_actual_portable_cli_run_replay_report_and_validation(tmp_path):
    world, _, cfg = config_file(tmp_path)
    doctor = cli("doctor", "--portable")
    assert doctor.returncode == 0, doctor.stderr
    assert json.loads(doctor.stdout)["ok"]
    valid = cli("validate-world", "--world", world)
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["valid"]
    output = tmp_path / "invocation"
    ran = cli("run", "--config", cfg, "--output", output)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout)["status"] == "completed"
    replayed = cli("replay", "--run", output / "run")
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout)["verified"]
    rendered = cli("report", "--runs", output / "run")
    assert rendered.returncode == 0, rendered.stderr
    html = (output / "run" / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html and "completed" in html
    assert "<script" not in html and 'src="http' not in html


def test_cli_compare_partial_budget_returns_failure_and_preserves_artifacts(tmp_path):
    _, _, cfg = config_file(tmp_path)
    output = tmp_path / "comparison"
    result = cli("compare", "--config", cfg, "--output", output, "--max-runs", 1)
    assert load(output / "comparison.json")["status"] == "partial"
    assert (output / "report.html").exists()
    assert result.returncode != 0, "partial comparison must not signal CLI success"


def test_actual_cli_complete_comparison_reports_every_run(tmp_path):
    _, _, cfg = config_file(tmp_path)
    output = tmp_path / "comparison"
    result = cli("compare", "--config", cfg, "--output", output)
    assert result.returncode == 0, result.stderr
    comparison = json.loads(result.stdout)
    assert comparison["status"] == "complete"
    assert comparison["planned"] == comparison["executed"] == 2
    summaries = load(output / "summaries.json")
    assert {row["policy"] for row in summaries} == {"fixed", "query"}
    assert load(output / "statistics.json")["contrasts"][0]["matched_pairs"] == 1
    assert (output / "report.html").exists()


def test_cli_interrupted_run_returns_failure_and_preserves_artifacts(tmp_path):
    _, _, cfg = config_file(tmp_path)
    output = tmp_path / "interrupted"
    result = cli("run", "--config", cfg, "--output", output, "--max-minutes", "1e-9")
    summary = load(output / "run" / "summary.json")
    assert summary["status"] == "interrupted"
    assert (output / "run" / "manifest.json").exists()
    assert result.returncode != 0, "interrupted run must not signal CLI success"


def test_cli_failed_policy_comparison_returns_failure(tmp_path):
    _, _, cfg = config_file(tmp_path, policies=["not-a-policy"])
    output = tmp_path / "failed"
    result = cli("compare", "--config", cfg, "--output", output)
    summary = load(output / "comparison.json")
    assert summary["status"] == "partial" and summary["failed"]
    assert (output / "report.html").exists(), "a declared setup failure must remain reportable"
    assert result.returncode != 0, "failed comparison must not signal CLI success"


def test_comparison_report_rebuild_needs_only_raw_artifacts(tmp_path):
    output = comparison_fixture(tmp_path)
    original = load(output / "summaries.json")
    for name in ("summaries.json", "summaries.parquet", "statistics.json", "derived-aggregate.json", "report.html"):
        (output / name).unlink(missing_ok=True)
    rendered = cli("report", "--runs", output)
    assert rendered.returncode == 0, rendered.stderr
    assert load(output / "summaries.json") == original
    assert (output / "summaries.parquet").exists()
    assert (output / "statistics.json").exists()
    assert (output / "derived-aggregate.json").exists()


def test_single_run_report_rejects_summary_tampering(tmp_path):
    world, cfg = fixture_run(tmp_path, "query")
    output = tmp_path / "run"
    run_one(world, cfg, output)
    summary = load(output / "summary.json")
    summary["reference_relevant_accounts"] += 1000
    (output / "summary.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="checksum|summary|artifact"):
        report(output)


@pytest.mark.parametrize("missing", ["summary.json", "evaluation.jsonl", "manifest.json"])
def test_report_cannot_silently_drop_missing_raw_run_artifact(tmp_path, missing):
    output = comparison_fixture(tmp_path)
    run = sorted((output / "runs").iterdir())[0]
    (run / missing).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="missing|artifact|summary|evaluation|No such file"):
        report(output)


def test_report_rejects_missing_whole_recorded_run(tmp_path):
    output = comparison_fixture(tmp_path)
    run = sorted((output / "runs").iterdir())[0]
    run.rename(tmp_path / "moved-outside-recorded-runs")
    with pytest.raises((ValueError, FileNotFoundError), match="missing|artifact|count|executed|run|No such file"):
        report(output)


def test_report_cannot_plot_corrupted_raw_evaluation(tmp_path):
    output = comparison_fixture(tmp_path)
    run = sorted((output / "runs").iterdir())[0]
    rows = read_jsonl(run / "evaluation.jsonl")
    rows[-1]["reference_relevant_accounts"] += 1000
    (run / "evaluation.jsonl").write_text("\n".join(canonical(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|evaluation|artifact"):
        report(output)


def test_report_does_not_trust_corrupted_derived_summaries(tmp_path):
    output = comparison_fixture(tmp_path)
    original = load(output / "summaries.json")
    rows = load(output / "summaries.json")
    rows[0]["reference_relevant_accounts"] += 1000
    (output / "summaries.json").write_text(canonical(rows), encoding="utf-8")
    report(output)
    assert load(output / "summaries.json") == original


def test_cli_rejects_nonpositive_resource_guard(tmp_path):
    _, _, cfg = config_file(tmp_path)
    output = tmp_path / "invalid"
    result = cli("compare", "--config", cfg, "--output", output, "--max-runs", 0)
    assert result.returncode != 0
    assert "positive" in result.stderr
    assert not output.exists()


def import_files(tmp_path, tick=0):
    world = audit_world(3)
    paths = {}
    inputs = {"accounts": world.accounts,
              "edges": [dict(source="acct:0", target="acct:1", type="INTERACTS_WITH", tick=tick)],
              "posts": world.posts}
    for name, rows in inputs.items():
        paths[name] = tmp_path / (name + ".jsonl")
        paths[name].write_text("\n".join(canonical(r) for r in rows) + "\n", encoding="utf-8")
    paths["targets"] = tmp_path / "targets.json"
    paths["targets"].write_text(canonical(world.truth), encoding="utf-8")
    return paths


def test_actual_cli_import_preserves_time_types_and_isolates(tmp_path):
    paths = import_files(tmp_path)
    output = tmp_path / "imported"
    argv = [arg for key, path in paths.items() for arg in ("--" + key, str(path))]
    result = cli("import-events", *argv, "--ticks", 3, "--output", output)
    assert result.returncode == 0, result.stderr
    world = load_world(output)
    assert len(world.accounts) == 6
    assert world.snapshots == [[["acct:0", "acct:1"]], [], []]
    assert world.metadata["generator"] == "imported-semi-synthetic"
    assert str(tmp_path) not in (output / "world.json").read_text(encoding="utf-8")
    edges = read_jsonl(output / "typed-events.jsonl")
    assert {e["type"] for e in edges} == {"INTERACTS_WITH", "PUBLISHED", "HAS_HASHTAG", "MENTIONS"}
    interaction = [e for e in edges if e["type"] == "INTERACTS_WITH"]
    assert interaction == [dict(source="acct:0", target="acct:1", type="INTERACTS_WITH", event_at=0, available_at=0, valid_until=1)]
    expected = load(output / "manifest.json")["files"]["typed-events.jsonl"]
    assert hashlib.sha256((output / "typed-events.jsonl").read_bytes()).hexdigest() == expected


def test_actual_cli_import_rejects_out_of_range_time(tmp_path):
    paths = import_files(tmp_path, tick=-1)
    output = tmp_path / "invalid-import"
    argv = [arg for key, path in paths.items() for arg in ("--" + key, str(path))]
    result = cli("import-events", *argv, "--ticks", 3, "--output", output)
    assert result.returncode != 0
    assert "tick" in result.stderr
    assert not (output / "manifest.json").exists()
