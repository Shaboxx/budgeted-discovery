"""Executed portable-run, replay, provenance and world-block statistics checks."""
from copy import deepcopy
from pathlib import Path
import json
import time

import pytest

from frontier_bench.config import resolve
from frontier_bench.evaluation import paired_summary
from frontier_bench.runner import compare, read_jsonl, replay, run_one, source_identity
from frontier_bench.schemas import canonical, stream_seed
from frontier_bench.world import save_world
from test_stateful_integrity import audit_world


def fixture_run(tmp_path, policy="learned", **overrides):
    world = audit_world(12)
    world.truth["target_definition"] = {"version": "fixed-reference-relevance-v1", "arrangement": "weak", "fraction": 1 / 3}
    world.metadata["config"] = {"seed": 91, "world": {"stationary": False},
                                "content": {"stationary": False}, "target": {"arrangement": "weak"}}
    cfg = resolve(dict(seed=91, initial=dict(accounts=1, vocabulary=["craft"]),
                       world=dict(stationary=False), target=dict(arrangement="weak"),
                       provider=dict(mode="restricted", failure_probability=0, page_size=2),
                       feedback=dict(mode="operational", delay=1, noise=0),
                       actions=dict(cap=24, revisit_after=1),
                       budget=dict(requests=10, cost=15.0, slots_per_tick=2, max_steps=30),
                       policy=dict(name=policy), experiment=dict(max_minutes=1, max_runs=3)))
    for group, patch in overrides.items():
        cfg[group].update(patch)
    source = save_world(world, tmp_path / "world")
    return source, cfg


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_actual_portable_run_replay_logs_and_delayed_outcomes(tmp_path):
    world, cfg = fixture_run(tmp_path)
    run = tmp_path / "run"
    summary = run_one(world, cfg, run)
    assert summary["status"] == "completed"
    assert summary["requests"] <= cfg["budget"]["requests"]
    decisions = read_jsonl(run / "decisions.jsonl")
    executions = read_jsonl(run / "executions.jsonl")
    outcomes = read_jsonl(run / "outcomes.jsonl")
    assert len(decisions) == len(executions) > 0
    assert any(o["kind"] == "assessment_revision" for o in outcomes)
    initial = {o["outcome_id"]: o for o in outcomes if o["kind"] == "initial_observed_outcome"}
    for outcome in outcomes:
        if outcome["kind"] == "assessment_revision":
            assert outcome["outcome_id"] in initial
            assert outcome["step"] >= initial[outcome["outcome_id"]]["step"]
            assert {v["id"] for v in outcome["labels"]} <= set(initial[outcome["outcome_id"]]["new_account_ids"])
    receipts = [e["info"]["receipt"] for e in executions if "receipt" in e["info"]]
    assert sum(r["execution_status"] != "admission_failed" for r in receipts) == summary["requests"]
    assert sum(r["cost"] for r in receipts) == pytest.approx(summary["cost"])
    assert summary["learned_examples"] > 0
    verified = replay(run)
    assert verified["verified"] and verified["decisions"] == len(decisions)
    assert (run / "curves.parquet").stat().st_size > 0


def test_replay_rejects_log_corruption(tmp_path):
    world, cfg = fixture_run(tmp_path, "query")
    run = tmp_path / "run"
    run_one(world, cfg, run)
    with (run / "outcomes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"unexpected":true}\n')
    with pytest.raises(ValueError, match="checksum"):
        replay(run)


def test_replay_rejects_corrupted_report_summary(tmp_path):
    world, cfg = fixture_run(tmp_path, "query")
    run = tmp_path / "run"
    run_one(world, cfg, run)
    summary = load(run / "summary.json")
    summary["reference_relevant_accounts"] += 1000
    (run / "summary.json").write_text(canonical(summary) + "\n", encoding="utf-8")
    with pytest.raises((ValueError, AssertionError), match="checksum|summary|artifact"):
        replay(run)


def test_manifest_records_actual_policy_random_stream(tmp_path):
    world, cfg = fixture_run(tmp_path, "learned")
    run = tmp_path / "run"
    run_one(world, cfg, run)
    manifest = load(run / "manifest.json")
    assert manifest["stream_ids"]["policy_exploration"] == stream_seed(cfg["seed"], "policy_exploration", "learned")


def test_exported_world_cannot_be_retagged_by_execution_configuration(tmp_path):
    world, cfg = fixture_run(tmp_path, "query", world=dict(stationary=True), target=dict(arrangement="concentrated"))
    try:
        summary = run_one(world, cfg, tmp_path / "run")
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("world", "condition", "arrangement", "mismatch"))
        return
    assert summary["condition"] == "evolving_restricted"
    assert summary["arrangement"] == "weak"


def contrast_row(policy, value, seed=1, condition="evolving_restricted", world_hash="shared-world", **extra):
    return dict(policy=policy, status="completed", seed=seed, condition=condition,
                arrangement="weak", sparse_probes=True, use_motif=False, regime="test",
                world_hash=world_hash, reference_relevant_accounts=value, **extra)


def test_pairing_requires_same_world_artifact():
    rows = [contrast_row("fixed", 0, world_hash="world-left"),
            contrast_row("learned", 100, world_hash="world-right")]
    contrast = paired_summary(rows, samples=30)["contrasts"][0]
    assert contrast["matched_pairs"] == 0
    assert contrast["mean_paired_difference"] is None


def test_paired_statistics_resample_seed_blocks_not_conditions():
    rows = []
    for seed in (1, 2):
        for condition in ("evolving_complete", "evolving_restricted"):
            rows += [contrast_row("fixed", 3, seed, condition, world_hash=f"world-{seed}"),
                     contrast_row("learned", 3 + seed, seed, condition, world_hash=f"world-{seed}")]
    result = paired_summary(rows, samples=40, seed=33)["contrasts"][0]
    assert result["matched_pairs"] == 4
    assert result["seed_blocks"] == 2
    assert result["mean_paired_difference"] == 1.5
    assert result["interval"] is not None
    assert paired_summary(rows, samples=40, seed=33) == paired_summary(rows, samples=40, seed=33)


def test_duplicate_baseline_pair_is_not_silently_last_write_wins():
    rows = [contrast_row("fixed", 0), contrast_row("fixed", 100), contrast_row("learned", 3)]
    try:
        result = paired_summary(rows, samples=20)
    except ValueError as error:
        assert any(word in str(error).lower() for word in ("duplicate", "ambiguous", "pair"))
        return
    reverse = paired_summary(list(reversed(rows)), samples=20)
    assert result == reverse


def test_policy_seed_repeats_on_one_world_are_not_independent_world_blocks():
    rows = []
    for policy_seed in (1, 2):
        rows.extend([contrast_row("fixed", 1, policy_seed, world_hash="one-fixed-world", world_seed=91),
                     contrast_row("learned", 1 + policy_seed, policy_seed, world_hash="one-fixed-world", world_seed=91)])
    result = paired_summary(rows, samples=30)["contrasts"][0]
    assert result["matched_pairs"] == 2
    assert result["seed_blocks"] == 1
    assert result["interval"] is None


def test_same_generation_seed_across_world_conditions_is_one_block():
    rows = []
    for condition in ("stationary_complete", "evolving_restricted"):
        rows.extend([contrast_row("fixed", 1, 91, condition, world_hash=f"artifact:{condition}", world_seed=32),
                     contrast_row("learned", 3, 91, condition, world_hash=f"artifact:{condition}", world_seed=32)])
    result = paired_summary(rows, samples=30)["contrasts"][0]
    assert result["matched_pairs"] == 2
    assert result["seed_blocks"] == 1


def test_unknown_generation_seed_falls_back_to_world_artifact_block():
    rows = []
    for policy_seed in (11, 12):
        rows.extend([contrast_row("fixed", 1, policy_seed, world_hash="imported-world"),
                     contrast_row("learned", 2, policy_seed, world_hash="imported-world")])
    result = paired_summary(rows, samples=30)["contrasts"][0]
    assert result["matched_pairs"] == 2
    assert result["seed_blocks"] == 1
    assert result["interval"] is None


def test_source_identity_does_not_assume_editable_src_layout(tmp_path, monkeypatch):
    import frontier_bench.runner as runner_module

    installed = tmp_path / "site-packages" / "frontier_bench"
    installed.mkdir(parents=True)
    runtime = installed / "runner.py"
    runtime.write_text("# synthetic installed-layout identity control\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "__file__", str(runtime))
    first = source_identity()
    assert first["files"], "installed package source identity cannot be the digest of an empty tree"
    runtime.write_text("# changed installed-layout identity control\n", encoding="utf-8")
    assert source_identity()["source_hash"] != first["source_hash"]


def test_expired_deadline_records_interruption_and_replays_prefix(tmp_path):
    world, cfg = fixture_run(tmp_path, "random")
    run = tmp_path / "run"
    summary = run_one(world, cfg, run, deadline=time.perf_counter() - 1)
    assert summary["status"] == "interrupted"
    assert summary["requests"] == 0
    assert replay(run)["decisions"] == 0


def test_standalone_run_honors_configured_time_guard(tmp_path):
    world, cfg = fixture_run(tmp_path, "random", experiment=dict(max_minutes=1e-9))
    summary = run_one(world, cfg, tmp_path / "run")
    assert summary["status"] == "interrupted"
    assert "watchdog" in summary["failure"] or "time" in summary["failure"]


def test_compare_honors_max_runs_with_exported_world(tmp_path):
    world, cfg = fixture_run(tmp_path, "random")
    cfg["experiment"].update(world_path=str(world), seeds=[91], arrangements=["weak"],
                              conditions=["evolving_restricted"], policies=["random", "query"], max_runs=1)
    report = compare(cfg, tmp_path / "comparison")
    assert report["executed"] == 1
    assert any(x["reason"] == "max-runs" for x in report["skipped"])


def test_storage_safeguard_interrupts_a_single_growing_run(tmp_path):
    world, cfg = fixture_run(tmp_path, "random")
    cfg["experiment"].update(world_path=str(world), seeds=[91], arrangements=["weak"],
                              conditions=["evolving_restricted"], policies=["random"],
                              max_storage_mb=0.02, max_runs=1)
    output = tmp_path / "comparison"
    report = compare(cfg, output)
    summaries = load(output / "summaries.json")
    if summaries:
        assert not all(row["status"] == "completed" for row in summaries)
        assert any("storage" in (row.get("failure") or "").lower() for row in summaries)
    else:
        assert any("storage" in row["reason"] for row in report["skipped"])
