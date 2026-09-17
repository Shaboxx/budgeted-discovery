"""World content identity and feedback-mode isolation contracts.

Both checks use the labeled deterministic control worlds from the other test
modules; they are not TADC-SBM acceptance evidence.
"""
from copy import deepcopy
from dataclasses import asdict
import json

from frontier_bench.environment import FrontierEnv
from frontier_bench.runner import run_one
from frontier_bench.schemas import canonical
from frontier_bench.world import load_world, save_world, world_content_hash, world_hash
from test_provider_environment import cfg as env_cfg, fixture_world
from test_runner_integrity import fixture_run
from test_stateful_integrity import audit_world


def test_world_content_hash_ignores_metadata_but_world_hash_does_not(tmp_path):
    # Reproduces the observed defect: identical worlds saved from different output
    # directories carried different world_hash values because metadata embeds the
    # generating configuration (including experiment.output and the first policy).
    first = audit_world(6)
    second = deepcopy(first)
    first.metadata["config"] = {"experiment": {"output": "runs/first"}, "policy": {"name": "random"}}
    second.metadata["config"] = {"experiment": {"output": "runs/second"}, "policy": {"name": "fixed"}}
    assert world_hash(first) != world_hash(second)
    assert world_content_hash(first) == world_content_hash(second)
    # Content identity must still change when exogenous content changes.
    third = deepcopy(first)
    third.truth["relevant_accounts"] = ["acct:0"]
    assert world_content_hash(third) != world_content_hash(first)
    fourth = deepcopy(first)
    fourth.posts[0]["text"] = "different"
    assert world_content_hash(fourth) != world_content_hash(first)
    for world, name in ((first, "first"), (second, "second")):
        save_world(world, tmp_path / name)
        manifest = json.loads((tmp_path / name / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["world_content_hash"] == world_content_hash(world)
        assert manifest["world_hash"] == world_hash(world)
        assert world_content_hash(load_world(tmp_path / name)) == manifest["world_content_hash"]
    a = json.loads((tmp_path / "first" / "manifest.json").read_text())
    b = json.loads((tmp_path / "second" / "manifest.json").read_text())
    assert a["world_hash"] != b["world_hash"] and a["world_content_hash"] == b["world_content_hash"]


def test_run_records_world_content_hash(tmp_path):
    world, cfg = fixture_run(tmp_path, policy="random")
    summary = run_one(world, cfg, tmp_path / "run")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    expected = json.loads((tmp_path / "world" / "manifest.json").read_text(encoding="utf-8"))["world_content_hash"]
    assert summary["world_content_hash"] == manifest["world_content_hash"] == expected


def test_feedback_mode_changes_only_permitted_label_values():
    """Oracle versus operational feedback must not alter timing, capabilities,
    candidates, receipts, costs, or pending schedules for the same action sequence.
    Only delivered label values (and the reward derived from them) may differ."""
    oracle = FrontierEnv(fixture_world(), env_cfg(feedback={"mode": "oracle", "delay": 1, "noise": 0.1}))
    operational = FrontierEnv(fixture_world(), env_cfg(feedback={"mode": "operational", "delay": 1, "noise": 0.1}))
    _, initial_a = oracle.reset(seed=5)
    _, initial_b = operational.reset(seed=5)
    assert canonical(initial_a) == canonical(initial_b)
    label_values_differed = False
    for step in range(8):
        view_a, view_b = oracle.public_view(), operational.public_view()
        assert view_a.actions == view_b.actions and view_a.valid == view_b.valid
        assert view_a.remaining_requests == view_b.remaining_requests and view_a.remaining_cost == view_b.remaining_cost
        # Pick the same physical action in both environments: a rotating operation preference.
        wanted = ("search", "neighbors", "posts", "inspect")[step % 4]
        slots = [i for i, (a, ok) in enumerate(zip(view_a.actions, view_a.valid)) if ok and a.operation == wanted]
        if not slots:
            slots = [i for i, (a, ok) in enumerate(zip(view_a.actions, view_a.valid)) if ok and a.operation not in ("wait", "stop")]
        if not slots:
            break
        slot = min(slots, key=lambda i: view_a.actions[i].id)
        _, reward_a, term_a, trunc_a, info_a = oracle.step(slot)
        _, reward_b, term_b, trunc_b, info_b = operational.step(slot)
        assert (term_a, trunc_a) == (term_b, trunc_b)
        assert info_a["receipt"] == info_b["receipt"]
        assert info_a["feedback"]["pending_ids"] == info_b["feedback"]["pending_ids"]
        assert info_a["feedback"]["cost"] == info_b["feedback"]["cost"]
        assert [u["outcome_id"] for u in info_a["feedback_updates"]] == [u["outcome_id"] for u in info_b["feedback_updates"]]
        assert [u["available_at"] for u in info_a["feedback_updates"]] == [u["available_at"] for u in info_b["feedback_updates"]]
        for ua, ub in zip(info_a["feedback_updates"], info_b["feedback_updates"]):
            assert [l["id"] for l in ua["labels"]] == [l["id"] for l in ub["labels"]]
            assert ua["action"] == ub["action"] and ua["features"] == ub["features"]
            if [l["value"] for l in ua["labels"]] != [l["value"] for l in ub["labels"]]:
                label_values_differed = True
        assert oracle.tick == operational.tick and oracle.requests == operational.requests and oracle.spent == operational.spent
        assert [p["due"] for p in oracle._pending] == [p["due"] for p in operational._pending]
        if term_a or trunc_a:
            break
    assert label_values_differed, "fixture should produce at least one differing label so the contrast is exercised"
    # Oracle labels are exact truth membership for legitimately exposed entities only.
    truth = set(fixture_world().truth["relevant_accounts"])
    snapshot = oracle.visible_snapshot()
    assert set(snapshot["labels"]) <= set(snapshot["accounts"])
    assert all(value == float(identity in truth) for identity, value in snapshot["labels"].items())
