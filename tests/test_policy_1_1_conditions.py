"""Policy 1.1.0 declared expert action sets, route-independent proxy, availability lag,
declared observation conditions and per-condition/interaction statistics.

Defaults must reproduce policy 1.0.0 behaviour so recorded histories replay unchanged.
"""
from copy import deepcopy
import json

import pytest

from frontier_bench.config import resolve
from frontier_bench.environment import FrontierEnv
from frontier_bench.evaluation import arrangement_summary, condition_summary, full_statistics
from frontier_bench.policies import DEFAULT_GRAPH_OPERATIONS, DEFAULT_ROTATION, POLICY_VERSION, base_operation, make_policy
from frontier_bench.provider import SimulatedProvider
from frontier_bench.runner import apply_condition, run_one
from frontier_bench.schemas import Action
from test_provider_environment import cfg as env_cfg, fixture_world
from test_runner_integrity import fixture_run


def test_defaults_preserve_policy_1_0_0_semantics():
    cfg = resolve()
    assert tuple(cfg["policy"]["graph_operations"]) == DEFAULT_GRAPH_OPERATIONS == ("inspect", "posts", "neighbors")
    assert tuple(cfg["policy"]["rotation"]) == DEFAULT_ROTATION == ("graph", "query", "heuristic")
    assert cfg["feedback"]["proxy"] == "visible_text" and cfg["provider"]["availability_lag"] == 0
    assert cfg["experiment"]["observation_conditions"] is None
    assert POLICY_VERSION == "1.1.0"


def _view_after_reset(config):
    env = FrontierEnv(fixture_world(), config)
    env.reset(seed=3)
    return env


def test_graph_expert_respects_declared_operations():
    env = _view_after_reset(env_cfg(policy={"name": "graph", "graph_operations": ["neighbors"]}))
    view = env.public_view()
    policy = make_policy("graph", 1, env.config["policy"])
    chosen = {base_operation(view.actions[i]) for i in policy.scores(view)}
    assert chosen == {"neighbors"}
    legacy = make_policy("graph", 1, {**env.config["policy"], "graph_operations": list(DEFAULT_GRAPH_OPERATIONS)})
    # Initial exposure is ingested as an inspect receipt, so inspect is on cooldown at tick 0;
    # the legacy expert still proposes the account-free posts operation alongside neighbors.
    legacy_ops = {base_operation(view.actions[i]) for i in legacy.scores(view)}
    assert legacy_ops >= {"posts", "neighbors"} and legacy_ops <= {"inspect", "posts", "neighbors"}
    with pytest.raises(ValueError):
        make_policy("graph", 1, {"graph_operations": ["search"]})
    with pytest.raises(ValueError):
        resolve({"policy": {"graph_operations": []}})


def test_fixed_mixture_uses_declared_graph_operations_and_round_robin_rotation():
    config = env_cfg(policy={"name": "fixed", "graph_weight": 1.0, "graph_operations": ["neighbors"], "rotation": ["graph", "query"]})
    env = _view_after_reset(config)
    view = env.public_view()
    fixed = make_policy("fixed", 5, config["policy"])
    for _ in range(5):
        choice = fixed.choose(view)
        assert choice.arm == "graph" and base_operation(view.actions[choice.slot]) == "neighbors"
    rotation = make_policy("round_robin", 5, config["policy"])
    arms = [rotation.choose(view).arm for _ in range(4)]
    assert arms == ["graph", "query", "graph", "query"]
    assert [e.name for e in make_policy("round_robin", 5, {}).experts] == list(DEFAULT_ROTATION)
    with pytest.raises(ValueError):
        make_policy("round_robin", 5, {"rotation": ["graph", "graph"]})


def test_profile_only_proxy_ignores_bundled_posts():
    # Profiles alternate craft/garden; every post contains the seed term "craft". With
    # visible-text labels every account with an observed post is positive; with
    # profile-only labels the garden-profile authors are negative regardless of posts.
    base = dict(provider={"mode": "complete", "failure_probability": 0.0}, feedback={"delay": 0, "noise": 0.0},
                initial={"accounts": 1, "vocabulary": ["craft"]}, budget={"requests": 6, "cost": 12.0, "slots_per_tick": 4})
    results = {}
    for proxy in ("visible_text", "profile_only"):
        env = FrontierEnv(fixture_world(), env_cfg(**{**base, "feedback": {**base["feedback"], "proxy": proxy}}))
        env.reset(seed=1)
        view = env.public_view()
        slot = next(i for i, a in enumerate(view.actions) if a.operation == "search" and view.valid[i])
        env.step(slot)
        labels = env.visible_snapshot()["labels"]
        assert labels, "search must surface labelled authors in this fixture"
        results[proxy] = labels
    assert set(results["visible_text"].values()) == {0.8}
    profiles = {a["id"]: a["profile"] for a in fixture_world().accounts}
    for identity, value in results["profile_only"].items():
        assert value == (0.8 if "craft" in profiles[identity] else 0.2)
    assert 0.2 in results["profile_only"].values()


def test_availability_lag_hides_recent_documents_from_search_and_posts():
    world = fixture_world()
    complete = {**env_cfg()["provider"], "mode": "complete"}  # full scope, so sets compare exactly
    plain = SimulatedProvider(world, complete, 1)
    lagged = SimulatedProvider(world, {**complete, "availability_lag": 2}, 1)
    search = Action("search", query=("craft",), limit=100)
    assert {p["id"] for p in plain.execute(search, 3).posts} == {p["id"] for p in world.posts if p["available_at"] <= 3}
    assert {p["id"] for p in lagged.execute(search, 3).posts} == {p["id"] for p in world.posts if p["available_at"] <= 1}
    posts = Action("posts", subject="a:0", limit=100)
    assert {p["event_at"] for p in lagged.execute(posts, 3).posts} == {0, 1}
    assert {p["event_at"] for p in plain.execute(posts, 3).posts} == {0, 1, 2, 3}
    with pytest.raises(ValueError):
        resolve({"provider": {"availability_lag": -1}})


def test_declared_observation_conditions_apply_provider_overrides():
    cfg = resolve({"experiment": {"observation_conditions": {
        "capped": {"mode": "restricted", "failure_probability": 0.0},
        "delayed": {"mode": "complete", "availability_lag": 1}}}})
    run_cfg = deepcopy(cfg)
    assert apply_condition(run_cfg, "evolving_delayed", cfg["experiment"]) == "evolving"
    assert run_cfg["provider"]["mode"] == "complete" and run_cfg["provider"]["availability_lag"] == 1
    assert run_cfg["world"]["stationary"] is False and run_cfg["content"]["stationary"] is False
    run_cfg = deepcopy(cfg)
    assert apply_condition(run_cfg, "stationary_capped", cfg["experiment"]) == "stationary"
    assert run_cfg["provider"]["failure_probability"] == 0.0 and run_cfg["world"]["stationary"] is True
    run_cfg = deepcopy(cfg)
    apply_condition(run_cfg, "evolving_restricted", cfg["experiment"])  # legacy label still works
    assert run_cfg["provider"]["mode"] == "restricted"
    with pytest.raises(ValueError):
        apply_condition(deepcopy(cfg), "evolving_ranked", cfg["experiment"])
    with pytest.raises(ValueError):
        resolve({"experiment": {"observation_conditions": {"bad_name": {"mode": "complete"}}}})
    with pytest.raises(ValueError):
        resolve({"experiment": {"observation_conditions": {"x": {"not_a_provider_key": 1}}}})


def test_run_summary_records_observation_label_and_route_counts(tmp_path):
    world, cfg = fixture_run(tmp_path, policy="random")
    summary = run_one(world, cfg, tmp_path / "run", tags={"observation": "capped"})
    assert summary["condition"] == "evolving_capped" and summary["observation"] == "capped"
    assert sum(summary["new_accounts_by_family"].values()) == summary["surfaced_accounts"]
    # Exported-world temporal label still comes from the world artifact, not the tag.
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["config"]["policy"]["graph_operations"] == list(DEFAULT_GRAPH_OPERATIONS)


def _row(policy, seed, condition, arrangement, value, status="completed"):
    return dict(policy=policy, seed=seed, world_seed=seed, condition=condition, arrangement=arrangement, status=status,
                reference_relevant_accounts=value, world_hash=f"w{seed}{arrangement}{condition[:3]}", sparse_probes=True,
                use_motif=True, regime="t", acquisition_contract_hash="c")


def test_condition_summary_reports_interaction_per_seed_block():
    rows = []
    for seed in (1, 2, 3):
        # fixed = 10 everywhere; graph loses 2 under capping relative to complete.
        rows += [_row("fixed", seed, "evolving_complete", "concentrated", 10), _row("fixed", seed, "evolving_capped", "concentrated", 10),
                 _row("graph", seed, "evolving_complete", "concentrated", 8 + seed), _row("graph", seed, "evolving_capped", "concentrated", 6 + seed)]
    stats = condition_summary(rows, baseline="fixed", samples=50)
    assert stats["reference_condition"] == "evolving_complete"
    graph = stats["contrasts"][0]
    assert graph["by_condition"]["evolving_complete"]["mean"] == pytest.approx(0.0)
    assert graph["by_condition"]["evolving_capped"]["mean"] == pytest.approx(-2.0)
    assert graph["by_condition"]["evolving_capped"]["seed_blocks"] == 3
    interaction = graph["interaction_vs_reference"]["evolving_capped"]
    assert interaction["mean"] == pytest.approx(-2.0) and interaction["negative"] == 3 and interaction["interval"] is not None
    # A missing pair is counted, not silently dropped.
    stats = condition_summary(rows[:-1], baseline="fixed", samples=50)
    assert stats["contrasts"][0]["by_condition"]["evolving_capped"]["missing_pairs"] == 0
    assert stats["contrasts"][0]["by_condition"]["evolving_capped"]["seed_blocks"] == 2


def test_arrangement_summary_pairs_mixture_with_components_per_world():
    rows = []
    for seed in (1, 2):
        for cond in ("evolving_complete", "evolving_capped"):
            rows += [_row("fixed", seed, cond, "distributed", 12), _row("graph", seed, cond, "distributed", 7 + seed),
                     _row("query", seed, cond, "distributed", 9), _row("fixed", seed, cond, "weak", 5), _row("graph", seed, cond, "weak", 6)]
    stats = arrangement_summary(rows, mixture="fixed", components=("graph", "query"), samples=50)
    distributed = stats["by_arrangement"]["distributed"]
    assert distributed["fixed_minus_query"]["mean"] == pytest.approx(3.0)
    assert distributed["fixed_minus_best_component"]["seed_blocks"] == 2 and distributed["fixed_minus_best_component"]["positive"] == 2
    assert stats["by_arrangement"]["weak"]["incomplete_cells"] == 4  # query missing in weak cells
    combined = full_statistics(rows, samples=50)
    assert {"contrasts", "by_condition", "by_arrangement"} <= set(combined)
