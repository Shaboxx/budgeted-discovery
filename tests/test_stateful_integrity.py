"""Independent property/stateful integrity audit of the public simulator boundary.

These fixtures are deliberately hand-specified controls, not TADC-SBM acceptance.
"""
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from frontier_bench.config import resolve
from frontier_bench.environment import FrontierEnv
from frontier_bench.knowledge import Knowledge
from frontier_bench.provider import SimulatedProvider
from frontier_bench.schemas import Action, Receipt, World, canonical
from frontier_bench.world import validate_world


def audit_world(ticks=24):
    accounts = [dict(id=f"acct:{i}", profile="craft common", attributes=[], available_at=0)
                for i in range(6)]
    # Account 5 remains isolated; some edges disappear on odd ticks.
    snapshots = [[["acct:0", "acct:1"], ["acct:1", "acct:2"],
                  *([["acct:2", "acct:3"]] if t % 2 == 0 else [])]
                 for t in range(ticks)]
    posts = [dict(id=f"post:{t}:{i}", author=f"acct:{i}", event_at=t,
                  available_at=t + (i % 2), text="craft common", terms=["craft", "common"],
                  hashtags=["craft"], mentions=["acct:4"] if i == 0 else [])
             for t in range(ticks) for i in range(6)]
    return World(accounts, snapshots, posts,
                 {"relevant_accounts": ["acct:1", "acct:3"], "regions": {}},
                 {"generator": "deterministic-integrity-control"})


def audit_config(**patches):
    cfg = dict(initial=dict(accounts=2, vocabulary=["craft"]),
               feedback=dict(mode="operational", delay=2, noise=0.1),
               provider=dict(mode="restricted", failure_probability=0.2, page_size=2,
                             ranking_noise=0.05, latency=0),
               actions=dict(cap=32, revisit_after=1),
               budget=dict(requests=12, cost=18.0, slots_per_tick=3, max_steps=30))
    for key, value in patches.items():
        cfg.setdefault(key, {}).update(value)
    return resolve(cfg)


def assert_same_step(left, right):
    for key in left[0]:
        np.testing.assert_array_equal(left[0][key], right[0][key])
    assert left[1:] == right[1:]


def checked_receipt(receipt, source_world):
    account_ids = {a["id"] for a in source_world.accounts}
    post_ids = {p["id"] for p in source_world.posts}
    assert all(p["event_at"] <= p["available_at"] <= receipt["scope_tick"]
               for p in receipt["posts"])
    assert all(a.get("available_at", 0) <= receipt["scope_tick"]
               for a in receipt["accounts"])
    assert receipt["scope_tick"] <= receipt["request_tick"] <= receipt["response_tick"]
    for edge in receipt["edges"]:
        source, target = edge["source"], edge["target"]
        if edge["type"] == "PUBLISHED":
            assert source in account_ids and target in post_ids
        elif edge["type"] == "HAS_HASHTAG":
            assert source in post_ids and target.startswith("term:")
        elif edge["type"] == "MENTIONS":
            assert source in post_ids and target in account_ids
        elif edge["type"] == "INTERACTS_WITH":
            assert source in account_ids and target in account_ids
        else:
            pytest.fail(f"unexpected observed edge type: {edge['type']}")


class EnvironmentIntegrityMachine(RuleBasedStateMachine):
    @initialize(seed=st.integers(min_value=0, max_value=5000))
    def begin(self, seed):
        self.world = audit_world()
        hidden_alternative = deepcopy(self.world)
        hidden_alternative.truth = {"relevant_accounts": [a["id"] for a in self.world.accounts],
                                    "regions": {a["id"]: "secret-community" for a in self.world.accounts}}
        cfg = audit_config()
        self.env = FrontierEnv(self.world, cfg)
        self.shadow = FrontierEnv(hidden_alternative, cfg)
        self.env.reset(seed=seed)
        self.shadow.reset(seed=seed)
        self.idle = FrontierEnv(self.world, cfg)
        self.idle.reset(seed=seed)
        self.idle_snapshot = self.idle.visible_snapshot()
        self.original_world = canonical(asdict(self.world))
        self.account_credit = set()
        self.post_credit = set()
        self.done = False
        self.requests = 0
        self.cost = 0.0

    @rule(index=st.integers(min_value=-4, max_value=90))
    def step_selected_slot(self, index):
        if self.done:
            with pytest.raises(RuntimeError):
                self.env.step(index)
            return
        before = self.env.visible_snapshot()
        actual = self.env.step(index)
        counterfactual = self.shadow.step(index)
        assert_same_step(actual, counterfactual)
        observation, reward, terminated, truncated, info = actual
        self.done = terminated or truncated
        assert not (terminated and truncated)
        assert self.env.observation_space.contains(observation)
        if "receipt" in info:
            receipt = info["receipt"]
            if receipt["execution_status"] != "admission_failed":
                self.requests += 1
                self.cost += receipt["cost"]
            if receipt["execution_status"] == "completed":
                checked_receipt(receipt, self.world)
            else:
                assert receipt["observation_status"] != "observed_zero"
            feedback = info["feedback"]
            new_accounts = set(feedback["new_account_ids"])
            new_posts = set(feedback["new_post_ids"])
            assert not self.account_credit.intersection(new_accounts)
            assert not self.post_credit.intersection(new_posts)
            self.account_credit.update(new_accounts)
            self.post_credit.update(new_posts)
            assert not set(feedback["pending_ids"]).intersection(
                label["id"] for label in feedback["assessed_labels"])
            assert reward == pytest.approx(sum(x["value"] for x in feedback["assessed_labels"]))
        else:
            # WAIT, STOP, and invalid admission may advance time/labels, never evidence/cost.
            after = self.env.visible_snapshot()
            assert after["accounts"] == before["accounts"]
            assert after["posts"] == before["posts"]
            assert reward == 0
        assert self.env.requests == self.requests
        assert self.env.spent == pytest.approx(self.cost)

    @invariant()
    def isolated_truth_world_and_accounting(self):
        assert self.env.public_view() == self.shadow.public_view()
        assert self.idle.visible_snapshot() == self.idle_snapshot
        assert canonical(asdict(self.world)) == self.original_world
        assert self.env.requests <= self.env.config["budget"]["requests"]
        assert self.env.spent <= self.env.config["budget"]["cost"] + 1e-9
        assert set(self.env.visible_snapshot()["labels"]) <= set(self.env.visible_snapshot()["accounts"])


TestEnvironmentIntegrity = EnvironmentIntegrityMachine.TestCase
TestEnvironmentIntegrity.settings = settings(max_examples=20, stateful_step_count=25, deadline=None,
                                             derandomize=True)


@given(extra=st.integers(min_value=1, max_value=40), tick=st.integers(min_value=0, max_value=3))
@settings(max_examples=20, deadline=None, derandomize=True)
def test_future_prefix_and_hidden_vocabulary_invariance(extra, tick):
    left_world = audit_world()
    right_world = deepcopy(left_world)
    right_world.posts.extend(dict(id=f"future:{i}", author="acct:5", event_at=tick + 5,
                                  available_at=tick + 6, text="unseenfuturetoken craft craft",
                                  terms=["unseenfuturetoken", "craft"], hashtags=[], mentions=[])
                             for i in range(extra))
    config = audit_config(provider=dict(failure_probability=0))
    providers = [SimulatedProvider(w, config["provider"], 113) for w in (left_world, right_world)]
    action = Action("search", query=("craft",), limit=2)
    assert providers[0].execute(action, tick).to_dict() == providers[1].execute(action, tick).to_dict()
    left, right = [FrontierEnv(w, config) for w in (left_world, right_world)]
    left.reset(seed=19)
    right.reset(seed=19)
    assert left.public_view() == right.public_view()
    assert "unseenfuturetoken" not in canonical([a.to_dict() for a in right.public_view().actions])
    for _ in range(3):
        slot = next(i for i, a in enumerate(left.public_view().actions)
                    if a.operation == "search" and left.public_view().valid[i])
        assert_same_step(left.step(slot), right.step(slot))
        assert left.public_view() == right.public_view()


@given(seed=st.integers(min_value=0, max_value=10000))
@settings(max_examples=15, deadline=None, derandomize=True)
def test_provider_randomness_independent_of_request_order(seed):
    cfg = audit_config()["provider"]
    left = SimulatedProvider(audit_world(), cfg, seed)
    right = SimulatedProvider(audit_world(), cfg, seed)
    first = Action("search", query=("craft",))
    unrelated = Action("neighbors", subject="acct:0")
    expected = left.execute(first, 2).to_dict()
    right.execute(unrelated, 1)
    assert right.execute(first, 2).to_dict() == expected


@given(seed=st.integers(min_value=0, max_value=5000))
@settings(max_examples=30, deadline=None, derandomize=True)
def test_provider_noise_depends_on_request_not_acquisition_ancestry(seed):
    cfg = audit_config(provider=dict(failure_probability=0.5))["provider"]
    left = SimulatedProvider(audit_world(), cfg, seed)
    right = SimulatedProvider(audit_world(), cfg, seed)
    # Same physical operation/scope, reached from two different query histories.
    base = Action("search", query=("craft",), parent_id="act:from-graph")
    alternate = Action("search", query=("craft",), parent_id="act:from-lexical")
    expected = left.execute(base, 2).to_dict()
    actual = right.execute(alternate, 2).to_dict()
    expected.pop("action_id")
    actual.pop("action_id")
    # Cursor metadata can identify a branch, but neither failure nor ranked evidence should.
    expected.pop("next_cursor")
    actual.pop("next_cursor")
    assert actual == expected


def test_continuation_cannot_be_replayed_before_its_scope():
    cfg = audit_config(provider=dict(failure_probability=0))["provider"]
    provider = SimulatedProvider(audit_world(), cfg, 19)
    receipt = provider.execute(Action("search", query=("craft",), limit=2), 5)
    assert receipt.next_cursor
    replayed = provider.execute(Action("page", cursor=receipt.next_cursor, base_operation="search"), 1)
    assert replayed.execution_status == "admission_failed"
    assert not replayed.posts and not replayed.accounts and replayed.cost == 0


def test_explicit_invalid_wait_stop_and_exhaustion():
    cfg = audit_config(provider=dict(failure_probability=1),
                       budget=dict(requests=1, slots_per_tick=2))
    env = FrontierEnv(audit_world(), cfg)
    env.reset(seed=8)
    assert env.step(-1)[4]["status"] == "invalid_action"
    assert env.requests == 0 and env.spent == 0
    assert env.step(0)[4]["status"] == "waited"
    assert env.requests == 0 and env.spent == 0 and env.tick == 1
    slot = next(i for i, a in enumerate(env.public_view().actions) if a.operation == "search")
    _, _, terminated, truncated, info = env.step(slot)
    assert terminated and not truncated and info["termination_reason"] == "budget"
    assert info["receipt"]["execution_status"] == "failed" and env.requests == 1
    env.reset(seed=8)
    _, _, terminated, truncated, info = env.step(1)
    assert terminated and not truncated and info["termination_reason"] == "stop"
    assert env.requests == 0 and env.spent == 0


def test_unseen_graph_structure_cannot_change_frontier_or_features():
    world = audit_world()
    alternative = deepcopy(world)
    alternative.snapshots = [[["acct:0", "acct:5"], ["acct:3", "acct:4"]]
                             for _ in range(world.ticks)]
    cfg = audit_config()
    left, right = FrontierEnv(world, cfg), FrontierEnv(alternative, cfg)
    left.reset(seed=18)
    right.reset(seed=18)
    assert left.public_view() == right.public_view()
    # Neither process has earned relationship observations by merely seeing identities.
    assert left._knowledge.edges == right._knowledge.edges == {}
    assert_same_step(left.step(0), right.step(0))


def test_hidden_future_accounts_do_not_change_initial_exposure():
    world = audit_world()
    alternative = deepcopy(world)
    alternative.accounts.append(dict(id="acct:future", profile="secretfutureword", attributes=[],
                                     available_at=20))
    cfg = audit_config()
    left, right = FrontierEnv(world, cfg), FrontierEnv(alternative, cfg)
    obs_left, info_left = left.reset(seed=18)
    obs_right, info_right = right.reset(seed=18)
    for key in obs_left:
        np.testing.assert_array_equal(obs_left[key], obs_right[key])
    assert info_left == info_right
    assert left.public_view() == right.public_view()


def test_policy_visible_exports_cannot_mutate_other_runs_or_world():
    world = audit_world()
    before = canonical(asdict(world))
    cfg = audit_config(provider=dict(failure_probability=0))
    env = FrontierEnv(world, cfg)
    env.reset(seed=18)
    snapshot = env.visible_snapshot()
    first = next(iter(snapshot["accounts"]))
    snapshot["accounts"][first]["profile"] = "mutated outside boundary"
    assert env.visible_snapshot()["accounts"][first]["profile"] != "mutated outside boundary"
    slot = next(i for i, a in enumerate(env.public_view().actions) if a.operation == "search")
    info = env.step(slot)[4]
    assert info["receipt"]["posts"]
    post_id = info["receipt"]["posts"][0]["id"]
    info["receipt"]["posts"][0]["text"] = "mutated outside boundary"
    assert env.visible_snapshot()["posts"][post_id]["text"] != "mutated outside boundary"
    assert canonical(asdict(world)) == before


def test_continuation_contract_preserves_requested_scope():
    knowledge = Knowledge(["craft"])
    action = Action("search", query=("craft",), since=2, until=5, limit=2,
                    direction="undirected", objective="unique_relevant_accounts_v1")
    receipt = Receipt(action.id, "completed", "partial", 5, 5, 1.5,
                      next_cursor="cursor:contract-test", scope_tick=5)
    knowledge.ingest(receipt, action)
    continuation = knowledge.cursors[receipt.next_cursor]
    assert continuation.since == action.since
    assert continuation.until == action.until
    assert continuation.provider_revision == action.provider_revision
    assert continuation.objective == action.objective
    assert continuation.scope_tick == receipt.scope_tick


def test_equivalent_query_branches_share_one_candidate_with_all_lineage():
    cfg = audit_config()
    left, right = Knowledge(["craft", "common"]), Knowledge(["craft", "common"])
    parents = [Action("search", query=("craft",)), Action("search", query=("common",))]
    left.branches = list(parents)
    right.branches = list(reversed(parents))
    first, first_overflow = left.candidates(0, cfg["actions"], cfg["provider"])
    second, second_overflow = right.candidates(0, cfg["actions"], cfg["provider"])
    assert first == second and first_overflow == second_overflow
    matches = [a for a in first if a.query == ("common", "craft")]
    assert len(matches) == 1
    assert set(left.candidate_lineage[matches[0].id]) == {p.id for p in parents}
    assert left.candidate_lineage == right.candidate_lineage


def test_canonical_entity_identity_cannot_have_two_types():
    world = audit_world()
    world.posts[0]["id"] = world.accounts[0]["id"]
    with pytest.raises(ValueError, match="identity|collision|type"):
        validate_world(world)


def test_world_object_version_is_validated():
    world = audit_world()
    world.schema_version = "99.0.0"
    with pytest.raises(ValueError, match="schema|version"):
        validate_world(world)
