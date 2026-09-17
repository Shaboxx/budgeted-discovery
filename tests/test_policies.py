import copy
import pickle
from dataclasses import replace

import pytest

from frontier_bench.policies import FEATURES, POLICY_NAMES, family, make_policy
from frontier_bench.schemas import Action, PublicView


def features(**kwargs):
    values = {"cost": 1.0, "novelty": 0.7, "relevance": 0.5, **kwargs}
    return tuple(float(values.get(key, 0.0)) for key in FEATURES)


def view(actions=None, rows=None):
    actions = tuple(actions or (Action("neighbors", "a"), Action("search", query=("astral",)),
                                Action("inspect", "b"), Action("wait"), Action("stop")))
    rows = tuple(rows or (features() for _ in actions))
    return PublicView(0, actions, rows, tuple(True for _ in actions), 20, 20.0, 2, 0, ())


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_only_valid_actions_and_probability_sum(name):
    policy = make_policy(name, 9)
    state = view()
    state = replace(state, valid=(False, True, True, True, True))
    probabilities = policy.probabilities(state)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    for _ in range(10):
        choice = policy.choose(state)
        assert state.valid[choice.slot]
        assert state.actions[choice.slot].operation not in ("wait", "stop")
        assert 0 < choice.probability <= 1


@pytest.mark.parametrize("name", ("graph", "query"))
def test_family_constraints_include_continuations(name):
    actions = (Action("page", "a", base_operation="posts"),
               Action("revisit", query=("word",), base_operation="search"),
               Action("page", query=("word",), base_operation="search"),
               Action("wait"), Action("stop"))
    state = view(actions)
    policy = make_policy(name, 5)
    assert family(state.actions[policy.choose(state).slot]) == name


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_controls_when_no_acquisitions(name):
    state = view((Action("wait"), Action("stop")))
    policy = make_policy(name, 1)
    assert state.actions[policy.choose(state).slot].operation == "wait"
    exhausted = replace(state, remaining_requests=0)
    assert exhausted.actions[policy.choose(exhausted).slot].operation == "stop"


def test_graph_only_does_not_switch_to_search():
    state = view((Action("search", query=("new",)), Action("wait"), Action("stop")))
    assert state.actions[make_policy("graph", 1).choose(state).slot].operation == "wait"
    assert state.actions[make_policy("fixed", 1, {"graph_weight": 1}).choose(state).slot].operation == "wait"


def test_exact_epsilon_marginal_includes_greedy_exploration_mass():
    state = view()
    policy = make_policy("learned", 2, {"epsilon": 0.3})
    probabilities = sorted(policy.probabilities(state).values())
    assert probabilities == pytest.approx([0.1, 0.1, 0.8])


def test_fixed_mixture_exact_probabilities_and_arm():
    state = view()
    policy = make_policy("fixed", 2, {"graph_weight": 0.7})
    assert sorted(policy.probabilities(state).values()) == pytest.approx([0.3, 0.7])
    for _ in range(20):
        choice = policy.choose(state)
        assert choice.probability == pytest.approx(0.7 if choice.arm == "graph" else 0.3)
    # This directly checks aggregation even if two future experts select one ID.
    policy.expert_choices = lambda _: [("one", 0, 0.7), ("two", 0, 0.3)]
    assert policy.probabilities(state) == {0: 1.0}
    assert policy.choose(state).probability == 1.0


def test_round_robin_is_deterministic_and_not_a_three_way_propensity():
    policy = make_policy("round_robin", 2)
    choices = [policy.choose(view()) for _ in range(6)]
    assert [choice.arm for choice in choices] == ["graph", "query", "heuristic"] * 2
    assert all(choice.probability == 1 for choice in choices)


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_order_invariance_and_no_prediction_mutation(name):
    state = view()
    permutation = (2, 4, 1, 3, 0)
    permuted = replace(state, actions=tuple(state.actions[i] for i in permutation),
                       features=tuple(state.features[i] for i in permutation),
                       valid=tuple(state.valid[i] for i in permutation))
    policy = make_policy(name, 42)
    before = pickle.dumps(policy)
    original = {state.actions[i].id: p for i, p in policy.probabilities(state).items()}
    reordered = {permuted.actions[i].id: p for i, p in policy.probabilities(permuted).items()}
    assert original == reordered
    assert pickle.dumps(policy) == before
    original_choice = policy.choose(state)
    reordered_choice = make_policy(name, 42).choose(permuted)
    assert state.actions[original_choice.slot].id == permuted.actions[reordered_choice.slot].id


def feedback(outcome_id="o:1", new_ids=("a", "b"), labels=(), status="complete"):
    return {"outcome_id": outcome_id, "new_account_ids": list(new_ids), "assessed_labels": list(labels),
            "pending_ids": list(new_ids), "status": status, "cost": 1.0, "new_posts": 0}


def test_pending_labels_are_not_negatives_and_use_original_snapshot():
    policy = make_policy("learned", 4)
    action = Action("search", query=("word",))
    original = features(support=2, search=1)
    policy.observe(action, original, feedback(labels=({"id": "a", "value": 1.0},)))
    assert policy.learned_examples == 0
    assert policy.scaler.counts == {}
    policy.observe(action, features(support=999), {"outcome_id": "o:1", "labels": [{"id": "b", "value": 0.0}]})
    assert policy.learned_examples == 1
    assert policy.training_targets == [1.0]
    assert policy.scaler.means["support"] == 2.0
    policy.observe(action, original, {"outcome_id": "o:1", "labels": [{"id": "b", "value": 0.0}]})
    assert policy.learned_examples == 1


def test_failed_calls_not_zero_examples_partial_evidence_is_valid():
    policy = make_policy("learned", 4)
    action = Action("search", query=("word",))
    policy.observe(action, features(), feedback("fail", (), status="failed"))
    assert policy.learned_examples == 0
    policy.observe(action, features(), feedback("partial", ("a",), ({"id": "a", "value": 1},), "partial"))
    assert policy.training_targets == [1.0]
    policy.observe(action, features(), feedback("zero", (), status="complete"))
    assert policy.training_targets == [1.0, 0.0]


def test_duplicate_receipts_and_duplicate_entities_have_no_new_credit():
    policy = make_policy("learned", 4)
    action = Action("neighbors", "source")
    event = feedback("one", ("a",), ({"id": "a", "value": 1},))
    policy.observe(action, features(), event)
    policy.observe(action, features(), event)
    policy.observe(action, features(), feedback("two", ("a",), ({"id": "a", "value": 1},)))
    assert policy.training_targets == [1.0, 0.0]


def test_no_future_labels_accepted_by_incidental_keys():
    policy = make_policy("learned", 3)
    event = feedback()
    event["truth"] = {"a": 1, "b": 1}
    event["hidden_relevant_count"] = 2
    policy.observe(Action("search", query=("word",)), features(), event)
    assert policy.learned_examples == 0


def test_trained_predictions_match_river_and_do_not_change_scaler():
    policy = make_policy("learned", 4)
    action = Action("search", query=("word",))
    for i in range(4):
        policy.observe(action, features(support=i), feedback(str(i), (str(i),), ({"id": str(i), "value": i % 2},)))
    before = pickle.dumps(policy)
    row = features(support=1.5)
    predicted = policy.predict_return(row)
    reference = copy.deepcopy(policy)
    expected = reference.model.predict_one(reference.scaler.transform_one(reference._features(row)))
    assert predicted == pytest.approx(expected)
    assert pickle.dumps(policy) == before
    policy.probabilities(view())
    assert pickle.dumps(policy) == before


def test_motif_ablation_removes_only_motif_feature():
    disabled = make_policy("learned", 3)
    enabled = make_policy("learned", 3, {"use_motif": True})
    assert "motif" not in disabled._features(features(motif=3))
    assert enabled._features(features(motif=3))["motif"] == 3
    plain = view((Action("neighbors", "a"), Action("neighbors", "b"), Action("wait")),
                 (features(motif=1), features(motif=10), features()))
    scores_off = make_policy("heuristic", 1).scores(plain)
    scores_on = make_policy("heuristic", 1, {"use_motif": True}).scores(plain)
    assert scores_off[0] == scores_off[1]
    assert scores_on[1] > scores_on[0]


def test_independent_policy_instances_do_not_share_learned_state():
    a, b = make_policy("learned", 5), make_policy("learned", 5)
    a.observe(Action("neighbors", "source"), features(), feedback("one", ("x",), ({"id": "x", "value": 1},)))
    assert a.learned_examples == 1
    assert b.learned_examples == 0
    assert b.pending == {}
    assert b.model.weights == {}


def test_configuration_validation():
    with pytest.raises(ValueError):
        make_policy("imaginary", 1)
    with pytest.raises(ValueError):
        make_policy("learned", 1, {"epsilon": 1.5})
    with pytest.raises(ValueError):
        make_policy("fixed", 1, {"graph_weight": -0.5})
    with pytest.raises(ValueError, match="cold-start"):
        make_policy("learned", 1, {"pretrained": True})
