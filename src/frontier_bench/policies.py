"""Small, inspectable acquisition policies over the allowlisted PublicView only.

Scores estimate acquisition value; none is a relevance probability or momentum.
The River baseline is an adaptation, not SocialSift, NOL-HTR, D3TS, or MCrawl.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from .schemas import Action, Choice, FEATURES, PublicView, stream_seed

POLICY_VERSION = "1.1.0"
POLICY_NAMES = ("random", "graph", "query", "heuristic", "fixed", "round_robin", "learned")
GRAPH_OPERATIONS = frozenset(("inspect", "posts", "neighbors"))
# Policy 1.0.0 defaults, retained so recorded histories replay unchanged.
DEFAULT_GRAPH_OPERATIONS = ("inspect", "posts", "neighbors")
DEFAULT_ROTATION = ("graph", "query", "heuristic")
EXPERT_NAMES = ("graph", "query", "heuristic")


def base_operation(action: Action) -> str:
    return action.base_operation if action.operation in ("page", "revisit") else action.operation


def family(action: Action) -> str:
    op = base_operation(action)
    if op in GRAPH_OPERATIONS:
        return "graph"
    if op == "search":
        return "query"
    return "control"


def feature_dict(values: tuple[float, ...], names: tuple[str, ...] = FEATURES) -> dict[str, float]:
    if len(values) != len(names):
        raise ValueError("Feature vector and feature names have different lengths")
    result = {name: float(value) for name, value in zip(names, values)}
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Policy inputs must contain finite features")
    return result


def acquisition_slots(view: PublicView, wanted_family: str | None = None,
                      operations: tuple[str, ...] | None = None) -> list[int]:
    """Stable order makes a fixed RNG draw invariant to candidate slot ordering.

    `operations` restricts eligibility by base operation (pages and revisits inherit
    their base). It is the declared action set of an expert, not a hidden score.
    """
    return sorted(
        (i for i, (action, valid) in enumerate(zip(view.actions, view.valid))
         if valid and family(action) != "control"
         and (wanted_family is None or family(action) == wanted_family)
         and (operations is None or base_operation(action) in operations)),
        key=lambda i: view.actions[i].id,
    )


def control_slot(view: PublicView) -> int:
    # Empty frontier can recover as the exogenous world evolves. Wait if the
    # environment permits it and there is acquisition budget left.
    order = ("wait", "stop") if view.remaining_requests > 0 and view.remaining_cost > 0 else ("stop", "wait")
    for operation in order:
        for i, (action, valid) in enumerate(zip(view.actions, view.valid)):
            if valid and action.operation == operation:
                return i
    raise ValueError("PublicView has no valid acquisition, WAIT, or STOP action")


def best_slot(view: PublicView, scores: dict[int, float]) -> int:
    if not scores:
        return control_slot(view)
    return min(scores, key=lambda i: (-scores[i], view.actions[i].id))


def _cost(x: dict[str, float]) -> float:
    return max(0.01, x.get("cost", 1.0))


def heuristic_score(x: dict[str, float], expert: str, use_motif: bool = False) -> float:
    """Independently written engineering baseline; weights are not paper findings."""
    support = max(0.0, x.get("support", 0.0))
    relevance = min(1.0, max(0.0, x.get("relevance", 0.0)))
    novelty = min(1.0, max(0.0, x.get("novelty", 0.0)))
    history = max(0.0, x.get("past_yield", 0.0))
    age = max(0.0, x.get("age", 0.0))
    motif = max(0.0, x.get("motif", 0.0)) if use_motif else 0.0
    if expert == "graph":
        value = 0.35 * math.log1p(support) + 0.35 * relevance + 0.3 * novelty
        value += 0.2 * history + 0.12 * math.log1p(motif)
    elif expert == "query":
        value = 0.2 * math.log1p(support) + 0.3 * relevance + 0.4 * novelty
        value += 0.3 * history + 0.12 * math.log1p(motif)
        # Breadth penalty is deliberately mild and tunable by comparing policies;
        # it is not a claim that a particular query length is universally best.
        value /= 1.0 + 0.08 * max(0.0, x.get("query_length", 1.0) - 1.0)
    else:
        value = 0.2 * math.log1p(support) + 0.3 * relevance + 0.3 * novelty + 0.2 * history
        value += 0.12 * math.log1p(motif)
    # Refresh is an explicit action, so staleness never turns discovery into growth.
    value += 0.05 * min(age, 10.0) * x.get("revisit", 0.0)
    return value / _cost(x)


class Policy:
    version = POLICY_VERSION

    def __init__(self, name: str, seed: int, config: dict[str, Any] | None = None):
        self.name = name
        self.config = dict(config or {})
        if self.config.get("pretrained", False):
            raise ValueError("Pretrained policy loading is not implemented; this version is cold-start only")
        self.rng = np.random.default_rng(stream_seed(seed, "policy_exploration", name))
        self.use_motif = bool(self.config.get("use_motif", False))

    def probabilities(self, view: PublicView) -> dict[int, float]:
        raise NotImplementedError

    def choose(self, view: PublicView) -> Choice:
        probabilities = self.probabilities(view)
        slots = sorted(probabilities, key=lambda i: view.actions[i].id)
        if len(slots) == 1:
            selected = slots[0]
        else:
            selected = slots[int(self.rng.choice(len(slots), p=[probabilities[i] for i in slots]))]
        return Choice(selected, probabilities[selected], self.name)

    def observe(self, action: Action, features: tuple[float, ...], feedback: dict) -> None:
        """Non-learning policies intentionally ignore outcomes."""


class RandomPolicy(Policy):
    def probabilities(self, view: PublicView) -> dict[int, float]:
        slots = acquisition_slots(view)
        return {i: 1.0 / len(slots) for i in slots} if slots else {control_slot(view): 1.0}


class HeuristicPolicy(Policy):
    """Graph, query, or all-operation heuristic expert.

    Policy 1.1.0: the graph expert's eligible operations are declared by
    `graph_operations`. The 1.0.0 default (inspect/posts/neighbors) has no operation
    preference, so `inspect` (which can never surface an account) was selected by hash
    tie-break; new experiments declare `["neighbors"]` so graph expansion means
    neighborhood expansion, its pages and revisits.
    """

    def __init__(self, name: str, seed: int, config: dict | None = None):
        super().__init__(name, seed, config)
        operations = tuple(self.config.get("graph_operations", DEFAULT_GRAPH_OPERATIONS))
        if not operations or not set(operations) <= GRAPH_OPERATIONS:
            raise ValueError("graph_operations must be a nonempty subset of inspect/posts/neighbors")
        self.graph_operations = operations

    def eligible(self, view: PublicView) -> list[int]:
        if self.name == "graph":
            return acquisition_slots(view, "graph", self.graph_operations)
        if self.name == "query":
            return acquisition_slots(view, "query")
        return acquisition_slots(view)

    def scores(self, view: PublicView) -> dict[int, float]:
        return {i: heuristic_score(feature_dict(view.features[i], view.feature_names), self.name, self.use_motif)
                for i in self.eligible(view)}

    def probabilities(self, view: PublicView) -> dict[int, float]:
        return {best_slot(view, self.scores(view)): 1.0}


class FixedMixturePolicy(Policy):
    """Randomly choose an available graph/query expert; log marginal action mass."""

    def __init__(self, name: str, seed: int, config: dict | None = None):
        super().__init__(name, seed, config)
        self.graph_weight = float(self.config.get("graph_weight", 0.5))
        if not 0 <= self.graph_weight <= 1:
            raise ValueError("graph_weight must lie in [0, 1]")
        self.experts = {name: HeuristicPolicy(name, seed, self.config) for name in ("graph", "query")}

    def expert_choices(self, view: PublicView) -> list[tuple[str, int, float]]:
        weighted = [("graph", self.graph_weight), ("query", 1.0 - self.graph_weight)]
        active = [(name, weight) for name, weight in weighted if weight > 0 and self.experts[name].eligible(view)]
        if not active:
            # At endpoint weights, preserve the strategy instead of secretly
            # switching a graph-only policy to search (or the reverse).
            return [("control", control_slot(view), 1.0)]
        total = sum(weight for _, weight in active)
        return [(name, next(iter(self.experts[name].probabilities(view))), weight / total) for name, weight in active]

    def probabilities(self, view: PublicView) -> dict[int, float]:
        result: dict[int, float] = {}
        for _, slot, weight in self.expert_choices(view):
            result[slot] = result.get(slot, 0.0) + weight
        return result

    def choose(self, view: PublicView) -> Choice:
        choices = self.expert_choices(view)
        index = 0 if len(choices) == 1 else int(self.rng.choice(len(choices), p=[c[2] for c in choices]))
        arm, slot, arm_probability = choices[index]
        return Choice(slot, self.probabilities(view)[slot], self.name, arm, arm_probability)


class RoundRobinPolicy(Policy):
    """Deterministic expert rotation; skip unavailable expert families."""

    def __init__(self, name: str, seed: int, config: dict | None = None):
        super().__init__(name, seed, config)
        rotation = tuple(self.config.get("rotation", DEFAULT_ROTATION))
        if not rotation or not set(rotation) <= set(EXPERT_NAMES) or len(set(rotation)) != len(rotation):
            raise ValueError("rotation must list distinct experts among graph/query/heuristic")
        self.experts = tuple(HeuristicPolicy(name, seed, self.config) for name in rotation)
        self.next_expert = 0

    def _proposal(self, view: PublicView) -> tuple[int, int]:
        for offset in range(len(self.experts)):
            position = (self.next_expert + offset) % len(self.experts)
            expert = self.experts[position]
            scores = expert.scores(view)
            if scores:
                return position, best_slot(view, scores)
        return self.next_expert, control_slot(view)

    def probabilities(self, view: PublicView) -> dict[int, float]:
        return {self._proposal(view)[1]: 1.0}

    def choose(self, view: PublicView) -> Choice:
        position, slot = self._proposal(view)
        self.next_expert = (position + 1) % len(self.experts)
        return Choice(slot, 1.0, self.name, self.experts[position].name, 1.0)


@dataclass
class PendingExample:
    features: dict[str, float]
    new_ids: set[str]
    labels: dict[str, float] = field(default_factory=dict)
    status: str = ""


class LearnedPolicy(Policy):
    """River linear regression on observed marginal relevant-account return.

    An example is trained exactly once when all newly surfaced identities have
    permitted labels. Pending labels are not zeros. Neither evaluator labels nor
    future-world features are consulted. Failed/unavailable calls are not labels.
    """

    def __init__(self, name: str, seed: int, config: dict | None = None):
        super().__init__(name, seed, config)
        from river import linear_model, optim, preprocessing

        self.epsilon = float(self.config.get("epsilon", 0.2))
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        self.scaler = preprocessing.StandardScaler()
        self.model = linear_model.LinearRegression(
            optimizer=optim.SGD(float(self.config.get("learning_rate", 0.02))),
            intercept_lr=float(self.config.get("learning_rate", 0.02)),
            l2=0.001, clip_gradient=10.0,
        )
        self.pending: dict[str, PendingExample] = {}
        self.completed: set[str] = set()
        self.seen_new_ids: set[str] = set()
        self.learned_examples = 0
        self.training_targets: list[float] = []

    def _features(self, values: tuple[float, ...], names: tuple[str, ...] = FEATURES) -> dict[str, float]:
        x = feature_dict(values, names)
        if not self.use_motif:
            x.pop("motif", None)
        return x

    def _transform(self, x: dict[str, float]) -> dict[str, float]:
        # River 0.21.2's scaler uses defaultdicts. Reading an unseen feature via
        # [] can add keys during transform. Use get() on the public statistics to
        # guarantee that prediction is observationally pure even before learning.
        return {key: (value - self.scaler.means.get(key, 0.0)) / math.sqrt(self.scaler.vars.get(key, 0.0))
                if self.scaler.vars.get(key, 0.0) > 0.0 else 0.0 for key, value in x.items()}

    def predict_return(self, values: tuple[float, ...], names: tuple[str, ...] = FEATURES) -> float:
        x = self._transform(self._features(values, names))
        # The public weights property is a snapshot dict. Direct dot product is
        # exactly LinearRegression's identity-link prediction, without lazy state.
        weights = self.model.weights
        return float(self.model.intercept + sum(weights.get(key, 0.0) * value for key, value in x.items()))

    def scores(self, view: PublicView) -> dict[int, float]:
        scores = {}
        for i in acquisition_slots(view):
            x = feature_dict(view.features[i], view.feature_names)
            if self.learned_examples == 0:
                scores[i] = heuristic_score(x, "heuristic", self.use_motif)
            else:
                scores[i] = max(0.0, self.predict_return(view.features[i], view.feature_names)) / _cost(x)
        return scores

    def probabilities(self, view: PublicView) -> dict[int, float]:
        scores = self.scores(view)
        if not scores:
            return {control_slot(view): 1.0}
        selected = best_slot(view, scores)
        result = {i: self.epsilon / len(scores) for i in scores}
        result[selected] += 1.0 - self.epsilon
        return {i: p for i, p in result.items() if p > 0}

    def observe(self, action: Action, features: tuple[float, ...], feedback: dict) -> None:
        if family(action) == "control":
            return
        outcome_id = str(feedback.get("outcome_id", ""))
        if not outcome_id:
            raise ValueError("Learned feedback requires stable outcome_id for revision/deduplication")
        if outcome_id in self.completed:
            return
        example = self.pending.get(outcome_id)
        if example is None:
            # A delayed update without its originating receipt is not a complete
            # training example. The runner must deliver the receipt first.
            if "new_account_ids" not in feedback:
                return
            status = str(feedback.get("status", ""))
            if status in ("failed", "unavailable", "blocked", "invalid", "admission_failed"):
                self.completed.add(outcome_id)
                return
            new_ids = set(map(str, feedback["new_account_ids"])) - self.seen_new_ids
            self.seen_new_ids.update(new_ids)
            example = PendingExample(self._features(features), new_ids, status=status)
            self.pending[outcome_id] = example
        for label in feedback.get("assessed_labels", feedback.get("labels", [])):
            identifier = str(label["id"])
            if identifier in example.new_ids and label.get("value") is not None:
                value = float(label["value"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Permitted relevance labels must lie in [0, 1]")
                example.labels[identifier] = value
        if not example.new_ids.issubset(example.labels):
            return
        target = sum(example.labels[identifier] for identifier in sorted(example.new_ids))
        self.scaler.learn_one(example.features)
        self.model.learn_one(self._transform(example.features), target)
        self.learned_examples += 1
        self.training_targets.append(target)
        self.completed.add(outcome_id)
        del self.pending[outcome_id]


def make_policy(name: str, seed: int, config: dict | None = None) -> Policy:
    classes = {"random": RandomPolicy, "graph": HeuristicPolicy, "query": HeuristicPolicy,
               "heuristic": HeuristicPolicy, "fixed": FixedMixturePolicy,
               "round_robin": RoundRobinPolicy, "learned": LearnedPolicy}
    if name not in classes:
        raise ValueError(f"Unknown policy {name!r}; expected one of {POLICY_NAMES}")
    return classes[name](name, seed, config)
