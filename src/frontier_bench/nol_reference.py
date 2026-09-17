"""Traceable basic NOL comparison in a compatible static node-query setting.

Source inspected: https://github.com/tlarock/nol at
39c9ec8bc8e05a91c623511302978d2de479c0ff, particularly NOL.py,
DefaultFeatures.py, Network.py. This is an independent implementation of the
basic squared-error OGD mechanism, not execution of the old reference package
and not a reproduction of NOL-HTR or the paper's experiments.

Reference source license (retained conservatively for algorithm adaptation):
MIT License, Copyright (c) 2019 Tim LaRock.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass
import networkx as nx
import numpy as np

from .schemas import stream_seed

REFERENCE_COMMIT = "39c9ec8bc8e05a91c623511302978d2de479c0ff"
NOL_FEATURES = ("degree", "clustering", "component_size", "probed_fraction", "lost_reward")


def _minmax(values: dict[str, float], equal: float = 0.0) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    return {node: (value - low) / (high - low) if high > low else equal for node, value in values.items()}


@dataclass(frozen=True)
class StaticView:
    """Only observed edges; no reference graph or undiscovered identifiers."""
    adjacency: dict[str, tuple[str, ...]]
    probed: frozenset[str]
    original_nodes: frozenset[str]
    lost_reward: dict[str, int]


def static_features(view: StaticView) -> dict[str, tuple[float, ...]]:
    observed = nx.Graph()
    observed.add_nodes_from(view.adjacency)
    observed.add_edges_from((node, neighbor) for node, neighbors in view.adjacency.items() for neighbor in neighbors)
    degree = _minmax(dict(observed.degree()))
    clustering = nx.clustering(observed)
    component = _minmax({node: len(nodes) for nodes in nx.connected_components(observed) for node in nodes}, equal=1.0)
    lost = _minmax({node: view.lost_reward.get(node, 0) for node in observed})
    return {node: (degree[node], clustering[node], component[node],
                   len(set(view.adjacency[node]) & view.probed) / max(1, len(view.adjacency[node])), lost[node])
            for node in sorted(observed) if node not in view.probed}


class BasicNOL:
    """Nonnegative OGD; objective is count of new nodes, without target labels."""

    def __init__(self, seed: int = 0, alpha: float = 0.05, epsilon: float = 0.2):
        if not 0 <= epsilon <= 1 or alpha <= 0:
            raise ValueError("epsilon must be in [0,1] and alpha must be positive")
        self.rng = np.random.default_rng(stream_seed(seed, "nol_reference_policy"))
        self.theta = np.maximum(0.0, self.rng.uniform(-0.2, 0.2, len(NOL_FEATURES)))
        self.alpha = alpha
        self.epsilon = epsilon

    def probabilities(self, view: StaticView) -> dict[str, float]:
        features = static_features(view)
        if not features:
            return {}
        greedy = min(features, key=lambda node: (-float(np.dot(self.theta, features[node])), node))
        # Matches the original reference's exploration priority for remaining
        # initial-sample nodes, then the whole currently observed frontier.
        exploration = sorted((set(features) & view.original_nodes) or set(features))
        result = {node: self.epsilon / len(exploration) for node in exploration}
        result[greedy] = result.get(greedy, 0.0) + 1.0 - self.epsilon
        return {node: p for node, p in result.items() if p > 0}

    def choose(self, view: StaticView) -> tuple[str, float, tuple[float, ...]]:
        probabilities = self.probabilities(view)
        if not probabilities:
            raise ValueError("Static frontier exhausted")
        nodes = sorted(probabilities)
        selected = nodes[0] if len(nodes) == 1 else nodes[int(self.rng.choice(len(nodes), p=[probabilities[n] for n in nodes]))]
        return selected, probabilities[selected], static_features(view)[selected]

    def observe(self, features: tuple[float, ...], new_nodes: int) -> None:
        if new_nodes < 0:
            raise ValueError("New-node reward must be nonnegative")
        x = np.asarray(features, dtype=float)
        error = float(new_nodes) - float(np.dot(self.theta, x))
        self.theta = np.maximum(0.0, self.theta + 2.0 * self.alpha * error * x)


class StaticNeighborhoodProvider:
    """Reference complete-neighbor oracle, separate from the policy."""

    def __init__(self, graph: nx.Graph, initial_nodes: list[str]):
        if graph.is_directed() or graph.is_multigraph():
            raise ValueError("NOL reference requires a simple undirected static graph")
        if not initial_nodes or not set(initial_nodes).issubset(graph):
            raise ValueError("Initial sample must contain existing nodes")
        self._graph = nx.freeze(graph.copy())
        self.observed = graph.subgraph(initial_nodes).copy()
        self.initial_nodes = frozenset(initial_nodes)
        self.probed: set[str] = set()
        self.lost_reward = {node: 0 for node in self.observed}

    def view(self) -> StaticView:
        return StaticView({node: tuple(sorted(self.observed.neighbors(node))) for node in sorted(self.observed)},
                          frozenset(self.probed), self.initial_nodes, dict(self.lost_reward))

    def probe(self, node: str) -> set[str]:
        if node not in self.observed or node in self.probed:
            raise ValueError("Only known unprobed nodes may be queried")
        neighbors = set(self._graph.neighbors(node))
        new_nodes = neighbors - set(self.observed)
        new_neighbors = neighbors - set(self.observed.neighbors(node))
        for neighbor in sorted(neighbors):
            if neighbor in new_neighbors:
                self.lost_reward[neighbor] = self.lost_reward.get(neighbor, 0) + 1
            self.observed.add_edge(node, neighbor)
        self.probed.add(node)
        return new_nodes


def run_static_nol(graph: nx.Graph, initial_nodes: list[str], budget: int, seed: int = 0,
                   alpha: float = 0.05, epsilon: float = 0.2) -> dict:
    """Small driver for the compatible reference control, not main benchmark."""
    provider = StaticNeighborhoodProvider(graph, initial_nodes)
    policy = BasicNOL(seed, alpha, epsilon)
    records = []
    for request in range(budget):
        view = provider.view()
        if not static_features(view):
            break
        node, probability, features = policy.choose(view)
        new_nodes = provider.probe(node)
        policy.observe(features, len(new_nodes))
        records.append({"request": request + 1, "node": node, "probability": probability,
                        "features": list(features), "new_nodes": sorted(new_nodes), "reward": len(new_nodes)})
    return {"reference_commit": REFERENCE_COMMIT, "objective": "new_nodes", "mode": "static_complete_neighbors",
            "records": records, "total_reward": sum(record["reward"] for record in records),
            "observed_nodes": sorted(provider.observed), "theta": policy.theta.tolist()}
