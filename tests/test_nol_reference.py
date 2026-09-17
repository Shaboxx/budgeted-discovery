import networkx as nx
import numpy as np
import pytest

from frontier_bench.nol_reference import BasicNOL, StaticNeighborhoodProvider, run_static_nol, static_features


def test_ogd_update_matches_reference_equation():
    model = BasicNOL(0, alpha=0.1)
    model.theta = np.array([0.2, 0, 0, 0, 0], dtype=float)
    model.observe((1, 0, 0, 0, 0), 2)
    assert model.theta.tolist() == pytest.approx([0.56, 0, 0, 0, 0])
    model.observe((10, 0, 0, 0, 0), 0)
    assert np.all(model.theta >= 0)


def test_deterministic_static_graph_has_known_new_node_reward():
    graph = nx.Graph([("a", "b"), ("a", "d"), ("b", "c"), ("d", "e")])
    result = run_static_nol(graph, ["a"], budget=5, seed=3, epsilon=0)
    assert result["total_reward"] == 4
    assert result["observed_nodes"] == ["a", "b", "c", "d", "e"]
    assert [row["reward"] for row in result["records"]] == [2, 1, 0, 1, 0]
    assert [row["node"] for row in result["records"]] == ["a", "b", "c", "d", "e"]
    assert result == run_static_nol(graph, ["a"], budget=5, seed=3, epsilon=0)


def test_complete_neighbors_hidden_until_probed_and_isolates_preserved():
    graph = nx.Graph([("a", "b"), ("b", "c")])
    graph.add_node("isolate")
    provider = StaticNeighborhoodProvider(graph, ["a", "isolate"])
    first = provider.view()
    assert set(first.adjacency) == {"a", "isolate"}
    assert provider.probe("a") == {"b"}
    assert "c" not in provider.view().adjacency
    assert "isolate" in provider.view().adjacency
    with pytest.raises(ValueError):
        provider.probe("a")
    with pytest.raises(ValueError):
        provider.probe("c")
    assert provider.probe("b") == {"c"}


def test_finite_features_on_singleton_and_equal_degree_graph():
    graph = nx.Graph()
    graph.add_node("a")
    provider = StaticNeighborhoodProvider(graph, ["a"])
    assert static_features(provider.view())["a"] == (0, 0, 1, 0, 0)
    assert run_static_nol(graph, ["a"], budget=8)["total_reward"] == 0


def test_exact_exploration_probability_in_static_reference():
    provider = StaticNeighborhoodProvider(nx.path_graph(["a", "b", "c"]), ["a", "b"])
    probabilities = BasicNOL(seed=2, epsilon=0.2).probabilities(provider.view())
    assert sorted(probabilities.values()) == pytest.approx([0.1, 0.9])
