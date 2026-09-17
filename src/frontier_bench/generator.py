"""Actual TADC-SBM adapter and independent precomputed canonical world builder."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from importlib.metadata import version
import io

import networkx as nx
import numpy as np

from .content import emit_content, NICHE_VOCABULARY, OTHER_VOCABULARY, SHARED_VOCABULARY
from .diffusion import legacy_random_stream, simulate_diffusion
from .schemas import World, digest, stream_seed


TADC_COMMIT = "786c943104030fd0d7a4a5613ff20bdc1bc9d82f"


def topology(config: dict, seed: int) -> tuple[list[nx.Graph], list[list[int]], np.ndarray, dict]:
    """Call TADC-SBM directly. No alternate generator fallback is permitted."""
    import graph_tool as gt
    from tadcsbm import tadcsbm_simulator
    from tadcsbm.simulations import MatchType
    from tadcsbm.utils import generate_block_matrix, generate_transition_matrix

    name = config.get("generator", "tadc")
    if name != "tadc":
        raise ValueError("This adapter executes only generator='tadc'")
    count = int(config.get("accounts", 120))
    ticks = int(config.get("snapshots", 8))
    communities = int(config.get("communities", 4))
    if count < communities or communities < 2 or ticks < 1:
        raise ValueError("Need accounts >= communities >= 2 and snapshots >= 1")
    feature_dim = int(config.get("feature_dim", 4))
    eta = float(config.get("eta", 0.8))
    sampling = float(config.get("edge_sampling_rate", 1.0))
    edges = int(config.get("num_edges", count * 5))
    if not 0 <= sampling <= 1 or not 0 <= eta <= 1 or edges < 0 or feature_dim < 0:
        raise ValueError("Invalid generator range")
    if config.get("reverse_snapshot_order", False):
        raise ValueError("Causal benchmark forbids reverse_snapshot_order=True")
    stationary = bool(config.get("stationary", False))
    if "evolving" in config:
        stationary = not bool(config["evolving"])
    fractions = np.asarray(config.get("pi") or [1.0 / communities] * communities, dtype=float)
    if len(fractions) != communities or np.any(fractions <= 0):
        raise ValueError("pi must contain one positive proportion per community")
    fractions = fractions / fractions.sum()
    prop_mat = np.asarray(config["prop_mat"], dtype=float) if config.get("prop_mat") is not None else (
        generate_block_matrix(communities, p=float(config.get("intra_probability", 0.8)),
                              q=float(config.get("inter_probability", 0.2))))
    transitions = (np.asarray(config["tau_mat"], dtype=float) if config.get("tau_mat") is not None else
                   generate_transition_matrix(communities, eta=eta,
                                              uniform_all=bool(config.get("uniform_all", False))))
    if prop_mat.shape != (communities, communities) or np.any(prop_mat < 0):
        raise ValueError("prop_mat must be a nonnegative k-by-k matrix")
    if not np.allclose(prop_mat, prop_mat.T) or prop_mat.sum() <= 0:
        raise ValueError("prop_mat must be symmetric with positive mass")
    if transitions.shape != (communities, communities) or np.any(transitions < 0):
        raise ValueError("tau_mat must be nonnegative k-by-k")
    if not np.allclose(transitions.sum(axis=1), 1):
        raise ValueError("tau_mat must be row stochastic")
    degree_mode = config.get("degree_correction", "constant")
    if isinstance(degree_mode, bool):
        degree_mode = "lognormal" if degree_mode else "constant"
    degree_rng = np.random.default_rng(stream_seed(seed, "degree_propensities"))
    if config.get("out_degs") is not None:
        degree_propensities = np.asarray(config["out_degs"], dtype=float)
        if len(degree_propensities) != count or np.any(degree_propensities <= 0):
            raise ValueError("out_degs must contain a positive value per account")
    elif degree_mode == "constant":
        degree_propensities = None
    elif degree_mode == "lognormal":
        degree_propensities = degree_rng.lognormal(0, float(config.get("degree_sigma", 1.0)), count)
    else:
        raise ValueError("degree_correction must be constant or lognormal")
    match = str(config.get("feature_match", "GROUPED")).upper()
    if match not in MatchType.__members__:
        raise ValueError("feature_match must be GROUPED, NESTED, or RANDOM")
    groups = int(config.get("feature_groups") or communities)
    gt.openmp_set_num_threads(1)
    with legacy_random_stream(seed), redirect_stdout(io.StringIO()):
        result = tadcsbm_simulator(
            num_vertices=count, num_edges=edges, pi=fractions,
            snapshots=1 if stationary else ticks, prop_mat=prop_mat, tau_mat=transitions,
            out_degs=degree_propensities, feature_dim=feature_dim,
            feature_center_distance=float(config.get("feature_center_distance", 2.0)),
            feature_cluster_variance=float(config.get("feature_cluster_variance", 1.0)),
            feature_group_match_type=MatchType[match], num_feature_groups=groups,
            reverse_snapshot_order=False,
            fixed_probabilities=bool(config.get("fixed_probabilities", int(config.get("gamma", 1)) == 0)),
            edge_sampling_rate=sampling, random_seed=int(seed),
        )
    snapshots, memberships, raw_sizes = [], [], []
    for graph, membership in zip(result.graph, result.graph_memberships):
        original = graph.vp["orig_idx"].a
        converted = nx.Graph()
        converted.add_nodes_from(range(count))
        converted.add_edges_from((int(original[int(u)]), int(original[int(v)]))
                                 for u, v in graph.get_edges()[:, :2])
        snapshots.append(converted)
        memberships.append([int(value) for value in membership])
        raw_sizes.append(int(graph.num_vertices()))
    if stationary:
        snapshots = [snapshots[0].copy() for _ in range(ticks)]
        memberships = [memberships[0].copy() for _ in range(ticks)]
        raw_sizes = raw_sizes * ticks
    attributes = np.asarray(result.node_features1, dtype=float) if feature_dim else np.empty((count, 0))
    if attributes.shape != (count, feature_dim) or not np.all(np.isfinite(attributes)):
        raise AssertionError("Unexpected TADC attribute shape or nonfinite values")
    return snapshots, memberships, attributes, {
        "name": "tadc-sbm", "version": version("tadc-sbm"),
        "inspected_source_commit": TADC_COMMIT, "graph_tool_version": gt.__version__,
        "requested_num_edges": edges, "raw_snapshot_nodes": raw_sizes,
        "persistent_accounts": count, "restored_isolates": [count - size for size in raw_sizes],
        "stationary_topology": stationary, "seed": int(seed),
        "causal_features": "generated once using initial memberships; reverse=False",
        "feature_match": match, "prop_mat": prop_mat.tolist(), "tau_mat": transitions.tolist(),
        "degree_correction": degree_mode, "openmp_threads": 1,
    }


def _targets(memberships: list[int], config: dict, seed: int) -> set[int]:
    rng = np.random.default_rng(seed)
    fraction = float(config.get("fraction", 0.25))
    if not 0 <= fraction <= 1:
        raise ValueError("Target fraction must lie in [0,1]")
    count = int(round(len(memberships) * fraction))
    arrangement = config.get("arrangement", "concentrated")
    if arrangement == "weak":
        return set(int(node) for node in rng.choice(len(memberships), count, replace=False))
    groups = {group: rng.permutation([i for i, g in enumerate(memberships) if g == group]).tolist()
              for group in sorted(set(memberships))}
    if arrangement == "concentrated":
        order = [node for group in groups.values() for node in group]
    elif arrangement == "distributed":
        order = []
        for index in range(max(map(len, groups.values()))):
            order.extend(group[index] for group in groups.values() if index < len(group))
    else:
        raise ValueError("Target arrangement must be concentrated, distributed, or weak")
    return set(order[:count])


def world_diagnostics(graphs: list[nx.Graph], memberships: list[list[int]], attributes: np.ndarray) -> dict:
    frames = []
    for tick, (graph, blocks) in enumerate(zip(graphs, memberships)):
        degrees = np.asarray([graph.degree(node) for node in sorted(graph)], dtype=float)
        cross = sum(blocks[u] != blocks[v] for u, v in graph.edges)
        partitions = [{node for node in graph if blocks[node] == block} for block in set(blocks)]
        frames.append({
            "tick": tick, "accounts": len(graph), "edges": graph.number_of_edges(),
            "isolates": nx.number_of_isolates(graph), "components": nx.number_connected_components(graph),
            "largest_component": max(map(len, nx.connected_components(graph)), default=0),
            "degree_mean": float(degrees.mean()), "degree_max": float(degrees.max()),
            "degree_quantiles": np.quantile(degrees, [0, .25, .5, .75, 1]).tolist(),
            "cross_community_edge_fraction": cross / graph.number_of_edges() if graph.number_of_edges() else None,
            "measured_modularity": nx.community.modularity(graph, partitions) if graph.number_of_edges() else None,
            "membership_transition_fraction": (sum(a != b for a, b in zip(memberships[tick-1], blocks)) / len(blocks)
                                               if tick else 0.0),
        })
    if attributes.shape[1]:
        blocks = np.asarray(memberships[0])
        center = attributes.mean(axis=0)
        total = float(np.sum((attributes-center)**2))
        between = sum(float(np.sum(blocks == group)) * float(np.sum((attributes[blocks == group].mean(axis=0)-center)**2))
                      for group in set(blocks))
        signal = between / total if total > 0 else 0.0
    else:
        signal = None
    return {"snapshots": frames, "initial_community_attribute_eta_squared": signal}


def generate_world(config: dict) -> World:
    """Build a complete policy-independent world; truth stays evaluator-private."""
    config = deepcopy(config)
    seed = int(config.get("seed", 17))
    seeds = {name: stream_seed(seed, name) for name in (
        "topology", "diffusion", "content", "identity", "target", "profiles",
        "provider", "feedback", "policy", "analysis")}
    graphs, memberships, attributes, generation = topology(config.get("world", {}), seeds["topology"])
    count = len(graphs[0])
    identity_rng = np.random.default_rng(seeds["identity"])
    opaque = ["a:" + identity_rng.bytes(12).hex() for _ in range(count)]
    if len(set(opaque)) != count:
        raise AssertionError("Account identity collision")
    targets = _targets(memberships[0], config.get("target", {}), seeds["target"])
    # Profiles provide noisy ordinary topical evidence, never explicit labels.
    profile_rng = np.random.default_rng(seeds["profiles"])
    accounts = []
    for node in range(count):
        lexical_niche = (node in targets) != (profile_rng.random() < 0.25)
        words = NICHE_VOCABULARY if lexical_niche else OTHER_VOCABULARY
        accounts.append({"id": opaque[node], "profile": str(profile_rng.choice(words)) + " " + str(profile_rng.choice(SHARED_VOCABULARY)),
                         "available_at": 0, "attributes": attributes[node].tolist()})
    accounts.sort(key=lambda account: account["id"])
    snapshots = [sorted([sorted((opaque[u], opaque[v])) for u, v in graph.edges]) for graph in graphs]
    states, diffusion = simulate_diffusion(opaque, snapshots, config.get("diffusion", {}), seeds["diffusion"])
    relevant = {opaque[node] for node in targets}
    posts, content = emit_content(accounts, snapshots, states, relevant, config.get("content", {}), seeds["content"])
    truth = {
        "relevant_accounts": sorted(relevant),
        "regions": {opaque[node]: "region:" + str(memberships[0][node]) for node in range(count)},
        "memberships": [{opaque[node]: int(frame[node]) for node in range(count)} for frame in memberships],
        "diffusion_states": states,
        "target_definition": {"version": "fixed-reference-relevance-v1", **config.get("target", {})},
    }
    metadata = {
        "generator": generation, "config": config, "config_hash": digest(config), "seeds": seeds,
        "versions": {name: version(name) for name in ("tadc-sbm", "ndlib", "dynetx", "numpy", "networkx", "networkx-temporal")},
        "diagnostics": world_diagnostics(graphs, memberships, attributes),
        "diffusion": diffusion, "content": content,
        "time": {"tick_interval": "[t,t+1)", "state_at_zero": "initial state without transition",
                 "emissions": "after diffusion iteration", "availability": "event_at + indexing delay",
                 "clock": "exogenous integer ticks, independent of acquisition actions"},
    }
    metadata["versions"]["graph-tool"] = generation["graph_tool_version"]
    return World(accounts=accounts, snapshots=snapshots, posts=posts, truth=truth, metadata=metadata)
