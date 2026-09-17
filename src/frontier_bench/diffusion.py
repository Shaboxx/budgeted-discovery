"""Precomputed NDlib adoption histories over an exact account projection.

Only upstream NDlib executes transitions. The DyNetx wrapper preserves the
declared snapshot calendar and isolated accounts, which ordinary time_slice
otherwise drops. This module imports generation dependencies lazily.
"""
from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Iterable

import numpy as np


@contextmanager
def legacy_random_stream(seed: int):
    """Contain libraries which mutate Python/legacy NumPy global RNG state."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def to_dynamic_graph(accounts: Iterable[str], snapshots: list, *, directed=False):
    """Return DyNetx with complete account membership and an explicit calendar.

    Edges are present only in snapshots where supplied. No artificial edges or
    self-loops are added to represent empty snapshots/isolated accounts.
    NDlib receives NetworkX Graph/DiGraph exact single-snapshot slices.
    """
    import dynetx as dn
    import networkx as nx

    account_ids = tuple(sorted(accounts))
    allowed = set(account_ids)
    if not snapshots:
        raise ValueError("At least one temporal snapshot is required")
    base = dn.DynDiGraph if directed else dn.DynGraph

    class CalendarDynamicGraph:
        def __init__(self):
            # DyNetx 0.3.2 uses super(self.__class__, self) in __init__,
            # which recurses for subclasses. Composition avoids altering it.
            self.dynamic = base(edge_removal=True)
            self.dynamic.add_nodes_from(account_ids)

        def __getattr__(self, name):
            return getattr(self.dynamic, name)

        def temporal_snapshots_ids(self):
            return list(range(len(snapshots)))

        def time_slice(self, t_from, t_to=None):
            if t_to is not None and t_to != t_from:
                raise ValueError("Diffusion wrapper permits exact single-tick slices only")
            if not 0 <= t_from < len(snapshots):
                raise ValueError("Snapshot outside the declared calendar")
            result = nx.DiGraph() if directed else nx.Graph()
            result.add_nodes_from(account_ids)
            # DyNetx 0.3.2 directed interactions_iter incorrectly suppresses
            # arcs to an already-seen node. Temporal successors preserve arcs.
            if directed:
                result.add_edges_from((source, target) for source in account_ids
                                      for target in self.dynamic.successors(source, t=t_from))
            else:
                result.add_edges_from(edge[:2] for edge in self.dynamic.interactions(t=t_from))
            return result

    dynamic = CalendarDynamicGraph()
    for tick, edges in enumerate(snapshots):
        for edge in edges:
            u, v = edge[:2]
            if u not in allowed or v not in allowed:
                raise ValueError("Interaction references an undeclared account")
            if u != v:
                dynamic.add_interaction(u, v, t=tick)
    return dynamic


def reconstruct_states(iterations: list[dict], accounts: Iterable[str]) -> list[dict]:
    """Apply NDlib's initial full state followed by incremental status deltas."""
    allowed = set(accounts)
    if not iterations or set(iterations[0]["status"]) != allowed:
        raise ValueError("NDlib initial iteration must contain every declared account")
    state, states = {}, []
    for index, result in enumerate(iterations):
        if result["iteration"] != index:
            raise ValueError("Non-contiguous NDlib iteration IDs")
        if not set(result["status"]).issubset(allowed):
            raise ValueError("Unexpected account in NDlib delta")
        state.update({node: int(status) for node, status in result["status"].items()})
        states.append(dict(sorted(state.items())))
    return states


def simulate_diffusion(accounts: Iterable[str], snapshots: list,
                       config: dict, seed: int) -> tuple[list[dict], dict]:
    """Execute one dynamic NDlib process before policy execution.

    Tick zero is NDlib's initial full-state iteration, not a transition. Later
    tick t updates using contacts at t. Initial adopters are sampled without
    reference labels/communities and the same model persists for the whole run.
    """
    import ndlib.models.dynamic as dynamic_models
    import ndlib.models.ModelConfig as model_config

    nodes = tuple(sorted(accounts))
    if not nodes:
        raise ValueError("Diffusion requires accounts")
    model_name = config.get("model", "DynSIR")
    if model_name not in {"DynSIR", "DynSIS"}:
        raise ValueError("Supported dynamic models are DynSIR and DynSIS")
    beta = float(config.get("beta", 0.2))
    recovery = float(config.get("gamma" if model_name == "DynSIR" else "lambda", 0.15))
    fraction = float(config.get("initial_fraction", 0.05))
    if not all(0 <= value <= 1 for value in (beta, recovery, fraction)):
        raise ValueError("Diffusion probabilities must lie in [0, 1]")
    if config.get("initial_accounts") is not None:
        initial = sorted(set(config["initial_accounts"]))
        if not set(initial).issubset(nodes):
            raise ValueError("Initial adopters must be declared accounts")
    else:
        count = min(len(nodes), max(1, int(round(fraction * len(nodes)))))
        initial = sorted(np.random.default_rng(seed).choice(nodes, count, replace=False).tolist())
    if not initial:
        raise ValueError("NDlib dynamic models require a nonempty initial adopter set")
    graph = to_dynamic_graph(nodes, snapshots, directed=bool(config.get("directed", False)))
    model_class = (dynamic_models.DynSIRModel if model_name == "DynSIR"
                   else dynamic_models.DynSISModel)
    with legacy_random_stream(seed):
        model = model_class(graph, seed=int(seed))
        parameters = model_config.Configuration()
        parameters.add_model_parameter("beta", beta)
        parameters.add_model_parameter("gamma" if model_name == "DynSIR" else "lambda", recovery)
        parameters.add_model_initial_configuration("Infected", initial)
        model.set_initial_status(parameters)
        iterations = model.execute_snapshots(node_status=True)
    states = reconstruct_states(iterations, nodes)
    if len(states) != len(snapshots):
        raise AssertionError("Diffusion timeline does not match topology")
    counts = [{str(status): sum(value == status for value in state.values())
               for status in (0, 1, 2)} for state in states]
    return states, {
        "model": model_name, "beta": beta, "recovery": recovery,
        "initial_accounts": initial, "seed": int(seed),
        "execution": "NDlib.execute_snapshots", "initial_iteration_has_transition": False,
        "snapshot_wrapper": "exact-edges-persistent-accounts-v1",
        "directed": bool(config.get("directed", False)), "counts": counts,
        "status_delta_sizes": [len(result["status"]) for result in iterations],
        "interpretation": "synthetic topic adoption; never permanent reference relevance",
    }
