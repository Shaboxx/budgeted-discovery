"""Versioned research records. Only Action/PublicView cross the policy boundary."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import hashlib
import json

SCHEMA_VERSION = "1.0.0"
OPERATIONS = ("inspect", "posts", "neighbors", "search", "page", "revisit", "wait", "stop")
FEATURES = ("inspect", "posts", "neighbors", "search", "page", "revisit", "support", "relevance", "novelty", "age", "cost", "motif", "past_yield", "query_length")

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def stream_seed(seed: int, stream: str, *keys: Any) -> int:
    """Order-independent named child stream, also safe for legacy library seeds."""
    return 1 + int(digest([int(seed), stream, *keys])[:8], 16) % (2**31 - 2)

@dataclass
class World:
    """Evaluator/provider private. Never passed to a policy."""
    accounts: list[dict]
    snapshots: list[list[list[str]]]
    posts: list[dict]
    truth: dict
    metadata: dict
    schema_version: str = SCHEMA_VERSION

    @property
    def ticks(self) -> int:
        return len(self.snapshots)

@dataclass(frozen=True)
class Action:
    operation: str
    subject: str = ""
    query: tuple[str, ...] = ()
    base_operation: str = ""
    cursor: str = ""
    scope_tick: int = -1
    limit: int = 5
    since: int = 0
    until: int = -1
    direction: str = "undirected"
    provider_revision: str = "synthetic-bm25s-v1"
    objective: str = "unique_relevant_accounts_v1"
    parent_id: str = ""

    @property
    def id(self) -> str:
        return "act:" + digest(asdict(self))[:20]

    def to_dict(self) -> dict:
        return {**asdict(self), "action_id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> Action:
        d = {k: v for k, v in data.items() if k != "action_id"}
        d["query"] = tuple(d.get("query", ()))
        return cls(**d)

@dataclass
class Receipt:
    action_id: str
    execution_status: str
    observation_status: str
    request_tick: int
    response_tick: int
    cost: float
    accounts: list[dict] = field(default_factory=list)
    posts: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    next_cursor: str = ""
    scope_tick: int = -1
    reason: str = ""
    provider_revision: str = "synthetic-bm25s-v1"

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass(frozen=True)
class PublicView:
    """Allowlisted policy input: no environment, provider, World, or truth handle."""
    tick: int
    actions: tuple[Action, ...]
    features: tuple[tuple[float, ...], ...]
    valid: tuple[bool, ...]
    remaining_requests: int
    remaining_cost: float
    known_accounts: int
    known_posts: int
    labels: tuple[tuple[str, float], ...]
    feature_names: tuple[str, ...] = FEATURES

@dataclass(frozen=True)
class Choice:
    slot: int
    probability: float
    policy: str
    arm: str = ""
    arm_probability: float = 1.0

