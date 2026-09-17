"""Single configuration mechanism: YAML resolved against typed defaults.

No hidden environment-variable interpolation; unknown keys are rejected.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from copy import deepcopy
import yaml
from .schemas import digest

@dataclass
class WorldConfig:
    generator: str = "tadc"
    accounts: int = 120
    snapshots: int = 8
    communities: int = 4
    num_edges: int = 600
    eta: float = 0.85
    gamma: int = 1
    intra_probability: float = 0.8
    inter_probability: float = 0.2
    edge_sampling_rate: float = 1.0
    feature_dim: int = 4
    feature_center_distance: float = 1.0
    feature_cluster_variance: float = 1.0
    degree_correction: bool = True
    degree_sigma: float = 1.0
    feature_match: str = "GROUPED"
    feature_groups: int | None = None
    pi: list | None = None
    prop_mat: list | None = None
    tau_mat: list | None = None
    out_degs: list | None = None
    uniform_all: bool = False
    reverse_snapshot_order: bool = False
    stationary: bool = False

@dataclass
class Config:
    seed: int = 41
    world: dict = field(default_factory=lambda: asdict(WorldConfig()))
    diffusion: dict = field(default_factory=lambda: {"model":"DynSIR", "beta":0.25, "gamma":0.15, "lambda":0.15, "initial_fraction":0.05,"directed":False,"initial_accounts":None})
    content: dict = field(default_factory=lambda: dict(background_probability=0.3, topic_probability=0.8, index_delay=1, stationary=False,fixture="",fixture_posts_per_tick=1,index_delay_jitter=0,ambiguity_probability=0.15,hashtag_probability=0.6,mention_probability=0.1,change_tick=None,change_multiplier=2))
    target: dict = field(default_factory=lambda: dict(arrangement="concentrated", fraction=0.25))
    initial: dict = field(default_factory=lambda: dict(accounts=3, vocabulary=["tabletop", "dice"]))
    provider: dict = field(default_factory=lambda: dict(mode="restricted", page_size=5, failure_probability=0.05, unavailable_operations=[], ranking_noise=0.05, recency_weight=0.15, latency=0, costs={"inspect":1.0,"posts":1.0,"neighbors":1.0,"search":1.5,"page":1.0,"revisit":1.0}))
    feedback: dict = field(default_factory=lambda: dict(mode="operational", delay=1, noise=0.1))
    actions: dict = field(default_factory=lambda: dict(cap=64, sparse_probes=True, min_term_support=2, revisit_after=2, max_terms=3, branch_history=12, proposals_per_family=16))
    budget: dict = field(default_factory=lambda: dict(requests=24, cost=32.0, slots_per_tick=4, max_steps=64))
    policy: dict = field(default_factory=lambda: dict(name="random", epsilon=0.15, learning_rate=0.02, graph_weight=0.5, use_motif=True, pretrained=False))
    evaluation: dict = field(default_factory=lambda: dict(bootstrap_samples=300, confidence=0.95, panel_size=8, growth_threshold=0.25))
    experiment: dict = field(default_factory=lambda: dict(world_path=None, output="runs/smoke", policies=["random","graph","query","heuristic","fixed","round_robin","learned"], seeds=[41,42], conditions=["stationary_complete","stationary_restricted","evolving_complete","evolving_restricted"], arrangements=["concentrated","distributed","weak"], max_runs=12, max_minutes=5.0, max_storage_mb=200, allow_paper=False, phase="test", fixed_weights=[0.25,0.5,0.75],fixed_selection=None))

def _merge(base: dict, patch: dict, prefix: str = "") -> dict:
    for k,v in patch.items():
        if k not in base:
            raise ValueError(f"Unknown config key {prefix}{k}")
        if isinstance(base[k],dict):
            if not isinstance(v,dict): raise ValueError(f"{prefix}{k} must be a mapping")
            _merge(base[k],v,prefix+k+".")
        else: base[k]=v
    return base

def resolve(patch: dict | None = None) -> dict:
    cfg=_merge(asdict(Config()),deepcopy(patch or {}))
    w=cfg["world"]; b=cfg["budget"]
    if not 2 <= w["accounts"] <= 100000: raise ValueError("accounts outside supported bound")
    if not 1 <= w["snapshots"] <= 1000: raise ValueError("snapshots outside supported bound")
    if not 2 <= w["communities"] <= w["accounts"]: raise ValueError("invalid communities (minimum two)")
    if cfg["actions"]["cap"] < 8: raise ValueError("candidate cap must be at least 8")
    if b["requests"] < 1 or b["cost"] <= 0 or b["slots_per_tick"] < 1: raise ValueError("invalid budget")
    if cfg["feedback"]["mode"] not in ("oracle","operational"): raise ValueError("unknown feedback mode")
    if cfg["provider"]["mode"] not in ("complete","restricted"): raise ValueError("unknown provider mode")
    if any(float(v)<=0 for v in cfg["provider"]["costs"].values()): raise ValueError("costs must be positive")
    if cfg["provider"]["latency"] < 0: raise ValueError("negative latency")
    if cfg["provider"]["page_size"]<1: raise ValueError("page_size must be positive")
    if cfg["actions"]["proposals_per_family"]<1 or cfg["budget"]["max_steps"]<1: raise ValueError("positive cap/step limits required")
    if any(cfg["experiment"][key]<=0 for key in ("max_runs","max_minutes","max_storage_mb")): raise ValueError("positive resource limits required")
    if cfg["target"]["fraction"]<=0: raise ValueError("target fraction must be positive")
    if cfg["feedback"]["delay"]<0 or cfg["actions"]["revisit_after"]<1: raise ValueError("invalid delay/cooldown")
    for group,key in [("provider","failure_probability"),("feedback","noise"),("target","fraction"),("content","background_probability"),("content","topic_probability"),("policy","epsilon"),("policy","graph_weight")]:
        if not 0<=cfg[group][key]<=1: raise ValueError(f"{group}.{key} outside [0,1]")
    if cfg["policy"]["pretrained"]: raise ValueError("pretrained experience is not implemented; cold-start only")
    if w["accounts"]>2000 and not cfg["experiment"]["allow_paper"]: raise ValueError("paper-size runs require explicit allow_paper")
    return cfg

def load_config(path: str | Path) -> dict:
    data=yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data,dict): raise ValueError("config root must be a mapping")
    return resolve(data)

def config_hash(cfg: dict) -> str:
    return digest(cfg)
