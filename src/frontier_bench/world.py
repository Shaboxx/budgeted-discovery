"""Portable canonical world storage; generation dependencies are never imported here."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import networkx as nx
import numpy as np
from .schemas import World, SCHEMA_VERSION, canonical, digest

def validate_world(w: World) -> dict:
    if w.schema_version!=SCHEMA_VERSION: raise ValueError("unsupported world schema")
    ids=[a["id"] for a in w.accounts]; known=set(ids)
    if not ids: raise ValueError("empty account population")
    if any(i.startswith("term:") for i in ids): raise ValueError("reserved term namespace")
    if len(ids)!=len(known): raise ValueError("duplicate account identity")
    if not w.snapshots: raise ValueError("missing snapshots")
    if not set(w.truth.get("relevant_accounts",[]))<=known: raise ValueError("unknown target ID")
    for edges in w.snapshots:
        seen=set()
        for u,v in edges:
            if u not in known or v not in known: raise ValueError("edge references unknown account")
            if u==v: raise ValueError("self loops excluded by projection")
            if tuple(sorted((u,v))) in seen: raise ValueError("duplicate undirected edge")
            seen.add(tuple(sorted((u,v))))
    post_ids=set()
    for p in w.posts:
        if p["id"] in known or p["id"].startswith("term:"): raise ValueError("entity identity has conflicting types")
        if p["id"] in post_ids: raise ValueError("duplicate post")
        post_ids.add(p["id"])
        if p["author"] not in known: raise ValueError("unknown author")
        if p["available_at"]<p["event_at"] or p["event_at"]<0: raise ValueError("invalid event availability")
        if any(a not in known for a in p.get("mentions",[])): raise ValueError("unknown mention")
        if not isinstance(p.get("terms"),list): raise ValueError("terms must be explicit")
    diagnostics=[]
    memberships=w.truth.get("memberships",[])
    for tick,edges in enumerate(w.snapshots):
        g=nx.Graph(); g.add_nodes_from(ids); g.add_edges_from(edges)
        degrees=list(dict(g.degree()).values()); memb=memberships[tick] if memberships else {}
        cross=sum(memb.get(u)!=memb.get(v) for u,v in edges) if memb else None
        diagnostics.append(dict(tick=tick,accounts=len(ids),edges=len(edges),isolates=nx.number_of_isolates(g),components=nx.number_connected_components(g),mean_degree=float(np.mean(degrees)),max_degree=max(degrees,default=0),degree_quantiles=[float(x) for x in np.quantile(degrees,[0,.25,.5,.75,1])],between_edge_fraction=cross/len(edges) if edges and cross is not None else None))
    return {"valid":True,"schema_version":SCHEMA_VERSION,"accounts":len(ids),"posts":len(w.posts),"ticks":w.ticks,"snapshots":diagnostics}

def save_world(w: World, destination: str | Path) -> Path:
    validate_world(w); path=Path(destination)
    if (path/"manifest.json").exists(): raise FileExistsError(f"World already exists: {path}")
    path.mkdir(parents=True,exist_ok=True)
    payload=asdict(w); truth=payload.pop("truth")
    contents={"world.json":payload,"truth.json":truth,"diagnostics.json":validate_world(w)}
    hashes={}
    for name,data in contents.items():
        raw=(canonical(data)+"\n").encode(); (path/name).write_bytes(raw); hashes[name]=hashlib.sha256(raw).hexdigest()
    # Explicit interchange edge table; snapshots are exact states, not an
    # accumulating edge-event list. Each interaction row applies to [tick,tick+1).
    typed=[]
    for tick,edges in enumerate(w.snapshots):
        typed.extend(dict(source=u,target=v,type="INTERACTS_WITH",event_at=tick,available_at=tick,valid_until=tick+1) for u,v in edges)
    for p in w.posts:
        typed.append(dict(source=p["author"],target=p["id"],type="PUBLISHED",event_at=p["event_at"],available_at=p["available_at"]))
        typed.extend(dict(source=p["id"],target="term:"+t,type="HAS_HASHTAG",event_at=p["event_at"],available_at=p["available_at"]) for t in p.get("hashtags",[]))
        typed.extend(dict(source=p["id"],target=a,type="MENTIONS",event_at=p["event_at"],available_at=p["available_at"]) for a in p.get("mentions",[]))
    raw=("\n".join(canonical(e) for e in typed)+("\n" if typed else "")).encode()
    (path/"typed-events.jsonl").write_bytes(raw); hashes["typed-events.jsonl"]=hashlib.sha256(raw).hexdigest()
    manifest={"schema_version":SCHEMA_VERSION,"world_hash":digest({"world":payload,"truth":truth}),"world_content_hash":world_content_hash(w),"files":hashes,"generator":w.metadata.get("generator"),"config_hash":digest(w.metadata.get("config",{}))}
    (path/"manifest.json").write_text(canonical(manifest)+"\n",encoding="utf-8")
    return path

def load_world(path: str | Path) -> World:
    path=Path(path); m=json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if m["schema_version"]!=SCHEMA_VERSION: raise ValueError("unsupported world schema")
    if not {"world.json","truth.json","diagnostics.json"}<=set(m["files"]): raise ValueError("incomplete artifact manifest")
    for name,checksum in m["files"].items():
        if Path(name).name!=name: raise ValueError("unsafe artifact member")
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=checksum: raise ValueError(f"World checksum failure: {name}")
    data=json.loads((path/"world.json").read_text(encoding="utf-8")); data["truth"]=json.loads((path/"truth.json").read_text(encoding="utf-8"))
    w=World(**data); validate_world(w)
    if world_hash(w)!=m["world_hash"]: raise ValueError("world identity checksum failure")
    return w

def world_hash(w: World) -> str:
    d=asdict(w); truth=d.pop("truth"); return digest({"world":d,"truth":truth})

def world_content_hash(w: World) -> str:
    """Content identity only: accounts, snapshots, posts and truth.

    `world_hash` (unchanged, artifact identity) also covers `metadata`, which embeds the
    generating configuration including `experiment.output` and the first scheduled
    policy name. Two byte-identical worlds saved under different output paths therefore
    have different `world_hash` values. This additive identity lets cross-directory
    pairing prove that two worlds share the same exogenous content.
    """
    return digest({"accounts":w.accounts,"snapshots":w.snapshots,"posts":w.posts,"truth":w.truth})

def import_events(accounts_path: str, edges_path: str, posts_path: str, target_path: str, ticks: int) -> World:
    """Explicit JSONL import boundary for later semi-synthetic experiments."""
    read=lambda p:[json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    snapshots=[[] for _ in range(ticks)]
    for e in read(edges_path):
        if e.get("type")!="INTERACTS_WITH": raise ValueError("import requires account interaction edges")
        if not 0<=int(e["tick"])<ticks: raise ValueError("edge tick outside horizon")
        snapshots[int(e["tick"])].append([e["source"],e["target"]])
    truth=json.loads(Path(target_path).read_text(encoding="utf-8"))
    w=World(read(accounts_path),snapshots,read(posts_path),truth,{"generator":"imported-semi-synthetic","source_paths_excluded":True})
    validate_world(w); return w
