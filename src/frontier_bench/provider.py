"""Simulated content retrieval. Private index never crosses the policy boundary."""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
import re
import numpy as np
import bm25s
from .schemas import Action, Receipt, World, digest, stream_seed

def words(text: str) -> list[str]:
    return sorted(set(re.findall(r"[a-z][a-z0-9_]{1,30}",text.lower())))

class SimulatedProvider:
    revision="synthetic-bm25s-v1"
    capabilities=("inspect","posts","neighbors","search","page","revisit")

    def __init__(self, world: World, config: dict, seed: int):
        self._world=world; self.config=deepcopy(config); self.seed=seed
        self._accounts={a["id"]:a for a in world.accounts}
        self._indices={}; self._cursors={}

    def account(self, identity: str, at: int) -> dict | None:
        a=self._accounts.get(identity)
        if a is None or a.get("available_at",0)>at: return None
        return {k:deepcopy(a[k]) for k in ("id","profile","attributes","available_at") if k in a}

    def cost(self, action: Action) -> float:
        return float(self.config["costs"].get(action.operation,1.0))

    def _index(self, tick: int):
        if tick not in self._indices:
            docs=sorted((p for p in self._world.posts if p["available_at"]<=tick),key=lambda p:p["id"])
            index=None
            if docs:
                index=bm25s.BM25()
                index.index(bm25s.tokenize([p["text"] for p in docs],stopwords=[],show_progress=False),show_progress=False)
            self._indices[tick]=(docs,index)
        return self._indices[tick]

    def search(self, action: Action, tick: int) -> list[dict]:
        docs,index=self._index(tick)
        if not docs or not action.query: return []
        tokens=bm25s.tokenize([" ".join(action.query)],stopwords=[],show_progress=False)
        indices,scores=index.retrieve(tokens,k=len(docs),show_progress=False)
        ranked=[]
        for i,s in zip(indices[0],scores[0]):
            p=docs[int(i)]
            # Explicit conjunctive query semantics; BM25 orders matching documents.
            if not set(action.query)<=set(words(p["text"])): continue
            if p["event_at"]<action.since or (action.until>=0 and p["event_at"]>action.until): continue
            noise=np.random.default_rng(stream_seed(self.seed,"provider-rank",tick,action.query,p["id"])).normal(0,self.config["ranking_noise"])
            recency=1/(1+max(0,tick-p["event_at"]))
            ranked.append((float(s)+float(noise)+self.config["recency_weight"]*recency,p))
        return [p for _,p in sorted(ranked,key=lambda z:(-z[0],z[1]["id"]))]

    def execute(self, action: Action, tick: int) -> Receipt:
        response_tick=tick+int(self.config["latency"])
        r=Receipt(action.id,"completed","complete",tick,response_tick,self.cost(action),scope_tick=tick)
        if action.operation not in self.capabilities:
            r.execution_status="admission_failed"; r.observation_status="unavailable"; r.cost=0; r.reason="unsupported operation"; return r
        base=action; offset=0; scope=tick
        if action.operation=="page":
            entry=self._cursors.get(action.cursor)
            if entry is None:
                r.execution_status="admission_failed"; r.observation_status="unavailable"; r.cost=0; r.reason="invalid continuation"; return r
            base,scope,offset=entry
            if scope>tick:
                r.execution_status="admission_failed"; r.observation_status="unavailable"; r.cost=0; r.reason="continuation from a future scope"; return r
        elif action.operation=="revisit":
            base=replace(action,operation=action.base_operation,cursor="")
        if base.operation in self.config["unavailable_operations"]:
            r.execution_status="failed"; r.observation_status="unavailable"; r.reason="simulated unavailable operation"; return r
        probability=self.config["failure_probability"] if self.config["mode"]=="restricted" else 0
        physical_request={"operation":base.operation,"subject":base.subject,"query":base.query,"direction":base.direction,"since":base.since,"until":base.until,"limit":base.limit,"revision":base.provider_revision,"scope":scope,"offset":offset}
        draw=np.random.default_rng(stream_seed(self.seed,"provider-failure",tick,physical_request)).random()
        if draw<probability:
            r.execution_status="failed"; r.observation_status="unavailable"; r.reason="simulated executed failure"; return r
        r.scope_tick=scope
        rows=[]; kind="account"
        if base.operation=="inspect":
            a=self.account(base.subject,scope); rows=[] if a is None else [a]
        elif base.operation=="neighbors":
            if base.subject not in self._accounts: rows=[]
            else:
                edges=self._world.snapshots[min(scope,self._world.ticks-1)]
                neighbors={v if u==base.subject else u for u,v in edges if base.subject in (u,v)}
                rows=[a for n in sorted(neighbors) if (a:=self.account(n,scope)) is not None]
        elif base.operation=="posts":
            kind="post"
            rows=sorted((p for p in self._world.posts if p["author"]==base.subject and p["available_at"]<=scope and p["event_at"]>=base.since and (base.until<0 or p["event_at"]<=base.until)),key=lambda p:(-p["event_at"],p["id"]))
        elif base.operation=="search":
            kind="post"; rows=self.search(base,scope)
        else:
            r.execution_status="admission_failed"; r.observation_status="unavailable"; r.cost=0; r.reason="unsupported base operation"; return r
        limit=len(rows) if self.config["mode"]=="complete" else min(base.limit,self.config["page_size"])
        selected=rows[offset:offset+limit]
        if offset+len(selected)<len(rows):
            r.observation_status="partial"
            cursor="cursor:"+digest([base.to_dict(),scope,offset+len(selected)])[:24]
            self._cursors[cursor]=(base,scope,offset+len(selected)); r.next_cursor=cursor
        elif not selected: r.observation_status="observed_zero"
        if kind=="post":
            r.posts=[{k:deepcopy(p[k]) for k in ("id","author","event_at","available_at","text","terms","hashtags","mentions") if k in p} for p in selected]
            authors={p["author"] for p in selected}
            r.accounts=[a for n in sorted(authors) if (a:=self.account(n,scope)) is not None]
            for p in selected:
                r.edges.append({"source":p["author"],"target":p["id"],"type":"PUBLISHED","event_at":p["event_at"]})
                for tag in p.get("hashtags",[]): r.edges.append({"source":p["id"],"target":"term:"+tag,"type":"HAS_HASHTAG","event_at":p["event_at"]})
                for mention in p.get("mentions",[]):
                    r.edges.append({"source":p["id"],"target":mention,"type":"MENTIONS","event_at":p["event_at"]})
                    if mention not in authors and (a:=self.account(mention,scope)) is not None:
                        r.accounts.append(a); authors.add(mention)
        else:
            r.accounts=deepcopy(selected)
            if base.operation=="neighbors":
                r.edges=[{"source":base.subject,"target":a["id"],"type":"INTERACTS_WITH","event_at":scope} for a in selected]
        return r
