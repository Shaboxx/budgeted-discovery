"""Policy-visible evidence and candidate generation, without hidden-world access."""
from __future__ import annotations
from dataclasses import replace
from collections import defaultdict
from .schemas import Action, Receipt, FEATURES, digest
from .provider import words

class Knowledge:
    def __init__(self, seed_terms: list[str]):
        self.accounts={}; self.posts={}; self.edges={}; self.labels={}
        self.first_observed={}; self.term_authors=defaultdict(set)
        self.seed_terms=tuple(sorted(set(seed_terms))); self.history={}; self.branches=[]
        self.cursors={}; self.consumed_cursors=set(); self.past_yield={}; self.provenance=[]

    def ingest(self, receipt: Receipt, action: Action) -> dict:
        new_accounts=[]; new_posts=[]; duplicates=0
        for a in receipt.accounts:
            if a["id"] not in self.accounts:
                new_accounts.append(a["id"]); self.first_observed[a["id"]]=receipt.response_tick
            else: duplicates+=1
            self.accounts[a["id"]]=a
            for term in words(a.get("profile","")): self.term_authors[term].add(a["id"])
        for p in receipt.posts:
            if p["id"] not in self.posts:
                new_posts.append(p["id"]); self.first_observed[p["id"]]=receipt.response_tick
            else: duplicates+=1
            self.posts[p["id"]]=p
            for term in p.get("terms",[]): self.term_authors[term].add(p["author"])
        for e in receipt.edges: self.edges[digest(e)]=e
        if receipt.next_cursor:
            base=action.base_operation if action.operation in ("page","revisit") else action.operation
            self.cursors[receipt.next_cursor]=replace(action,operation="page",base_operation=base,cursor=receipt.next_cursor,scope_tick=receipt.scope_tick,parent_id=action.id)
        if action.operation=="page" and receipt.execution_status=="completed": self.consumed_cursors.add(action.cursor)
        if action.operation in ("search","revisit") and action.query and action.query not in [b.query for b in self.branches]: self.branches.append(action)
        base=action.base_operation if action.operation=="revisit" else action.operation
        if action.operation!="page": self.history[(base,action.subject,action.query)]=(receipt.response_tick,action,receipt.observation_status)
        self.provenance.append({"action_id":action.id,"observed_at":receipt.response_tick,"entities":new_accounts+new_posts,"class":"acquisition"})
        return dict(new_account_ids=new_accounts,new_post_ids=new_posts,new_posts=len(new_posts),duplicate_returns=duplicates,refresh_observations=len(receipt.posts)-len(new_posts))

    def candidates(self, tick: int, cfg: dict, provider_cfg: dict) -> tuple[list[Action],int]:
        page_size=int(provider_cfg["page_size"]); candidates={}; parents=defaultdict(set)
        def retain(a):
            physical=replace(a,parent_id="").id
            if a.parent_id: parents[physical].add(a.parent_id)
            # Physical request identity is independent of acquisition ancestry.
            candidates[physical]=replace(a,parent_id="")
        def add(a):
            if a.operation in provider_cfg["unavailable_operations"]: return
            key=(a.operation,a.subject,a.query)
            if key in self.history:
                last,old,status=self.history[key]
                if tick-last<cfg["revisit_after"]: return
                a=replace(a,operation="revisit",base_operation=old.operation if old.operation!="revisit" else old.base_operation,parent_id=old.id)
            retain(a)
        for identity in sorted(self.accounts):
            for op in ("inspect","posts","neighbors"): add(Action(op,subject=identity,limit=page_size))
        terms=set(self.seed_terms)
        for term,authors in self.term_authors.items():
            if cfg["sparse_probes"] or len(authors)>=cfg["min_term_support"]: terms.add(term)
        # Alphabetic vocabulary and branch caps are fixed, policy-independent rules.
        terms=sorted(terms)[:64]
        for t in terms: add(Action("search",query=(t,),limit=page_size))
        for parent in self.branches[-cfg["branch_history"]:]:
            if len(parent.query)<cfg["max_terms"]:
                for t in terms:
                    if t not in parent.query: add(Action("search",query=tuple(sorted((*parent.query,t))),limit=page_size,parent_id=parent.id))
            if len(parent.query)>1:
                for t in parent.query: add(Action("search",query=tuple(x for x in parent.query if x!=t),limit=page_size,parent_id=parent.id))
        for cur,a in self.cursors.items():
            if cur not in self.consumed_cursors: retain(a)
        # Fair cyclic admission by operation, then stable action hash: no proposed-policy score.
        groups=defaultdict(list)
        for a in candidates.values(): groups[a.operation].append(a)
        for op,group in groups.items():
            group.sort(key=lambda a:a.id)
            groups[op]=group[:cfg["proposals_per_family"]]
        chosen=[]; limit=cfg["cap"]-2
        while len(chosen)<limit and any(groups.values()):
            for op in sorted(groups):
                if groups[op] and len(chosen)<limit: chosen.append(groups[op].pop(0))
        self.candidate_lineage={a.id:sorted(parents[a.id]) for a in chosen if parents[a.id]}
        return [Action("wait"),Action("stop"),*chosen],max(0,len(candidates)-len(chosen))

    def features(self, action: Action, tick: int, provider_cfg: dict) -> tuple[float,...]:
        base=action.base_operation if action.operation in ("revisit","page") else action.operation
        authors=set().union(*(self.term_authors.get(t,set()) for t in action.query)) if action.query else ({action.subject} if action.subject else set())
        support=len(authors)
        known=[self.labels[a] for a in authors if a in self.labels]
        relevance=sum(known)/len(known) if known else 0.5
        hist=self.history.get((base,action.subject,action.query)); age=max(0,tick-hist[0]) if hist else tick+1
        neighbor_ids={e["target"] if e["source"]==action.subject else e["source"] for e in self.edges.values() if e["type"]=="INTERACTS_WITH" and action.subject in (e["source"],e["target"])}
        # Two observed relation families; bounded counts carry observation lineage in knowledge.
        motif=min(1.,len(neighbor_ids)/10+sum(len(self.term_authors.get(t,set()))>=2 for t in action.query)/3)
        cost=0. if action.operation in ("wait","stop") else provider_cfg["costs"][action.operation]
        result=[float(action.operation==op) for op in FEATURES[:6]]
        result.extend([min(1.,support/10),relevance,1. if not hist else 0.25,min(1.,age/10),cost,motif,self.past_yield.get((base,action.subject,action.query),0.),len(action.query)/3])
        return tuple(result)
