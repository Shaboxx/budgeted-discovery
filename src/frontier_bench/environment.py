"""Gymnasium facade over a private exogenous world and an allowlisted knowledge state."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .schemas import World, Action, Receipt, PublicView, FEATURES, stream_seed
from .config import resolve
from .provider import SimulatedProvider, words
from .knowledge import Knowledge

class FrontierEnv(gym.Env):
    metadata={"render_modes":["ansi"],"render_fps":1}

    def __init__(self, world: World, config: dict | None=None, render_mode: str | None=None):
        super().__init__()
        self._world=world; self.config=resolve(config); self.render_mode=render_mode
        n=self.config["actions"]["cap"]
        self.action_space=spaces.Discrete(n)
        self.observation_space=spaces.Dict({
            "candidates":spaces.Box(0.,1e6,(n,len(FEATURES)),np.float32),
            "mask":spaces.MultiBinary(n),
            "time":spaces.Box(0,1e6,(1,),np.int32),
            "budget":spaces.Box(0,1e9,(2,),np.float32),
            "summary":spaces.Box(0,1e9,(4,),np.float32),
        })
        self._closed=False

    def reset(self, *, seed: int | None=None, options: dict | None=None):
        super().reset(seed=seed)
        self.tick=0; self.steps=0; self.requests=0; self.spent=0.; self._done=False; self._closed=False; self.feature_seconds=0.
        self._provider=SimulatedProvider(self._world,self.config["provider"],stream_seed(self.config["seed"],"provider"))
        self._knowledge=Knowledge(self.config["initial"]["vocabulary"])
        self._pending=[]; self._observations=[]; self._next_outcome=0
        # Declared initial account exposure: uniform identities, never target-seeded.
        ids=sorted(a["id"] for a in self._world.accounts if a.get("available_at",0)<=0)
        count=min(len(ids),int(self.config["initial"]["accounts"]))
        initial=sorted(self.np_random.choice(ids,size=count,replace=False).tolist()) if count else []
        for identity in initial:
            a=self._provider.account(identity,0)
            self._knowledge.ingest(Receipt("initial","completed","complete",0,0,0,accounts=[a]),Action("inspect",subject=identity))
        self._initial_ids=set(initial)
        self._initial_observations=deepcopy([self._knowledge.accounts[i] for i in initial])
        self._refresh_candidates()
        return self._observation(), {"schema_version":"1.0.0","initial_accounts":deepcopy(self._initial_observations),"overflow":self._overflow}

    def _refresh_candidates(self):
        started=time.perf_counter()
        self._actions,self._overflow=self._knowledge.candidates(self.tick,self.config["actions"],self.config["provider"])
        self._features=[self._knowledge.features(a,self.tick,self.config["provider"]) for a in self._actions]
        self._valid=[a.operation in ("wait","stop") or (self.requests<self.config["budget"]["requests"] and self.spent+self._provider.cost(a)<=self.config["budget"]["cost"]+1e-10) for a in self._actions]
        self.feature_seconds+=time.perf_counter()-started

    def _observation(self):
        n=self.action_space.n; features=np.zeros((n,len(FEATURES)),dtype=np.float32); mask=np.zeros(n,dtype=np.int8)
        features[:len(self._features)]=self._features; mask[:len(self._valid)]=self._valid
        return {"candidates":features,"mask":mask,"time":np.array([self.tick],dtype=np.int32),"budget":np.array([max(0,self.config["budget"]["requests"]-self.requests),max(0,self.config["budget"]["cost"]-self.spent)],dtype=np.float32),"summary":np.array([len(self._knowledge.accounts),len(self._knowledge.posts),len(self._knowledge.labels),self._overflow],dtype=np.float32)}

    def public_view(self) -> PublicView:
        return PublicView(self.tick,tuple(self._actions),tuple(self._features),tuple(self._valid),max(0,self.config["budget"]["requests"]-self.requests),max(0.,self.config["budget"]["cost"]-self.spent),len(self._knowledge.accounts),len(self._knowledge.posts),tuple(sorted(self._knowledge.labels.items())))

    def _label(self, identity: str) -> float:
        f=self.config["feedback"]
        if f["mode"]=="oracle": return float(identity in set(self._world.truth["relevant_accounts"]))
        # Operational estimates derive ONLY from currently observed text, not evaluator truth.
        a=self._knowledge.accounts[identity]
        visible=a.get("profile","")+" "+" ".join(p["text"] for p in self._knowledge.posts.values() if p["author"]==identity)
        tokens=set(words(visible)); overlap=tokens.intersection(self._knowledge.seed_terms)
        value=0.8 if overlap else 0.2
        rng=np.random.default_rng(stream_seed(self.config["seed"],"feedback",identity,self.tick))
        return 1-value if rng.random()<f["noise"] else value

    def _deliver_labels(self):
        remaining=[]; updates=[]
        for pending in self._pending:
            if pending["due"]<=self.tick:
                labels=[{"id":i,"value":self._label(i)} for i in pending["ids"]]
                for label in labels: self._knowledge.labels[label["id"]]=label["value"]
                updates.append({"outcome_id":pending["outcome_id"],"labels":labels,"action":pending["action"],"features":pending["features"],"available_at":self.tick})
            else: remaining.append(pending)
        self._pending=remaining
        return updates

    def step(self, action: int):
        if self._done or self._closed: raise RuntimeError("reset required before stepping completed/closed environment")
        info={"feedback_updates":[]}; reward=0.; stopped=False
        if not self.action_space.contains(action) or int(action)>=len(self._actions) or not self._valid[int(action)]:
            info.update({"status":"invalid_action","reason":"slot absent or not affordable","admission_status":"rejected"})
        else:
            a=self._actions[int(action)]; features=self._features[int(action)]
            info["action"]=a.to_dict(); info["features"]=list(features)
            if a.operation=="stop": stopped=True; info["status"]="stopped"
            elif a.operation=="wait": info["status"]="waited"
            else:
                receipt=self._provider.execute(a,self.tick)
                if receipt.execution_status!="admission_failed": self.requests+=1; self.spent+=receipt.cost
                self.tick=max(self.tick,receipt.response_tick)
                observed=self._knowledge.ingest(receipt,a)
                outcome_id=f"outcome:{self._next_outcome}"; self._next_outcome+=1
                new_ids=observed["new_account_ids"]
                feedback={**observed,"outcome_id":outcome_id,"assessed_labels":[],"pending_ids":list(new_ids),"status":receipt.execution_status,"observation_status":receipt.observation_status,"cost":receipt.cost}
                if new_ids:
                    due=self.tick+int(self.config["feedback"]["delay"])
                    self._pending.append({"due":due,"outcome_id":outcome_id,"ids":new_ids,"action":a.to_dict(),"features":list(features)})
                updates=self._deliver_labels()
                for update in updates:
                    if update["outcome_id"]==outcome_id:
                        feedback["assessed_labels"]=update["labels"]; feedback["pending_ids"]=[]
                        reward=sum(v["value"] for v in update["labels"])
                    else: info["feedback_updates"].append(update)
                # No true evaluator score is returned in operational mode.
                info.update({"status":receipt.execution_status,"receipt":receipt.to_dict(),"feedback":feedback})
                self._observations.append({"receipt":receipt.to_dict(),"feedback":deepcopy(feedback)})
                base=a.base_operation if a.operation=="revisit" else a.operation
                if receipt.execution_status=="completed": self._knowledge.past_yield[(base,a.subject,a.query)]=min(1.,len(new_ids)/max(1,len(receipt.accounts)))
        self.steps+=1
        if self.steps%self.config["budget"]["slots_per_tick"]==0: self.tick+=1
        info["feedback_updates"].extend(self._deliver_labels())
        exhausted=self.requests>=self.config["budget"]["requests"] or self.spent>=self.config["budget"]["cost"]-1e-10
        terminated=stopped or exhausted or self.tick>=self._world.ticks
        truncated=not terminated and self.steps>=self.config["budget"]["max_steps"]
        info["termination_reason"]="stop" if stopped else "budget" if exhausted else "horizon" if self.tick>=self._world.ticks else "watchdog" if truncated else None
        self._done=terminated or truncated
        self._refresh_candidates()
        return self._observation(),float(reward),bool(terminated),bool(truncated),info

    def visible_snapshot(self) -> dict:
        return {"accounts":deepcopy(self._knowledge.accounts),"posts":deepcopy(self._knowledge.posts),"labels":dict(self._knowledge.labels),"first_observed":dict(self._knowledge.first_observed),"initial_ids":sorted(self._initial_ids),"pending_ids":sorted({i for p in self._pending for i in p["ids"]}),"requests":self.requests,"cost":self.spent,"tick":self.tick}

    def render(self):
        return f"tick={self.tick} observed_accounts={len(self._knowledge.accounts)} observed_posts={len(self._knowledge.posts)} requests={self.requests} cost={self.spent:.2f}"

    def close(self): self._closed=True

if "DiscoveryFrontier-v0" not in gym.registry:
    gym.register(id="DiscoveryFrontier-v0",entry_point="frontier_bench.environment:FrontierEnv")
