from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from frontier_bench.schemas import World,Action,canonical
from frontier_bench.config import resolve
from frontier_bench.environment import FrontierEnv
from frontier_bench.provider import SimulatedProvider

def fixture_world():
    accounts=[dict(id=f"a:{i}",profile="craft common" if i%2 else "garden common",attributes=[],available_at=0) for i in range(8)]
    posts=[dict(id=f"p:{t}:{i}",author=f"a:{i}",event_at=t,available_at=t,text="craft common",terms=["craft","common"],hashtags=["craft"],mentions=[]) for t in range(5) for i in range(8)]
    return World(accounts,[[[f"a:{i}",f"a:{i+1}"] for i in range(7)] for _ in range(5)],posts,{"relevant_accounts":["a:1","a:3"],"regions":{}},{"generator":"deterministic-test-fixture"})

def cfg(**patch):
    base={"provider":{"mode":"restricted","failure_probability":0.,"page_size":2},"feedback":{"delay":1},"initial":{"accounts":2,"vocabulary":["craft","common"]},"budget":{"requests":10,"cost":20.,"slots_per_tick":4}}
    for k,v in patch.items(): base.setdefault(k,{}).update(v)
    return resolve(base)

def choose(env,op):
    return next(i for i,a in enumerate(env.public_view().actions) if a.operation==op and env.public_view().valid[i])

def test_checker():
    check_env(FrontierEnv(fixture_world(),cfg()),skip_render_check=True)

def test_prefix_index_no_future_statistics():
    w=fixture_world(); other=deepcopy(w)
    other.posts.extend([dict(id=f"future:{i}",author="a:0",event_at=4,available_at=4,text="craft rare",terms=["craft","rare"],hashtags=[],mentions=[]) for i in range(30)])
    action=Action("search",query=("craft",),limit=2)
    p=SimulatedProvider(w,cfg()["provider"],1); q=SimulatedProvider(other,cfg()["provider"],1)
    assert asdict(p.execute(action,0))==asdict(q.execute(action,0))
    assert all(d["available_at"]<=0 for d in p._index(0)[0])

def test_operational_truth_invariance():
    w=fixture_world(); other=deepcopy(w); other.truth["relevant_accounts"]=[a["id"] for a in w.accounts]
    other.truth["regions"]={a["id"]:"secret-region" for a in w.accounts}
    a=FrontierEnv(w,cfg()); b=FrontierEnv(other,cfg()); a.reset(seed=7); b.reset(seed=7)
    for _ in range(7):
        assert a.public_view()==b.public_view()
        slot=choose(a,"search") if any(x.operation=="search" for x in a.public_view().actions) else 0
        ra=a.step(slot); rb=b.step(slot)
        assert ra[1:]==rb[1:]
        for k in ra[0]: np.testing.assert_array_equal(ra[0][k],rb[0][k])

def test_pagination_partial_and_cost():
    p=SimulatedProvider(fixture_world(),cfg()["provider"],1)
    a=Action("search",query=("craft",),limit=2); first=p.execute(a,0)
    assert first.observation_status=="partial" and len(first.posts)==2
    second=p.execute(Action("page",cursor=first.next_cursor,base_operation="search",query=a.query),1)
    assert second.cost==1 and not {x["id"] for x in first.posts}&{x["id"] for x in second.posts}
    assert all(p["available_at"]<=0 for p in second.posts)

def test_failure_distinct_from_zero():
    c=cfg(provider={"failure_probability":1.})
    p=SimulatedProvider(fixture_world(),c["provider"],1)
    r=p.execute(Action("search",query=("absent",)),0)
    assert r.execution_status=="failed" and r.cost>0 and r.observation_status=="unavailable"
    c["provider"]["failure_probability"]=0
    r=SimulatedProvider(fixture_world(),c["provider"],1).execute(Action("search",query=("absent",)),0)
    assert r.execution_status=="completed" and r.observation_status=="observed_zero"

def test_replay_world_immutable_and_isolated():
    w=fixture_world(); before=canonical(asdict(w)); a=FrontierEnv(w,cfg()); b=FrontierEnv(w,cfg())
    a.reset(seed=4); b.reset(seed=4)
    a.step(choose(a,"search")); assert a.public_view()!=b.public_view()
    assert canonical(asdict(w))==before
    c=FrontierEnv(w,cfg()); c.reset(seed=4); c.step(choose(c,"search"))
    assert a.public_view()==c.public_view() and a.visible_snapshot()==c.visible_snapshot()

def test_pending_invalid_wait_stop():
    e=FrontierEnv(fixture_world(),cfg(feedback={"delay":3})); e.reset(seed=2)
    _,reward,done,truncated,info=e.step(choose(e,"search"))
    assert reward==0 and info["feedback"]["pending_ids"] and not info["feedback"]["assessed_labels"]
    requests=e.requests; _,_,_,_,info=e.step(999); assert info["status"]=="invalid_action" and e.requests==requests
    e.step(0); _,_,terminated,truncated,info=e.step(1); assert terminated and not truncated and info["termination_reason"]=="stop"

def test_no_hidden_vocabulary_in_candidates():
    w=fixture_world(); w.posts.append(dict(id="secret",author="a:7",event_at=4,available_at=4,text="classifiedfuture",terms=["classifiedfuture"],hashtags=[],mentions=[]))
    e=FrontierEnv(w,cfg()); e.reset(seed=2)
    assert "classifiedfuture" not in canonical([a.to_dict() for a in e.public_view().actions])

def test_duplicate_credit_and_revisit_new_events():
    e=FrontierEnv(fixture_world(),cfg(provider={"mode":"complete"},feedback={"delay":0,"mode":"oracle"},actions={"revisit_after":1})); e.reset(seed=2)
    _,_,_,_,info=e.step(choose(e,"search")); first=info["feedback"]
    assert first["new_account_ids"]
    e.step(0); e.step(0); e.step(0)
    candidates=e.public_view().actions
    slot=next(i for i,a in enumerate(candidates) if a.operation=="revisit" and a.query==tuple(info["action"]["query"]))
    _,reward,_,_,second=e.step(slot)
    assert second["feedback"]["new_account_ids"]==[] and reward==0
    assert second["feedback"]["new_posts"]>0
    assert all(p["event_at"]<=e.tick for p in second["receipt"]["posts"])
