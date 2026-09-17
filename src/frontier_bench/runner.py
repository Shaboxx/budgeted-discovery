"""Bounded runs, append-only event logs, exact history replay, paired comparisons."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import importlib.metadata as metadata
import json
import hashlib
import os
import platform
import subprocess
import time
import traceback
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from .config import resolve, config_hash
from .schemas import Action, canonical, digest, SCHEMA_VERSION, FEATURES, stream_seed
from .world import load_world, save_world, world_hash, world_content_hash
from .environment import FrontierEnv
from .policies import make_policy, POLICY_VERSION
from .evaluation import evaluate, paired_summary, EVALUATOR_VERSION

def write_json(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(canonical(value)+"\n",encoding="utf-8")

def append(path: Path, value):
    with path.open("a",encoding="utf-8") as stream:
        stream.write(canonical(value)+"\n")
        stream.flush()

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def source_identity() -> dict:
    root=Path(__file__).resolve().parents[2]
    package=Path(__file__).resolve().parent
    files={"frontier_bench/"+str(p.relative_to(package)).replace("\\","/"):digest(p.read_text(encoding="utf-8")) for p in sorted(package.rglob("*.py"))}
    try:
        git_root=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()).resolve()
        if git_root != root:
            raise ValueError("Source directory is not the repository root")
        revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,stderr=subprocess.DEVNULL,text=True).strip()
        dirty=bool(subprocess.check_output(["git","status","--porcelain"],cwd=root,stderr=subprocess.DEVNULL,text=True).strip())
    except (FileNotFoundError,subprocess.CalledProcessError,ValueError):
        provenance=root/"source-provenance.json"
        if not provenance.exists(): provenance=Path("/opt/frontier-source-provenance.json")
        recorded=json.loads(provenance.read_text()) if provenance.exists() else {}
        revision=recorded.get("revision","unavailable-in-isolated-container"); dirty=recorded.get("dirty")
    return {"revision":revision,"dirty":dirty,"source_hash":digest(files),"files":files}

def dependency_versions() -> dict:
    result={"python":platform.python_version(),"platform":platform.platform()}
    for name in ("tadc-sbm","ndlib","dynetx","networkx","networkx-temporal","numpy","scipy","bm25s","gymnasium","river","pyarrow","duckdb"):
        try: result[name]=metadata.version(name)
        except metadata.PackageNotFoundError: result[name]=None
    return result

def view_record(env: FrontierEnv) -> dict:
    v=env.public_view()
    return dict(tick=v.tick,remaining_requests=v.remaining_requests,remaining_cost=v.remaining_cost,
        known_accounts=v.known_accounts,known_posts=v.known_posts,labels=list(v.labels),
        feature_names=list(FEATURES),actions=[a.to_dict() for a in v.actions],features=[list(x) for x in v.features],
        valid=list(v.valid),overflow=env._overflow,as_of=v.tick,
        evidence_prefix_hash=digest(env.visible_snapshot()),
        lineage=getattr(env._knowledge,"candidate_lineage",{}))

def receive(policy, info: dict):
    if "feedback" in info:
        policy.observe(Action.from_dict(info["action"]),tuple(info["features"]),info["feedback"])
    for update in info.get("feedback_updates",[]):
        policy.observe(Action.from_dict(update["action"]),tuple(update["features"]),update)

def run_one(world_path: str|Path, config: dict, destination: str|Path, *, deadline: float|None=None, tags: dict|None=None, storage_root: str|Path|None=None, storage_base_bytes: int|None=None) -> dict:
    cfg=resolve(config); path=Path(destination); path.mkdir(parents=True,exist_ok=True)
    if (path/"manifest.json").exists(): raise FileExistsError(f"Run exists: {path}")
    world_path=Path(world_path).resolve(); world=load_world(world_path)
    origin=world.metadata.get("config",{})
    origin_stationary=origin.get("world",{}).get("stationary",cfg["world"]["stationary"])
    origin_arrangement=world.truth.get("target_definition",{}).get("arrangement",origin.get("target",{}).get("arrangement",cfg["target"]["arrangement"]))
    storage_root=Path(storage_root or path)
    if storage_base_bytes is None:
        storage_base_bytes=sum(p.stat().st_size for p in storage_root.rglob("*") if p.is_file() and not p.is_relative_to(path))
    if deadline is None: deadline=time.perf_counter()+60*cfg["experiment"]["max_minutes"]
    identity=source_identity(); start=time.perf_counter(); cpu=time.process_time()
    process=psutil.Process(); peak=process.memory_info().rss; feature_seconds=0.; policy_seconds=0.
    manifest=dict(schema_version=SCHEMA_VERSION,config=cfg,config_hash=config_hash(cfg),world_hash=world_hash(world),world_content_hash=world_content_hash(world),
        world_path=os.path.relpath(world_path,path.resolve()),source=identity,dependencies=dependency_versions(),
        policy_version=POLICY_VERSION,evaluator_version=EVALUATOR_VERSION,initialization="cold-start; learner reset each run",
        stream_ids={name:stream_seed(cfg["seed"],name) for name in ("provider","feedback","policy_exploration","statistical_analysis")},
        target_definition=world.truth.get("target_definition",cfg["target"]),
        initial_exposure="declared uniform free seed set; excluded from incremental primary outcome",
        admission="cyclic operation families, stable physical action hashes, fixed cap; no policy score",
        status="started",tags=tags or {})
    manifest["stream_ids"]["policy_exploration"]=stream_seed(cfg["seed"],"policy_exploration",cfg["policy"]["name"])
    manifest["stream_derivation"]={"provider":"stream_seed(seed, provider), then request/rank identities","feedback":"stream_seed(seed, feedback, entity_id, delivery_tick)","policy":"stream_seed(seed, policy_exploration, policy_name)","statistical_analysis":"stream_seed(seed, statistical_analysis, policy_name)"}
    write_json(path/"manifest.json",manifest)
    for name in ("decisions","executions","outcomes","evaluation"):
        (path/f"{name}.jsonl").touch(exist_ok=False)
    env=FrontierEnv(world,cfg); policy=make_policy(cfg["policy"]["name"],cfg["seed"],cfg["policy"])
    rows=[]; status="completed"; failure=None; steps=0
    try:
        begin=time.perf_counter(); _,reset=env.reset(seed=cfg["seed"]); feature_seconds+=time.perf_counter()-begin
        write_json(path/"initial.json",reset)
        rows.append({"step":0,**evaluate(world,env.visible_snapshot())})
        append(path/"evaluation.jsonl",rows[-1])
        while True:
            if storage_base_bytes+sum(p.stat().st_size for p in path.iterdir() if p.is_file())>cfg["experiment"]["max_storage_mb"]*2**20:
                status="interrupted"; failure="storage safeguard; permits one operation and finalization overshoot"; break
            if deadline and time.perf_counter()>=deadline:
                status="interrupted"; failure="experiment wall-clock watchdog"; break
            record=view_record(env); v=env.public_view()
            begin=time.perf_counter(); choice=policy.choose(v); policy_seconds+=time.perf_counter()-begin
            decision={"schema_version":SCHEMA_VERSION,"step":steps,"view":record,"view_hash":digest(record),"choice":asdict(choice),"action_id":v.actions[choice.slot].id,
                "eligibility":"valid" if v.valid[choice.slot] else "invalid","scheduling":"selected","policy":policy.name,"policy_version":policy.version}
            append(path/"decisions.jsonl",decision)
            begin=time.perf_counter(); _,reward,terminated,truncated,info=env.step(choice.slot); feature_seconds+=time.perf_counter()-begin
            append(path/"executions.jsonl",dict(step=steps,reward=reward,terminated=terminated,truncated=truncated,info=info))
            if "feedback" in info: append(path/"outcomes.jsonl",dict(step=steps,kind="initial_observed_outcome",**info["feedback"]))
            for update in info.get("feedback_updates",[]): append(path/"outcomes.jsonl",dict(step=steps,kind="assessment_revision",**update))
            begin=time.perf_counter(); receive(policy,info); policy_seconds+=time.perf_counter()-begin
            steps+=1; rows.append({"step":steps,**evaluate(world,env.visible_snapshot())}); append(path/"evaluation.jsonl",rows[-1])
            peak=max(peak,process.memory_info().rss)
            if terminated or truncated:
                if truncated: status="truncated"; failure=info["termination_reason"]
                break
    except Exception as exc:
        status="failed"; failure=f"{type(exc).__name__}: {exc}"
        (path/"failure.txt").write_text(traceback.format_exc(),encoding="utf-8")
    finally: env.close()
    elapsed=time.perf_counter()-start; cpu_seconds=time.process_time()-cpu
    executions=read_jsonl(path/"executions.jsonl")
    receipts=[e["info"]["receipt"] for e in executions if "receipt" in e["info"]]
    outcomes=[e["info"]["feedback"] for e in executions if "feedback" in e["info"]]
    summary={**(rows[-1] if rows else {}),"policy":policy.name,"seed":cfg["seed"],"condition":f'{"stationary" if origin_stationary else "evolving"}_{cfg["provider"]["mode"]}',
        "arrangement":origin_arrangement,"sparse_probes":cfg["actions"]["sparse_probes"],"use_motif":cfg["policy"]["use_motif"],"regime":(tags or {}).get("regime","default"),
        "world_seed":origin.get("seed"),
        "acquisition_contract_hash":digest({k:cfg[k] for k in ("initial","provider","feedback","actions","budget")}),
        "graph_weight":cfg["policy"]["graph_weight"],"status":status,"failure":failure,"world_hash":manifest["world_hash"],"world_content_hash":manifest["world_content_hash"],"config_hash":manifest["config_hash"],
        "wall_seconds":elapsed,"cpu_seconds":cpu_seconds,"peak_rss_mb":peak/2**20,"peak_memory_scope":"sampled process RSS; includes reused interpreter/library caches",
        "provider_and_features_seconds":feature_seconds,"policy_seconds":policy_seconds,"learned_examples":getattr(policy,"learned_examples",0),
        "candidate_feature_seconds":env.feature_seconds,
        "failed_calls":sum(r["execution_status"]=="failed" for r in receipts),"partial_calls":sum(r["observation_status"]=="partial" for r in receipts),
        "zero_calls":sum(r["observation_status"]=="observed_zero" for r in receipts),"completed_calls":sum(r["execution_status"]=="completed" for r in receipts),
        "duplicate_returns":sum(o["duplicate_returns"] for o in outcomes),"returned_entities":sum(len(r["accounts"])+len(r["posts"]) for r in receipts),
        "refresh_observations":sum(o["refresh_observations"] for o in outcomes),
        "revisit_new_posts":sum(e["info"].get("feedback",{}).get("new_posts",0) for e in executions if e["info"].get("action",{}).get("operation")=="revisit"),
        "revisit_calls":sum(e["info"].get("action",{}).get("operation")=="revisit" for e in executions)}
    summary["duplicate_rate"]=summary["duplicate_returns"]/summary["returned_entities"] if summary["returned_entities"] else None
    summary["zero_rate"]=summary["zero_calls"]/summary["completed_calls"] if summary["completed_calls"] else None
    write_json(path/"summary.json",summary); write_json(path/"visible-final.json",env.visible_snapshot())
    if rows: pq.write_table(pa.Table.from_pylist(rows),path/"curves.parquet")
    manifest.update(status=status,failure=failure,checksum_algorithm="sha256-bytes",artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file() and p.name!="manifest.json"})
    write_json(path/"manifest.json",manifest)
    return summary

def replay(run_path: str|Path) -> dict:
    path=Path(run_path); manifest=json.loads((path/"manifest.json").read_text()); cfg=manifest["config"]
    if config_hash(cfg)!=manifest["config_hash"]: raise ValueError("run configuration hash mismatch")
    for name,checksum in manifest["artifacts"].items():
        if Path(name).name!=name: raise ValueError("unsafe run artifact path")
        actual=hashlib.sha256((path/name).read_bytes()).hexdigest() if manifest.get("checksum_algorithm")=="sha256-bytes" else digest((path/name).read_text(encoding="utf-8"))
        if actual!=checksum: raise ValueError(f"artifact checksum mismatch: {name}")
    world=load_world(path/manifest["world_path"])
    if world_hash(world)!=manifest["world_hash"]: raise ValueError("run world mismatch")
    env=FrontierEnv(world,cfg); _,initial=env.reset(seed=cfg["seed"])
    if canonical(initial)!=canonical(json.loads((path/"initial.json").read_text())): raise AssertionError("reset replay mismatch")
    decisions=read_jsonl(path/"decisions.jsonl"); executions=read_jsonl(path/"executions.jsonl"); evaluations=read_jsonl(path/"evaluation.jsonl")
    if len(decisions)!=len(executions): raise ValueError("incomplete decision/execution log")
    policy=make_policy(cfg["policy"]["name"],cfg["seed"],cfg["policy"])
    for d,e in zip(decisions,executions):
        if digest(view_record(env))!=d["view_hash"]: raise AssertionError(f"decision evidence mismatch at {d['step']}")
        if canonical(asdict(policy.choose(env.public_view())))!=canonical(d["choice"]): raise AssertionError("policy selection/propensity mismatch")
        _,reward,terminated,truncated,info=env.step(d["choice"]["slot"])
        actual=dict(step=d["step"],reward=reward,terminated=terminated,truncated=truncated,info=info)
        if canonical(actual)!=canonical(e): raise AssertionError(f"execution replay mismatch at {d['step']}")
        receive(policy,info)
        actual_eval={"step":d["step"]+1,**evaluate(world,env.visible_snapshot())}
        if canonical(actual_eval)!=canonical(evaluations[d["step"]+1]): raise AssertionError("evaluator replay mismatch")
    if canonical(env.visible_snapshot())!=canonical(json.loads((path/"visible-final.json").read_text())): raise AssertionError("final accounting mismatch")
    env.close()
    result={"verified":True,"decisions":len(decisions),"world_hash":manifest["world_hash"],"policy":cfg["policy"]["name"],"source_hash_matches":source_identity()["source_hash"]==manifest["source"]["source_hash"]}
    write_json(path/"replay.json",result); return result

def compare(config: dict, output: str|Path|None=None) -> dict:
    cfg=resolve(config); exp=cfg["experiment"]; root=Path(output or exp["output"])
    if (root/"comparison.json").exists(): raise FileExistsError(f"Comparison exists: {root}")
    root.mkdir(parents=True,exist_ok=True); write_json(root/"resolved-config.json",cfg)
    start=time.perf_counter(); deadline=start+60*exp["max_minutes"]
    summaries=[]; skipped=[]; worlds={}; failures=[]
    used_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    jobs=[(seed,arrangement,condition,policy) for seed in exp["seeds"] for arrangement in exp["arrangements"] for condition in exp["conditions"] for policy in exp["policies"]]
    for jobno,(seed,arrangement,condition,policy) in enumerate(jobs):
        label=f"{jobno:04d}-{seed}-{arrangement}-{condition}-{policy}"
        reason=None
        if len(summaries)+len(failures)>=exp["max_runs"]: reason="max-runs"
        elif time.perf_counter()>=deadline: reason="max-minutes"
        elif used_bytes>exp["max_storage_mb"]*2**20: reason="storage safeguard"
        if reason: skipped.append({"job":label,"reason":reason}); continue
        run_cfg=deepcopy(cfg); run_cfg["seed"]=seed; run_cfg["target"]["arrangement"]=arrangement
        temporal,provider=condition.split("_")
        if temporal not in ("stationary","evolving") or provider not in ("complete","restricted"): raise ValueError("unknown factorial condition")
        run_cfg["world"]["stationary"]=temporal=="stationary"; run_cfg["content"]["stationary"]=temporal=="stationary"; run_cfg["provider"]["mode"]=provider; run_cfg["policy"]["name"]=policy
        key=(seed,arrangement,temporal)
        try:
            if key not in worlds:
                if exp["world_path"]:
                    wp=Path(exp["world_path"]).resolve(); loaded=load_world(wp)
                    if len(exp["conditions"])!=1 or len(exp["arrangements"])!=1 or len(exp["seeds"])!=1: raise ValueError("exported-world compare requires one declared world condition/arrangement/seed")
                else:
                    from .generator import generate_world
                    wp=root/"worlds"/f"{seed}-{arrangement}-{temporal}"; save_world(generate_world(run_cfg),wp)
                    used_bytes+=sum(p.stat().st_size for p in wp.iterdir() if p.is_file())
                worlds[key]=wp
            result=run_one(worlds[key],run_cfg,root/"runs"/label,deadline=deadline,tags={"regime":exp["phase"]},storage_root=root,storage_base_bytes=used_bytes)
            used_bytes+=sum(p.stat().st_size for p in (root/"runs"/label).iterdir() if p.is_file())+2000
            result["run_path"]=f"runs/{label}"; summaries.append(result)
        except Exception as exc:
            failures.append({"job":label,"reason":f"{type(exc).__name__}: {exc}"}); append(root/"failures.jsonl",dict(job=label,traceback=traceback.format_exc()))
        append(root/"progress.jsonl",{"job":label,"completed":len(summaries),"failures":len(failures)})
    write_json(root/"summaries.json",summaries)
    if summaries: pq.write_table(pa.Table.from_pylist(summaries),root/"summaries.parquet")
    statistics=paired_summary(summaries,samples=cfg["evaluation"]["bootstrap_samples"],confidence=cfg["evaluation"]["confidence"],seed=cfg["seed"])
    write_json(root/"statistics.json",statistics)
    result=dict(schema_version=SCHEMA_VERSION,planned=len(jobs),executed=len(summaries),failed=failures,skipped=skipped,
        wall_seconds=time.perf_counter()-start,config_hash=config_hash(cfg),source=source_identity(),
        status="complete" if not failures and not skipped and all(r["status"]=="completed" for r in summaries) else "partial")
    write_json(root/"comparison.json",result); return result
