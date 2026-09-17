"""Command-line acceptance, bounded experiments, portable replay and reports."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import importlib
import importlib.metadata
import json
import sys
from .config import load_config, resolve
from .runner import compare, run_one, replay, write_json, dependency_versions, source_identity
from .world import load_world, save_world, validate_world, import_events
from .reporting import report

def doctor(portable=False):
    required=["numpy","networkx","gymnasium","bm25s","river","pyarrow","duckdb","psutil","yaml"]
    if not portable: required += ["graph_tool","tadcsbm","ndlib.models.dynamic","dynetx"]
    checks={}
    for module in required:
        try: importlib.import_module(module); checks[module]="imported"
        except Exception as exc: checks[module]=f"FAILED {type(exc).__name__}: {exc}"
    result={"ok":all(v=="imported" for v in checks.values()),"mode":"portable-runtime" if portable else "full-generation", "checks":checks,"versions":dependency_versions()}
    if not portable and checks.get("graph_tool")=="imported": result["versions"]["graph-tool"]=importlib.import_module("graph_tool").__version__
    return result

def generate(cfg,output):
    from .generator import generate_world
    world=generate_world(cfg); save_world(world,output)
    return {"path":str(output),**validate_world(world)}

def apply_limits(cfg,args):
    for field in ("max_runs","max_minutes","max_storage_mb"):
        value=getattr(args,field,None)
        if value is not None:
            if value<=0: raise ValueError(f"{field} must be positive")
            cfg["experiment"][field]=value
    if getattr(args,"fixed_from",None):
        tuned=json.loads(Path(args.fixed_from).read_text())
        if set(cfg["experiment"]["seeds"]) & set(tuned["development_seeds"]): raise ValueError("test seeds overlap mixture development")
        current=source_identity()["files"]
        for module in ("policies.py","generator.py","evaluation.py"):
            key="frontier_bench/"+module
            if current[key]!=tuned["source"]["files"][key]: raise ValueError("development policy/generator/evaluator changed; retune before comparing")
        cfg["policy"]["graph_weight"]=tuned["graph_weight"]
        cfg["experiment"]["fixed_selection"]={k:tuned[k] for k in ("graph_weight","development_seeds","objective","source")}
    return cfg

def smoke(output,world_path=None):
    check=doctor(portable=bool(world_path))
    if not check["ok"]: raise RuntimeError(json.dumps(check))
    cfg=resolve()
    if world_path:
        world=load_world(world_path); cfg=resolve(world.metadata.get("config",{})); cfg["experiment"]["world_path"]=str(Path(world_path).resolve())
    cfg["experiment"].update(output=str(output),seeds=[cfg["seed"]],arrangements=[cfg["target"]["arrangement"]],
        conditions=[f'{"stationary" if cfg["world"]["stationary"] else "evolving"}_{cfg["provider"]["mode"]}'],max_runs=7,max_minutes=5.,max_storage_mb=200,phase="acceptance")
    result=compare(cfg)
    write_json(Path(output)/"doctor.json",check)
    from .measurement import diagnostic_suite
    measurement=diagnostic_suite(Path(output)/"measurement")
    summaries=json.loads((Path(output)/"summaries.json").read_text())
    verified=[replay(Path(output)/r["run_path"]) for r in summaries if r["status"]=="completed"]
    report_path=report(output)
    if result["status"]!="complete" or len(verified)!=7: raise RuntimeError("Acceptance incomplete; inspect preserved artifacts")
    return {"verified":True,"comparison":result,"replays":len(verified),"measurement_false_alerts":measurement["false_alerts"],"report":str(report_path)}

def tune(cfg,output):
    if cfg["experiment"]["phase"]!="development": raise ValueError("tuning must use declared development worlds")
    candidates=[]; total=len(cfg["experiment"]["fixed_weights"])*len(cfg["experiment"]["seeds"])*len(cfg["experiment"]["conditions"])*len(cfg["experiment"]["arrangements"])
    if total>cfg["experiment"]["max_runs"]: raise ValueError("development grid exceeds max-runs")
    import time
    start=time.perf_counter()
    for weight in cfg["experiment"]["fixed_weights"]:
        trial=deepcopy(cfg); trial["policy"]["graph_weight"]=weight; trial["experiment"]["policies"]=["fixed"]
        remaining=cfg["experiment"]["max_minutes"]-(time.perf_counter()-start)/60
        if remaining<=0: raise RuntimeError("development watchdog reached; incomplete trials retained")
        trial["experiment"]["max_minutes"]=remaining
        destination=Path(output)/f"weight-{weight:g}"
        result=compare(trial,destination)
        if result["status"]!="complete": raise RuntimeError("Incomplete development grid; no winner selected")
        rows=json.loads((destination/"summaries.json").read_text())
        candidates.append({"graph_weight":weight,"mean_reference_relevant":sum(r["reference_relevant_accounts"] for r in rows)/len(rows),"runs":len(rows)})
    winner=min(candidates,key=lambda r:(-r["mean_reference_relevant"],r["graph_weight"]))
    selection={**winner,"candidates":candidates,"development_seeds":cfg["experiment"]["seeds"],"objective":"mean new reference-relevant accounts; tie choose smaller graph weight","config":cfg,"source":source_identity(),"frozen":True}
    write_json(Path(output)/"selected-mixture.json",selection); return selection

def main(argv=None):
    parser=argparse.ArgumentParser(prog="frontier-bench",description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor"); p.add_argument("--portable",action="store_true")
    p=sub.add_parser("generate"); p.add_argument("--config",required=True); p.add_argument("--output",default="runs/generated-world")
    p=sub.add_parser("validate-world"); p.add_argument("--world",required=True)
    for command in ("run","compare","tune"):
        p=sub.add_parser(command); p.add_argument("--config",required=True); p.add_argument("--output"); p.add_argument("--max-runs",type=int); p.add_argument("--max-minutes",type=float); p.add_argument("--max-storage-mb",type=int); p.add_argument("--fixed-from")
    p=sub.add_parser("replay"); p.add_argument("--run",required=True)
    p=sub.add_parser("report"); p.add_argument("--runs",required=True)
    p=sub.add_parser("smoke"); p.add_argument("--output",default="runs/smoke"); p.add_argument("--world")
    p=sub.add_parser("measurement"); p.add_argument("--output",default="runs/measurement")
    p=sub.add_parser("import-events")
    for name in ("accounts","edges","posts","targets","output"): p.add_argument("--"+name,required=True)
    p.add_argument("--ticks",required=True,type=int)
    args=parser.parse_args(argv)
    try:
        if args.command=="doctor":
            result=doctor(args.portable)
            if not result["ok"]: print(json.dumps(result,indent=2)); return 1
        elif args.command=="generate": result=generate(load_config(args.config),args.output)
        elif args.command=="validate-world": result=validate_world(load_world(args.world))
        elif args.command in ("run","compare","tune"):
            cfg=apply_limits(load_config(args.config),args); destination=args.output or cfg["experiment"]["output"]
            if args.command=="tune": result=tune(cfg,destination)
            elif args.command=="compare": result=compare(cfg,destination); report(destination)
            else:
                world_path=cfg["experiment"]["world_path"]
                if not world_path: world_path=str(Path(destination)/"world"); generate(cfg,world_path)
                result=run_one(world_path,cfg,Path(destination)/"run"); report(Path(destination)/"run")
        elif args.command=="replay": result=replay(args.run)
        elif args.command=="report": result={"report":str(report(args.runs))}
        elif args.command=="smoke": result=smoke(args.output,args.world)
        elif args.command=="measurement":
            from .measurement import diagnostic_suite
            result=diagnostic_suite(args.output)
        elif args.command=="import-events":
            world=import_events(args.accounts,args.edges,args.posts,args.targets,args.ticks); save_world(world,args.output); result=validate_world(world)
        print(json.dumps(result,indent=2))
        return 2 if result.get("status") in ("partial","failed","interrupted","truncated") else 0
    except Exception as exc:
        print(f"ERROR {type(exc).__name__}: {exc}",file=sys.stderr); return 1

if __name__=="__main__": raise SystemExit(main())
