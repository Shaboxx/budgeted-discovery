"""Post-run evaluator. Nothing in this module is passed to a policy."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .schemas import World, stream_seed

EVALUATOR_VERSION = "1.1.0"  # evaluate() unchanged; adds per-condition, interaction and arrangement summaries

def evaluate(world: World, visible: dict) -> dict:
    targets=set(world.truth["relevant_accounts"])
    initial=set(visible["initial_ids"])
    reached=set(visible["accounts"])-initial
    qualified=reached & targets
    labels=visible["labels"]
    regions=world.truth.get("regions",{})
    represented=sorted({str(regions[i]) for i in qualified if i in regions})
    eligible_regions={str(regions[i]) for i in targets-initial if i in regions}
    return dict(requests=visible["requests"],cost=visible["cost"],tick=visible["tick"],
        surfaced_accounts=len(reached),reference_relevant_accounts=len(qualified),
        initial_accounts=len(initial),initial_reference_relevant=len(initial&targets),
        eligible_reference_targets=len(targets-initial),posts=len(visible["posts"]),
        assessed_accounts=len(reached&set(labels)),pending_accounts=len(set(visible["pending_ids"])),
        assessed_reference_relevant=len(qualified&set(labels)),
        operational_positive_accounts=sum(labels.get(i,0)>=.5 for i in reached),
        target_regions_reached=len(represented),eligible_target_regions=len(eligible_regions),
        region_coverage=len(represented)/len(eligible_regions) if eligible_regions else None,
        reached_regions=represented)

def paired_summary(rows: list[dict], baseline: str="fixed", samples: int=300, confidence: float=.95, seed: int=41) -> dict:
    """Resample independent world seed blocks, retaining all conditions in a block.

    Only complete same-condition pairs enter a contrast. Missing pairs are counted.
    Two seeds can exercise the code but cannot support a precise uncertainty claim.
    """
    keys=("world_hash","seed","condition","arrangement","sparse_probes","use_motif","regime","acquisition_contract_hash")
    ref={}
    seen=set()
    for row in rows:
        key=tuple(row.get(k) for k in keys)
        identity=(row["policy"],key)
        if identity in seen: raise ValueError("duplicate ambiguous policy/world pair")
        seen.add(identity)
        if row["policy"]==baseline and row["status"]=="completed": ref[key]=row
    result=[]
    for policy in sorted({r["policy"] for r in rows}-{baseline}):
        blocks=defaultdict(list); missing=0
        for r in rows:
            if r["policy"]!=policy: continue
            other=ref.get(tuple(r.get(k) for k in keys))
            if r["status"]!="completed" or other is None: missing+=1; continue
            block=str(r.get("world_seed")) if r.get("world_seed") is not None else str(r.get("world_hash",r["seed"]))
            blocks[block].append(r["reference_relevant_accounts"]-other["reference_relevant_accounts"])
        values=np.array([np.mean(v) for _,v in sorted(blocks.items())],dtype=float)
        mean=float(values.mean()) if len(values) else None
        interval=None
        if len(values)>=2:
            rng=np.random.default_rng(stream_seed(seed,"statistical_analysis",policy))
            means=np.mean(rng.choice(values,size=(samples,len(values)),replace=True),axis=1)
            alpha=(1-confidence)/2
            interval=[float(x) for x in np.quantile(means,[alpha,1-alpha])]
        result.append(dict(policy=policy,baseline=baseline,mean_paired_difference=mean,
            seed_blocks=len(values),matched_pairs=sum(map(len,blocks.values())),missing_pairs=missing,
            interval=interval,confidence=confidence,
            interpretation="descriptive only; very few independent blocks" if len(values)<10 else "paired seed-cluster percentile bootstrap"))
    return {"evaluator_version":EVALUATOR_VERSION,"metric":"new unique reference-relevant accounts", "contrasts":result}


# ---- Evaluator 1.1.0 additions: per-condition contrasts, interaction, arrangement checks ----
# `evaluate()` and `paired_summary()` are unchanged; these functions add estimands for
# the primary (strategy x observation) and secondary (complementarity) hypotheses.

PAIR_KEYS=("world_hash","seed","condition","arrangement","sparse_probes","use_motif","regime","acquisition_contract_hash")

def _block_of(row: dict) -> str:
    return str(row.get("world_seed")) if row.get("world_seed") is not None else str(row.get("world_hash",row["seed"]))

def _interval(values: list[float], samples: int, confidence: float, seed: int, *keys) -> list[float] | None:
    if len(values)<2: return None
    rng=np.random.default_rng(stream_seed(seed,"statistical_analysis",*keys))
    arr=np.asarray(values,dtype=float)
    means=np.mean(rng.choice(arr,size=(samples,len(arr)),replace=True),axis=1)
    alpha=(1-confidence)/2
    return [float(x) for x in np.quantile(means,[alpha,1-alpha])]

def _describe(values: dict, samples: int, confidence: float, seed: int, *keys) -> dict:
    ordered=[values[k] for k in sorted(values)]
    return dict(mean=float(np.mean(ordered)) if ordered else None,interval=_interval(ordered,samples,confidence,seed,*keys),
        seed_blocks=len(ordered),positive=sum(v>0 for v in ordered),zero=sum(v==0 for v in ordered),negative=sum(v<0 for v in ordered))

def condition_summary(rows: list[dict], baseline: str="fixed", samples: int=300, confidence: float=.95, seed: int=41, reference_condition: str|None=None) -> dict:
    """Paired policy-minus-baseline contrasts within each observation condition, and the
    interaction (contrast in condition minus contrast in the reference condition), both
    aggregated over independent world-seed blocks. Only complete same-condition pairs enter."""
    ref={}; seen=set()
    for row in rows:
        identity=(row["policy"],tuple(row.get(k) for k in PAIR_KEYS))
        if identity in seen: raise ValueError("duplicate ambiguous policy/world pair")
        seen.add(identity)
        if row["policy"]==baseline and row["status"]=="completed": ref[identity[1]]=row
    conditions=sorted({r["condition"] for r in rows})
    if reference_condition is None and conditions:
        reference_condition=next((c for c in conditions if c.endswith("_complete")),conditions[0])
    contrasts=[]
    for policy in sorted({r["policy"] for r in rows}-{baseline}):
        by_condition={}; block_values={}
        for cond in conditions:
            blocks=defaultdict(list); missing=0
            for r in rows:
                if r["policy"]!=policy or r["condition"]!=cond: continue
                other=ref.get(tuple(r.get(k) for k in PAIR_KEYS))
                if r["status"]!="completed" or other is None: missing+=1; continue
                blocks[_block_of(r)].append(r["reference_relevant_accounts"]-other["reference_relevant_accounts"])
            block_values[cond]={b:float(np.mean(v)) for b,v in blocks.items()}
            by_condition[cond]={**_describe(block_values[cond],samples,confidence,seed,policy,cond),"missing_pairs":missing}
        interactions={}
        base_blocks=block_values.get(reference_condition,{})
        for cond in conditions:
            if cond==reference_condition: continue
            common={b:block_values[cond][b]-base_blocks[b] for b in block_values[cond] if b in base_blocks}
            interactions[cond]=_describe(common,samples,confidence,seed,policy,cond,"interaction")
        contrasts.append(dict(policy=policy,baseline=baseline,reference_condition=reference_condition,by_condition=by_condition,interaction_vs_reference=interactions))
    return {"evaluator_version":EVALUATOR_VERSION,"metric":"new unique reference-relevant accounts",
        "estimand":"per-condition paired difference (policy - baseline) averaged within world-seed blocks; interaction = difference of that contrast between a condition and the reference condition",
        "reference_condition":reference_condition,"contrasts":contrasts,
        "interpretation":"descriptive; intervals resample independent world-seed blocks and are informative only with many blocks"}

def arrangement_summary(rows: list[dict], mixture: str="fixed", components: tuple[str,...]=("graph","query"), samples: int=300, confidence: float=.95, seed: int=41) -> dict:
    """Secondary hypothesis check: within each target arrangement, mixture minus each
    component and mixture minus the better component, paired per world/condition cell and
    averaged within world-seed blocks."""
    cells=defaultdict(dict)
    for r in rows:
        if r["status"]!="completed": continue
        cells[(r["arrangement"],_block_of(r),tuple(r.get(k) for k in PAIR_KEYS if k!="condition"),r["condition"])][r["policy"]]=r["reference_relevant_accounts"]
    result={}
    for arrangement in sorted({k[0] for k in cells}):
        per_block=defaultdict(lambda: defaultdict(list)); incomplete=0
        for key,values in cells.items():
            if key[0]!=arrangement: continue
            if mixture not in values or any(c not in values for c in components): incomplete+=1; continue
            block=key[1]
            for c in components: per_block[f"{mixture}_minus_{c}"][block].append(values[mixture]-values[c])
            per_block[f"{mixture}_minus_best_component"][block].append(values[mixture]-max(values[c] for c in components))
        result[arrangement]={name:_describe({b:float(np.mean(v)) for b,v in blocks.items()},samples,confidence,seed,arrangement,name) for name,blocks in per_block.items()}
        result[arrangement]["incomplete_cells"]=incomplete
    return {"evaluator_version":EVALUATOR_VERSION,"mixture":mixture,"components":list(components),"by_arrangement":result,
        "interpretation":"descriptive complementarity check; a positive mixture-minus-best-component value in a block means the mixture found more than either component alone in that world"}

def full_statistics(rows: list[dict], baseline: str="fixed", samples: int=300, confidence: float=.95, seed: int=41) -> dict:
    """Backward-compatible statistics record: the pooled paired contrasts plus the
    per-condition/interaction and arrangement summaries."""
    pooled=paired_summary(rows,baseline=baseline,samples=samples,confidence=confidence,seed=seed)
    pooled["by_condition"]=condition_summary(rows,baseline=baseline,samples=samples,confidence=confidence,seed=seed)
    pooled["by_arrangement"]=arrangement_summary(rows,mixture=baseline,samples=samples,confidence=confidence,seed=seed)
    return pooled
