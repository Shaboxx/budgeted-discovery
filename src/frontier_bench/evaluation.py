"""Post-run evaluator. Nothing in this module is passed to a policy."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from .schemas import World, stream_seed

EVALUATOR_VERSION = "1.0.0"

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
