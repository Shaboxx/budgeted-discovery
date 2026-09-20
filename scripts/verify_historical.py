"""Check the released historical grid's completeness, pairing, and schema."""
from pathlib import Path
import itertools
import json


def validate(rows):
    policies = {"random", "graph", "query", "fixed", "round_robin"}
    conditions = {"evolving_" + c for c in ("complete", "capped", "ranked", "delayed", "stress")}
    arrangements = {"concentrated", "distributed", "weak"}
    expected = set(itertools.product(range(201, 221), arrangements, conditions, policies))
    observed = [(r["world_seed"], r["arrangement"], r["condition"], r["policy"]) for r in rows]
    if len(rows) != 1500 or len(set(observed)) != 1500 or set(observed) != expected:
        raise ValueError("Missing, extra, or duplicate experimental cells")
    if any(r["status"] != "completed" for r in rows):
        raise ValueError("Non-completed run in recorded main grid")
    if any(r["requests"] > 24 or r["cost"] > 32 or r["reference_relevant_accounts"] < 0 for r in rows):
        raise ValueError("Invalid budget or outcome")
    worlds = {}
    for row in rows:
        block = (row["world_seed"], row["arrangement"])
        worlds.setdefault(block, set()).add(row["world_content_hash"])
    if len(worlds) != 60 or any(len(values) != 1 for values in worlds.values()):
        raise ValueError("Unmatched world content within an experimental block")
    if len({next(iter(v)) for v in worlds.values()}) != 60:
        raise ValueError("World content repeated across distinct world blocks")
    return {"completed": len(rows), "strategies": sorted(policies), "world_seeds": 20, "worlds": 60}


if __name__ == "__main__":
    rows = json.loads(Path("data/historical/summaries.json").read_text())
    print(json.dumps(validate(rows), indent=2))
