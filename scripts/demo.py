"""Run five budgeted acquisition strategies offline on one exported synthetic world."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier_bench.config import resolve
from frontier_bench.runner import compare, replay
from frontier_bench.reporting import report
from frontier_bench.world import load_world

POLICIES = ["random", "graph", "query", "fixed", "round_robin"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args(argv)
    world_path = Path("data/example-world")
    world = load_world(world_path)
    if args.output.exists():
        parser.error("Choose a fresh output directory; recorded runs are never overwritten.")
    cfg = resolve(world.metadata["config"])
    cfg["policy"].update(graph_weight=0.75, graph_operations=["neighbors"], rotation=["graph", "query"])
    cfg["feedback"].update(mode="operational", proxy="profile_only", delay=1, noise=0.1)
    cfg["provider"].update(mode="restricted", page_size=5, failure_probability=0.05,
                           ranking_noise=0.05, recency_weight=0.15, availability_lag=1)
    cfg["experiment"].update(world_path=world_path.as_posix(), policies=POLICIES,
        seeds=[201], arrangements=["concentrated"], conditions=["evolving_stress"],
        observation_conditions={"stress": {"mode": "restricted", "page_size": 5,
            "failure_probability": 0.05, "ranking_noise": 0.05,
            "recency_weight": 0.15, "availability_lag": 1}},
        max_runs=5, max_minutes=2, max_storage_mb=40, phase="public-demo")
    result = compare(cfg, args.output)
    if result["status"] != "complete":
        raise RuntimeError("Demo incomplete; inspect the preserved comparison.json.")
    rows = json.loads((args.output / "summaries.json").read_text())
    verified = [replay(args.output / row["run_path"]) for row in rows]
    assert len(verified) == 5 and all(r["verified"] for r in verified)
    report_path = report(args.output)
    print("Five strategies completed; five event-log replays verified.")
    print("policy          relevant  requests  cost")
    for row in rows:
        print(f"{row['policy']:15} {row['reference_relevant_accounts']:8} {row['requests']:9} {row['cost']:5.1f}")
    print(f"Offline report: {report_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
